// Copyright 2020 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#define BYTEPS_DEBUG

#include "sparse.h"
#include "sparse_dense.h"
#include "sparse.cuh"
#include <iostream>

namespace byteps {
namespace sparse {

namespace{
  inline int local_session_id_to_gpu_id(int local_session_id, int local_gpu_size){
    return local_session_id % local_gpu_size;
  }
} // namespace

void bytepsSparseInit(std::vector<void*>& embedBuffers,
                      std::vector<void*>& embedGradBuffers,
                      std::vector<void*>& denseBuffers,
                      std::vector<size_t>& embedBufferLens,
                      size_t denseBufferLen) {
  LocalRankCallback scatter_followup_fwd = [](int request_sess_id, int storage_device_id) {};
  LocalRankCallback scatter_followup_bwd = [](int request_sess_id, int storage_device_id) {};
  LocalRankCallback gather_precondition = [](int request_sess_id, int storage_device_id) {};

  std::vector<std::vector<void*>> dataSendBuf;
  std::vector<std::vector<void*>> dataRecvBuf;

  bytepsSparseInitWithHook( embedBuffers, 
                            embedGradBuffers,
                            denseBuffers, 
                            embedBufferLens, 
                            denseBufferLen,
                            dataSendBuf,
                            dataRecvBuf,
                            scatter_followup_fwd,
                            scatter_followup_bwd,
                            gather_precondition);               
}

/*
  embedBuffers: the addresses of all embedding buffers (could have variable length)
  embedGradBuffers: the addresses of embedding grad buffers (could have variable length
                    same with embedBuffers elementwise)
  denseBuffers: the addresses of all dense buffers (the length should be identical)
  embedBufferLens: the length of the embedding buffers (could have variable length)
  size: the length of a dense buffer (in bytes), it is equivalent for all GPUs
  scatter_followup: the callback function that takes a local rank as arg and will be executed
                    AFTER scatter is performed on each rank
  gather_precondition: the callback function that takes a local rank as arg and will be
                       executed BEFORE gather is performed on that rank, it should block
                       until that rank is safe to gather

  Important Note on Number of Buffers:
    1. embedBuffers, embedGradBuffers must have the same len as local GPUs, they each map to
       a single thread on GPU performing embedding fwd/bwd computation.
    2. denseBuffers must have len equal or more than embedBuffers, as more than one computation
       session can map to a same GPU, with gpu_idx = denseBuffers_idx // local_gpu_num.
    3. length rules:
      3.1 embedBuffers / embedGradBuffers has size of local_num_gpu, with each buffer in it a 
          size of global_session_size * embedded_vec_len * sizeof(float).
      3.2 denseBuffer has size of local_session_num, with each a size equal to 
          denseBufferLen = Sum(embedded_vec_len) * sizeof(float).
    4. GPU placement rule: the i-th session(denseBuffer) will be placed on the GPU i % gpu_num.
 */
void bytepsSparseInitWithHook(std::vector<void*>& embedBuffers, 
                              std::vector<void*>& embedGradBuffers,
                              std::vector<void*>& denseBuffers, 
                              std::vector<size_t>& embedBufferLens, 
                              size_t denseBufferLen,
                              std::vector<std::vector<void*>>& dataCPUSendBuffers,
                              std::vector<std::vector<void*>>& dataGPURecvBuffers,
                              LocalRankCallback scatter_followup_fwd,
                              LocalRankCallback scatter_followup_bwd,
                              LocalRankCallback gather_precondition) {
  BytePSSparseCommon::Init();
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  auto globalSize = localSize * workerNum;
  auto localSessionSize = denseBuffers.size();
  auto globalSessionSize = localSessionSize * workerNum;

  int comm_workerID;
  _comm_socket_worker = std::shared_ptr<BytePSCommSocket>(
      new BytePSCommSocket(false /* server_mode */));
  LOG(INFO) << "Byteps worker successfully init CommSocket for signaling with server.";
  _compute_hook_manager = std::shared_ptr<ComputeHookManager>(
    new ComputeHookManager(scatter_followup_fwd, scatter_followup_bwd, gather_precondition, _comm_socket_worker));
  LOG(INFO) << "Byteps worker successfully init compute hook.";

  _comm_socket_worker->init(&comm_workerID, _compute_hook_manager);

  // We now allow denseBuffers to have different size than Embed/GradBuffer.
  CHECK_EQ(embedGradBuffers.size(), embedBuffers.size());
  CHECK_EQ(embedBufferLens.size(), embedBuffers.size());
  CHECK_GE(denseBuffers.size(), embedBuffers.size());

  CHECK_EQ(dataGPURecvBuffers.size(), localSize);

  for (auto& remote_buf : dataGPURecvBuffers) CHECK_EQ(remote_buf.size(), globalSessionSize);

  // Init IPC stuff
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  CHECK_EQ(createCudaIpcSharedMemory(bpsCudaIpcShmName, sizeof(*shm), &info), 0);
  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  for (int i = 0; i < localSize; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // CUDA IPC is only supported on devices with unified addressing
    CHECK(prop.unifiedAddressing)
        << "Device " << i << " does not support unified addressing.";

    shm->devices[shm->nprocesses++] = i;
    CHECK_GT(MAX_CUDA_DEVICES, shm->nprocesses);
  }
  CHECK(shm->nprocesses > 0) 
      << "No cuda device suppported";
  CHECK_EQ(shm->nprocesses, embedBuffers.size())
      << "Shared memory processes: " << shm->nprocesses 
      << ", send buffers: " << embedBuffers.size();

  _embedBuffers.assign(embedBuffers.begin(), embedBuffers.end());
  _embedGradBuffers.assign(embedGradBuffers.begin(), embedGradBuffers.end());
  _denseBuffers.assign(denseBuffers.begin(), denseBuffers.end());

  _localEmbedBufLens.resize(localSize);
  _globalEmbedBufLens.resize(workerNum, std::vector<size_t>(localSize));
  _globalTotalEmbedBufLens.resize(workerNum, 0);

  // Allocate memory for each process and fill 
  // the shared memory buffer with the IPC handles
  for (size_t i = 0; i < shm->nprocesses; i++) {
    CUDA_CALL(cudaSetDevice(
        shm->devices[i]));
    CUDA_CALL(cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->embedMemHandle[i], embedBuffers[i]));
      CUDA_CALL(cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->embedGradMemHandle[i], embedGradBuffers[i]));
      for (int j=0; j < globalSessionSize; j++){
        CUDA_CALL(cudaIpcGetMemHandle(
            (cudaIpcMemHandle_t *)&shm->dataMemHandle[i][j], dataGPURecvBuffers[i][j]));
    }

  
    shm->embedBufferLength[i] = embedBufferLens[i];
    // Store the buffers 
    _localEmbedBufLens[i] = embedBufferLens[i]; // local buffer length
  }
  _denseBufferLen = denseBufferLen;
  shm->denseBufferLength = denseBufferLen;

#ifdef BYTEPS_DEBUG
  // For debug: print _localEmbedBufLens
  std::cout << "_localEmbedBufLens:" << std::endl;
  for (auto len : _localEmbedBufLens) 
    std::cout << len << " ";
  std::cout << std::endl;
#endif

  for (int i = 0; i < localSize; i++) {
    _globalEmbedBufLens[workerID][i] = _localEmbedBufLens[i];
  }
  
  // The followings are for the global coordination of 
  // the embedding buffer length, which is equivalent to all-gather 
  auto ps = BytePSSparseCommon::GetPS();
  if (BytePSSparseCommon::IsDistributed()) {
    CHECK(ps); // must init the pslite instance before
    
    // keys
    std::vector<ps::Key> pskeys(workerNum);
    std::vector<ps::SArray<ps::Key>> keys_array; 

    // lens
    std::vector<int> pslens(workerNum);
    std::vector<ps::SArray<int>> lens_array; 

    // vals
    std::vector<ps::SArray<char>> vals_array; 

    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    for (int i = 0; i < workerNum; i++) {
      ps::Key key = i;
      int server = i;
      
      // keys 
      pskeys[i] = krs[server].begin() + key;
      ps::SArray<ps::Key> keys;
      keys.reset(&pskeys[i], 1, [](void *){});
      keys_array.push_back(keys);
      
      // lens 
      pslens[i] = sizeof(size_t) * localSize;
      ps::SArray<int> lens;
      lens.reset(&pslens[i], 1, [](void *){});
      lens_array.push_back(lens);

      // vals 
      ps::SArray<char> vals;
      vals.reset((char*)_globalEmbedBufLens[i].data(), localSize * sizeof(size_t), [](void *){});
      vals_array.push_back(vals);
    }

    // Push once to the associated server
    {
      int server = workerID;
      auto keys = keys_array[server];
      auto vals = vals_array[server];
      auto lens = lens_array[server];
      ps->Wait(ps->ZPush(keys, vals, lens));
    }

    ps::Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);

    // Pull the embedding buffer length of other workers
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue; // skip myself
      int server = i;
      auto keys = keys_array[server];
      auto vals = vals_array[server];
      auto lens = lens_array[server];
      ps->Wait(ps->ZPull(keys, &vals, &lens));
    }
  } // BytePSSparseCommon::IsDistributed()

  for (int wid = 0; wid < workerNum; wid++) {
    for (int gpu = 0; gpu < localSize; gpu++) {
      _globalTotalEmbedBufLens[wid] += _globalEmbedBufLens[wid][gpu];
    }
  }

#ifdef BYTEPS_DEBUG
  // For debug: print _globalEmbedBufLens
  std::cout << "_globalEmbedBufLens:" << std::endl;
  for (auto vec : _globalEmbedBufLens) {
    for (auto len : vec) {
      std::cout << len << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // For debug: print _globalTotalEmbedBufLens
  std::cout << "_globalTotalEmbedBufLens:" << std::endl;
  for (auto len : _globalTotalEmbedBufLens) {
    std::cout << len << " ";
  }
  std::cout << std::endl;
#endif 

  // Check the buffer size 
  size_t accmul = 0;
  for (auto len : _globalTotalEmbedBufLens) accmul += len / globalSessionSize;
  CHECK_EQ(accmul, _denseBufferLen) << accmul << " " << _denseBufferLen;

  // Calc the global offset for the communication buffers
  // i.e. worker i's first gpu's embedding's start location in a denseBuffer.
  size_t global_offset = 0;
  for (int id = 0; id < workerID; id++) {
    global_offset += _globalTotalEmbedBufLens[id] / globalSessionSize;
  }

  // Prepare gossip-gather communication
  _local_gather_comms.resize(localSessionSize);
  for (int i = 0; i < localSessionSize; i++) {
    std::vector<float*> srcs(localSize);
    std::vector<size_t> srcs_lens(localSize);
    std::vector<size_t> send_counts(localSize);

    for (int j = 0; j < localSize; j++) {
      srcs[j] = 
          (float*)_embedBuffers[j] + 
          _localEmbedBufLens[j] / globalSessionSize * (i + localSessionSize * workerID);

      srcs_lens[j] = 
          _localEmbedBufLens[j] / globalSessionSize * 
          (globalSessionSize - (i + localSessionSize * workerID));

      send_counts[j] = 
          _localEmbedBufLens[j] / globalSessionSize;
    }
    float* dst = (float *)_denseBuffers[i] + global_offset;
    size_t dst_len = _globalTotalEmbedBufLens[workerID] / globalSessionSize;

    std::string planfile_name("gather_plan_");
    int gpu_idx = local_session_id_to_gpu_id(i, localSize);
    planfile_name += std::to_string(gpu_idx) + std::string(".json");
    _local_gather_comms[i] = std::unique_ptr<LocalGatherComm>(new LocalGatherComm(planfile_name, localSize, srcs, srcs_lens, send_counts, dst, dst_len));
  }

  

  // Prepare gossip-scatter communication
  _local_scatter_comms.resize(localSessionSize);
  for (int i = 0; i < localSessionSize; i++) {
    float* src = (float *)_denseBuffers[i] + global_offset;
    size_t src_len = _globalTotalEmbedBufLens[workerID] / globalSessionSize;
    std::vector<float*> dsts(localSize);
    std::vector<size_t> dsts_lens(localSize);
    std::vector<size_t> send_counts(localSize);
    for (int j = 0; j < localSize; j++) {
      dsts[j] = 
          (float*)_embedGradBuffers[j] + 
          _localEmbedBufLens[j] / globalSessionSize * (i + localSessionSize * workerID);

      dsts_lens[j] = 
          _localEmbedBufLens[j] / globalSessionSize * 
          (globalSessionSize - (i + localSessionSize * workerID));

      send_counts[j] = 
          _localEmbedBufLens[j] / globalSessionSize;
    }

    std::string planfile_name("scatter_plan_");
    int gpu_idx = local_session_id_to_gpu_id(i, localSize);
    planfile_name += std::to_string(gpu_idx) + std::string(".json");
    _local_scatter_comms[i] = std::unique_ptr<LocalScatterComm>(
       new LocalScatterComm(planfile_name, localSize, src, src_len, send_counts, dsts, dsts_lens));
  }
  
  if (BytePSSparseCommon::IsDistributed()) {
    // Prepare distributed gather communication
    _dist_gather_comms.resize(localSessionSize);
    for (int i = 0; i < localSessionSize; i++) {
      auto ps = BytePSSparseCommon::GetPS();
      _dist_gather_comms[i] = std::unique_ptr<DistGatherComm>(new DistGatherComm(ps, _globalEmbedBufLens, 
        _denseBuffers[i], _denseBufferLen, i, localSize, localSessionSize, workerID, workerNum));
    }
    // Prepare distributed scatter communication
    _dist_scatter_comms.resize(localSessionSize);
    for (int i = 0; i < localSessionSize; i++) {
      auto ps = BytePSSparseCommon::GetPS();
      _dist_scatter_comms[i] = std::unique_ptr<DistScatterComm>(new DistScatterComm(ps, _globalEmbedBufLens, 
        _denseBuffers[i], _denseBufferLen, i, localSize, localSessionSize, workerID, workerNum));
    }
  }

  if (BytePSSparseCommon::IsDistributed()) {
    // Prepare distributed data scatter communication
  
      _dist_data_scatter_comms = std::unique_ptr<DistScatterDataComm>(new DistScatterDataComm(ps, localSize, localSessionSize, workerID, workerNum));
  }

  LOG(INFO) << "Init BytePS Sparse for embedding layers completed";
}

extern "C" void bytepsSparseInitDensePerGPU(int device_id /* starts with 0 */,
                                            void* denseDeltaBeforeReduceBuffer,
                                            void* denseDeltaAfterReduceBuffer,
                                            int sizeDenseDelta) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  CHECK_LT(device_id, localSize) << "Device id must be within local gpu size.";

  _denseDeltaBufferLength = sizeDenseDelta;
  
  auto ps = BytePSSparseCommon::GetPS();
  _dense_reduce_comms.push_back(
      std::unique_ptr<DenseReduceComm>( new DenseReduceComm(
        ps, 
        sizeDenseDelta, 
        denseDeltaBeforeReduceBuffer,
        denseDeltaAfterReduceBuffer,
        device_id,
        localSize, 
        workerID, 
        workerNum
      ))
  );
  LOG(INFO) << "Successfully Init BytePS Sparse for dense layers: Device " << device_id;
}

void bytepsSparseShutdown() {
}


void bytepsGatherExecAsync(int local_sess_rank, cudaStream_t stream) {
  // Gather from local peer GPUs on the same worker
  _local_gather_comms[local_sess_rank]->ExecAsync();
  
  // Gather from distributed peer GPUs on other workers
  if (BytePSSparseCommon::IsDistributed()) {
    _dist_gather_comms[local_sess_rank]->ExecAsync();
  }
}

void bytepsdataScatterExecAsync(const std::vector<std::vector<void*>>& src, 
const std::vector<std::vector<int>> & buf_size); {

  if (BytePSSparseCommon::IsDistributed()) {
    _dist_data_scatter_comms->ExecAsync( src, buf_size );
  }
}

void bytepsScatterExecAsync(int local_sess_rank, cudaStream_t stream) {
  // Scatter to local peer GPUs on the same worker
  _local_scatter_comms[local_sess_rank]->ExecAsync();
  
  // Scatter to distributed peer GPUs on other workers
  if (BytePSSparseCommon::IsDistributed()) {
    _dist_scatter_comms[local_sess_rank]->ExecAsync();
  }
}


// TODO (chengyu.dai): Add Broadcast for initializing the latestBuffer.
void bytepsDenseReduceExecAsync(int local_sess_rank, cudaStream_t stream) {
  _dense_reduce_comms[local_sess_rank]->ExecAsync();
}

//Add Broadcast for initializing the latestBuffer.
void bytepsDenseBCastExec(int local_rank, cudaStream_t stream) {
  _dense_reduce_comms[local_rank]->ExecBCast();
}

void bytepsSynchronize(int local_sess_rank, cudaStream_t stream, OP op) { 
  switch (op) {
    case GATHER: {
      _local_gather_comms[local_sess_rank]->Sync();
      if (BytePSSparseCommon::IsDistributed()) {
        _dist_gather_comms[local_sess_rank]->Sync();
      }
    } break;
    case SCATTER: {
      _local_scatter_comms[local_sess_rank]->Sync();
      if (BytePSSparseCommon::IsDistributed()) {
        _dist_scatter_comms[local_sess_rank]->Sync();
      }
    } break;
    default:
      CHECK(0) << "unrecognized operation: " << op;
  }
  CUDA_CALL(cudaStreamSynchronize(stream));
}


// TODO: should merge this with bytepsSynchronize
void bytepsDenseSynchronize(int local_sess_rank, cudaStream_t stream) {
  _dense_reduce_comms[local_sess_rank]->Sync();
}

} // namespace sparse
} // namespace byteps 
