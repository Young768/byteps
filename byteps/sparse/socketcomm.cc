// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include <cerrno>
#include <cstring>

#include "socketcomm.h"
#include "compute_hook.h"

namespace byteps {
namespace sparse {

// Copy constructor that provides the option to reconfigure members.
// The ranks in members always use local_rank, regardless that the members
// may be a subset of all local ranks.
BytePSCommSocket::BytePSCommSocket(std::shared_ptr<BytePSSparseComm> comm,
                                   const std::string& path_suffix) {
  std::shared_ptr<BytePSCommSocket> sock_comm =
      std::static_pointer_cast<BytePSCommSocket>(comm);
  // TODO: use private members directly
  _worker_id = sock_comm->getWorkerID();
  _send_path = sock_comm->getSendPath() + path_suffix;
  _recv_path = sock_comm->getRecvPath() + path_suffix;
  _send_fd = initSocket(_worker_id, _send_path);
  _recv_fd = initSocket(_worker_id, _recv_path);

  _listen_thread =
        new std::thread(&BytePSCommSocket::startListenThread, this);

  LOG(INFO) << "all sockets create successfully";
}

void BytePSCommSocket::init(int* worker_id, std::shared_ptr<ComputeHookManager> compute_hook) {
  LOG(INFO) << "Using Communicator=Socket";
  if (! _server_mode){
    CHECK(compute_hook != nullptr) << "Worker mode BytePSCommSocket init must provide a compute hook.";
    _compute_hook = compute_hook;
  } else{
    _compute_hook = nullptr;
  }

  // We should init size, etc. using getenv
  // do env check
  CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
  CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

  *worker_id = atoi(getenv("DMLC_WORKER_ID"));

  // We assume there is only one byteps worker per node in sparse mode
  // server's worker_id = DMLC_WORKER_ID + 1, worker's worker_id = DMLC_WORKER_ID
  _worker_id = *worker_id + _server_mode;

  if (getenv("BYTEPS_SOCKET_PATH")) {
    _send_path = std::string(getenv("BYTEPS_SOCKET_PATH")) +
                 std::string("/socket_send_");
    _recv_path = std::string(getenv("BYTEPS_SOCKET_PATH")) +
                 std::string("/socket_recv_");
  } else {
    _send_path = std::string(DEFAULT_BASE_SOCKET_PATH_SEND);
    _recv_path = std::string(DEFAULT_BASE_SOCKET_PATH_RECV);
  }

  _send_fd = initSocket(_worker_id, _send_path);
  _recv_fd = initSocket(_worker_id, _recv_path);

  _listen_thread =
        new std::thread(&BytePSCommSocket::startListenThread, this);

  LOG(INFO) << "all sockets create successfully";
}

int BytePSCommSocket::initSocket(int rank, const std::string& path) {
  int fd;
  // init the socket
  fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK_GE(fd, 0) << "recv socket create failed";

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));

  // TODO: use absolute unique socket path name (consider multi-tenancy)
  std::string fd_path(path);
  fd_path +=
      std::to_string(rank);  // should use the rank id to guarantee no conflict

  // filling addr information
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, fd_path.c_str(), sizeof(addr.sun_path) - 1);

  // before bind, clear the path first
  unlink(fd_path.c_str());

  // set recv timeout value for socket
  struct timeval tv;
  tv.tv_sec = 3; // in seconds
  tv.tv_usec = 0;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

  // bind the socket to addr
  int ret = bind(fd, (struct sockaddr*)&addr, sizeof(addr));
  CHECK_GE(ret, 0) << fd_path << " bind failed: " << strerror(errno);

  LOG(INFO) << "Init socket at " << fd_path;

  return fd;
}

void BytePSCommSocket::startListenThread() {  
  // each worker starts this in background thread
  LOG(INFO) << "Listening on socket " << _worker_id;
  char buffer[MAX_LINE];
  while (true) {
    int rc;
    while (true) {
      rc = recv(_recv_fd, buffer, sizeof(buffer), MSG_WAITALL);
      if (rc < 0 && errno == EINTR) continue;
      if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { // timeout
        //if (BytePSGlobal::ShouldShutdown()) break; // on exit
        //else 
        continue; // normal timeout
      }
      CHECK_GE(rc, 0) << std::strerror(errno) << ", workerID=" << _worker_id;
      break;
    }
    //if (BytePSGlobal::ShouldShutdown()) break;

    auto message = *(BytePSCommMsg*)buffer;

    if (_server_mode){
      // Server Listen Mode
      switch (message.signal) {
        case GATHER_PRE_READY:
          //To-do: add ready table operation here.
          LOG(INFO) << "receive info GATHER_PRE_READY from worker";
          
          // exec_msg = msg_list_[message.req_session_id][message.storage_id];
          // {
          //   CUDA_CALL(cudaMemcpyAsync(
          //       (void*) exec_msg.dst, 
          //       (const void *) exec_msg.src, 
          //       (size_t) exec_msg.len, 
          //       (cudaMemcpyKind) cudaMemcpyDeviceToHost, 
          //       (cudaStream_t) *exec_msg.stream));
          //   CUDA_CALL(cudaStreamSynchronize(*exec_msg.stream));

          //   byteps_server_->Response(exec_msg.req_meta, exec_msg.kvpairs);
          // }

          break;
        default:
          CHECK(0) << "unsupported signal: " << message.signal;
      }
    } else { 
      // Worker Listen Mode
      switch (message.signal) {
        case GATHER_PRE:
          LOG(INFO) << "receive info GATHER_PRE from server";
          _compute_hook->addWorkerGatherMonitor(message.req_session_id, message.storage_id);
          break;
        case SCATTER_FOLLOWUP_FWD:
          LOG(INFO) << "receive info SCATTER_FOLLOWUP_FWD from server";
          _compute_hook->scatter_followup_fwd(message.req_session_id, message.storage_id);
          break;
        case SCATTER_FOLLOWUP_BWD:
          LOG(INFO) << "receive info SCATTER_FOLLOWUP_BWD from server";
          _compute_hook->scatter_followup_bwd(message.req_session_id, message.storage_id);
          break;
        default:
          CHECK(0) << "unsupported signal: " << message.signal;
      }
    }

    LOG(INFO) << "root socket recved: src=" << message.src
                   << ", signal=" << message.signal
                   << ", worker=" << _worker_id;
  }
  LOG(INFO) << "listen thread joined"
                 << " (worker=" << _worker_id << ")";
}

int BytePSCommSocket::sendSignal(int destination, void* data, int len) {
  std::lock_guard<std::mutex> lock(_socket_mu);
  struct sockaddr_un destaddr;
  memset(&destaddr, 0, sizeof(destaddr));
  destaddr.sun_family = AF_UNIX;

  std::string fd_path(_recv_path);
  fd_path += std::to_string(destination);
  strncpy(destaddr.sun_path, fd_path.c_str(), sizeof(destaddr.sun_path) - 1);

  int ret = -1;
  while (ret < 0) {
    ret = sendto(_send_fd, data, len, 0, (struct sockaddr*)&destaddr,
                 sizeof(struct sockaddr_un));
    if (ret < 0) {
      LOG(INFO) << "Socket send error " << std::strerror(errno)
                     << ", workerID=" << _worker_id;
      std::this_thread::sleep_for(std::chrono::microseconds(1000000));
    }
  }

  return ret;
}


int BytePSCommSocket::recvSignal(int* source, void* data, int max_len) {
  int rc;
  while (true) {
    rc = recv(_recv_fd, data, MAX_LINE, MSG_WAITALL);
    if (rc < 0 && errno == EINTR) continue;
    if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { // timeout
        //if (BytePSGlobal::ShouldShutdown()) break; // on exit
        //else 
        continue; // normal timeout
    }
    CHECK_GE(rc, 0) << std::strerror(errno) << ", rank=" << _worker_id;
    CHECK_LE(rc, max_len)
        << "recv_len=" << rc << ", but given max_len=" << max_len;
    break;
  }
  //if (BytePSGlobal::ShouldShutdown()) return rc;
  auto message = *(BytePSCommMsg*)data;
  *source = message.src;

  LOG(INFO) << "non-root socket recved: src=" << message.src
                 << ", signal=" << message.signal
                 << ", myrank=" << _worker_id;

  return rc;
}


int BytePSCommSocket::send_gather_ready(int req_session_id, int storage_id) {
  int dst = _worker_id + 1; // dst is server
  int op;
  BytePSCommSignal sig = GATHER_PRE_READY;
  struct BytePSCommMsg data = {_worker_id, sig, req_session_id, storage_id, op};
  return sendSignal(dst, &data, sizeof(data));
}

int BytePSCommSocket::send_scatter_followup_fwd(int req_session_id, int storage_id, int op) {
  int dst = _worker_id - 1; // dst is worker
  BytePSCommSignal sig = SCATTER_FOLLOWUP_FWD;
  struct BytePSCommMsg data = {_worker_id, sig, req_session_id, storage_id, op};
  return sendSignal(dst, &data, sizeof(data));
}

int BytePSCommSocket::send_scatter_followup_bwd(int req_session_id, int storage_id, int op) {
  int dst = _worker_id - 1; // dst is worker
  BytePSCommSignal sig = SCATTER_FOLLOWUP_BWD;
  struct BytePSCommMsg data = {_worker_id, sig, req_session_id, storage_id, op};
  return sendSignal(dst, &data, sizeof(data));
}

int BytePSCommSocket::send_gather_pre(int req_session_id, int storage_id) {
  int dst = _worker_id - 1; // dst is worker
  BytePSCommSignal sig = GATHER_PRE;
  int op;
  struct BytePSCommMsg data = {_worker_id, sig, req_session_id, storage_id, op};
  return sendSignal(dst, &data, sizeof(data));
}

}  // namespace common
}  // namespace byteps
