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

#ifndef BYTEPS_SOCKETCOMM_H
#define BYTEPS_SOCKETCOMM_H

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include "util.h"

#define DEFAULT_BASE_SOCKET_PATH_RECV "/tmp/socket_recv_"
#define DEFAULT_BASE_SOCKET_PATH_SEND "/tmp/socket_send_"
#define MAX_LINE 8000

namespace byteps {
namespace sparse {
//enum BytePSRole { LOCAL_ROOT, LOCAL_WORKER };

enum BytePSCommSignal {
  GATHER_PRE,
  SCATTER_FOLLOWUP,
  GATHER_PRE_READY
};

struct BytePSCommMsg {
  int src;
  BytePSCommSignal signal;
  int req_session_id;
  int storage_id;
  int op;
};

class BytePSSparseComm {
 public:
  BytePSSparseComm() { _comm = nullptr; }

  virtual void init(int* local_size, int* worker_id) = 0;
  virtual int sendSignal(int destination, void* data, int len) = 0;
  virtual int recvSignal(int* source, void* data, int max_len) = 0;

  virtual int getLocalSize() { return _local_size; }
  virtual int getWorkerID() { return _worker_id; }

 protected:
  int _local_size;
  int _worker_id;

  void* _comm;
};

class BytePSCommSocket : public BytePSSparseComm {
 public:
  BytePSCommSocket() {}
  BytePSCommSocket(std::shared_ptr<BytePSSparseComm> comm,
                   const std::string& path_suffix);

  ~BytePSCommSocket() {
    if (_listen_thread) {
      _listen_thread->join();
    }
    close(_send_fd);
    close(_recv_fd);

    auto fd_path = _send_path + std::to_string(_worker_id);
    if (!std::remove(fd_path.c_str())) {
      LOG(INFO) << "Clear socket " << fd_path;
    }
    fd_path = _recv_path + std::to_string(_worker_id);
    if (!std::remove(fd_path.c_str())) {
      LOG(INFO) << "Clear socket " << fd_path;
    }

    LOG(INFO) << "Clear BytePSCommSocket"
                   << " (rank=" << _worker_id << ")";
  }

  void init(int* local_size, int* worker_id);
  int sendSignal(int destination, void* data, int len);
  int recvSignal(int* source, void* data, int max_len);
  int gather_ready(int req_session_id, int storage_id, int max_len);
  int gather_pre(int req_session_id, int storage_id, int max_len, int op);

  int getSendFd() { return _send_fd; }
  int getRecvFd() { return _recv_fd; }

  std::string getSendPath() { return _send_path; }
  std::string getRecvPath() { return _recv_path; }

 protected:
  void startListenThread();
  int initSocket(int rank, const std::string& path);

  std::thread* _listen_thread;

  std::string _send_path;
  std::string _recv_path;
  int _recv_fd;
  int _send_fd;

  std::mutex _socket_mu;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMMUNICATOR_H
