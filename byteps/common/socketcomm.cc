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

#include "socketcomm.h"
#include <cerrno>
#include <cstring>


namespace byteps {
namespace sparse {

// Copy constructor that provides the option to reconfigure members.
// The ranks in members always use local_rank, regardless that the members
// may be a subset of all local ranks.
BytePSCommSocket::BytePSCommSocket(std::shared_ptr<BytePSComm> comm,
                                   const std::string& path_suffix) {
  std::shared_ptr<BytePSCommSocket> sock_comm =
      std::static_pointer_cast<BytePSCommSocket>(comm);
  // TODO: use private members directly
  _local_size = sock_comm->getLocalSize();
  _worker_id = sock_comm->getWorkerID();
  _send_path = sock_comm->getSendPath() + path_suffix;
  _recv_path = sock_comm->getRecvPath() + path_suffix;
  _send_fd = initSocket(_worker_id, _send_path);
  _recv_fd = initSocket(_worker_id, _recv_path);

  _listen_thread =
        new std::thread(&BytePSCommSocket::startListenThread, this);

  LOG(INFO) << "all sockets create successfully";
}

void BytePSCommSocket::init(int* local_size, int* worker_id) {
  LOG(INFO) << "Using Communicator=Socket";

  // We should init size, etc. using getenv
  // do env check
  BPS_CHECK(getenv("BYTEPS_LOCAL_SIZE"))
      << "error: env BYTEPS_LOCAL_SIZE not set";
  BPS_CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
  BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

  *local_size = atoi(getenv("BYTEPS_LOCAL_SIZE"));
  *worker_id = atoi(getenv("DMLC_WORKER_ID"));

  _local_size = *local_size;
  _worker_id = *worker_id;


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
  BPS_CHECK_GE(fd, 0) << "recv socket create failed";

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
  BPS_CHECK_GE(ret, 0) << fd_path << " bind failed: " << strerror(errno);

  LOG(INFO) << "Init socket at " << fd_path;

  return fd;
}

void BytePSCommSocket::startListenThread() {  // each worker starts this in
                                              // background thread
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
      //}
      BPS_CHECK_GE(rc, 0) << std::strerror(errno) << ", workerID=" << _worker_id;
      break;
    }
    //if (BytePSGlobal::ShouldShutdown()) break;

    auto message = *(BytePSCommMsg*)buffer;

    switch (message.signal) {
      case GATHER_PRE:
        //To-do
        break;
      case SCATTER_FOLLOWUP:
        //To-do
        break;
      case GATHER_PRE_READY:
        //To-do
        break;
      default:
        BPS_CHECK(0) << "unsupported signal: " << message.signal;
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
      BPS_LOG(DEBUG) << "Socket send error " << std::strerror(errno)
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
    BPS_CHECK_GE(rc, 0) << std::strerror(errno) << ", rank=" << _worker_id;
    BPS_CHECK_LE(rc, max_len)
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


int BytePSCommSocket::gather_ready(int req_session_id, int storage_id, int max_len) {
  int dst = _worker_id*2;
  int op = 0;
  struct BytePSCommMsg data = {_worker_id, GATHER_PRE_READY, req_session_id, storage_id, op};
  return sendSignal(dst, &data, max_len);
}

}  // namespace common
}  // namespace byteps
