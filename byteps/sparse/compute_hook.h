#ifndef BYTEPS_COMPUTE_HOOK_H
#define BYTEPS_COMPUTE_HOOK_H

#include "queue_exec.h"

namespace byteps {
namespace sparse {

using LocalRankCallback = std::function<void(int, int)>;

// Forward declaration for debugging.
class BytePSCommSocket;

class ComputeHookManager{
 public:

  ComputeHookManager(LocalRankCallback scatter_followup, LocalRankCallback gather_precondition)
    : _scatter_followup(scatter_followup),
      _gather_precondition(gather_precondition),
      _gather_ready_monitor_loop(nullptr) 
    {
      _gather_ready_monitor_loop = std::unique_ptr<QueueExecLoop>(QueueExecLoop::init_loop());
    }

  inline void addWorkerGatherMonitor(int request_sess_id, int storage_device_id) {
    auto monitor_job = [this, request_sess_id, storage_device_id] () {
      _gather_precondition(request_sess_id, storage_device_id);

      // Uncomment this line after BytePSCommSocket is plug backed in.
      // _comm_socket->sendGatherReady(request_sess_id, storage_device_id);
    };
    _gather_ready_monitor_loop->add_worker(monitor_job);
  }

 private:
  LocalRankCallback _scatter_followup;
  LocalRankCallback _gather_precondition;

  BytePSCommSocket* _comm_socket;
  std::unique_ptr<QueueExecLoop> _gather_ready_monitor_loop;
};


} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_COMPUTE_HOOK_H
