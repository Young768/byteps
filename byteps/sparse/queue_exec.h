#ifndef BYTEPS_SPARSE_QUEUE_EXEC_H
#define BYTEPS_SPARSE_QUEUE_EXEC_H

#include <functional>
#include <condition_variable>  // std::condition_variable
#include <functional>
#include <memory>
#include <mutex>  // std::mutex, std::unique_lock
#include <queue>  // std::queue
#include <thread>
#include <cuda_runtime.h>

#include "util.h"
#include "cpu_reducer.h"

namespace byteps {
namespace sparse {

class QueueExecLoop {
 public:
  static QueueExecLoop* init_loop();

  // This should be invoked by any participant that want to post a job.
  void add_worker(std::function<void()> job);

  // This should be only called at destructor or during test.
  void stop_executors();

  QueueExecLoop(int num_worker = 1) : num_worker_(num_worker), running_(true) {}

  ~QueueExecLoop() {
    stop_executors();
  }

  void start_executors();

 private:
  int num_worker_;

  // This is to enqueue and dequeue the forward jobs accordingly.
  std::mutex job_queue_mtx_;
  std::condition_variable job_queue_cv_;

  std::queue<std::function<void()>> job_queue_;  // Protected by job_queue_mtx_.
  volatile bool running_;

  // Since each TF session would not be able to consume everything, we would
  // like to make it more aggressive to consume all CPU/GPU via multi-session.
  std::vector<std::thread> background_job_polls_;
};

struct DenseTask{
  int workerID;
  int local_rank;
  size_t buffer_size; // In bytes.
  cudaStream_t streamH2D;
  cudaStream_t streamD2H;

  void * baseSrcPtr;
  void * cpuDenseDeltaPtr;
  void * cpuDenseLatestPtr;
  void * baseResultPtr;

  std::function<void(int local_rank)> allFinishCallback;
};


class PredefinedDenseQueueExecLoop : public QueueExecLoop{
 public:
  void add_predefined_worker(DenseTask task);

  void set_downstream(PredefinedDenseQueueExecLoop * downstream);

//  protected:
  PredefinedDenseQueueExecLoop() : QueueExecLoop() {}
  virtual ~PredefinedDenseQueueExecLoop() {};

 private:
  virtual void predefined_work(DenseTask task) = 0;
  
  PredefinedDenseQueueExecLoop * downstream_;
};


class MemcpyH2DQueueExecLoop : public PredefinedDenseQueueExecLoop{
 public:
  static MemcpyH2DQueueExecLoop* init_loop(std::mutex * mtx_DenseLatestBuffers);
  ~MemcpyH2DQueueExecLoop() override {
    stop_executors();
    delete mtx_DenseLatestBuffers_;
  }

 private:
  MemcpyH2DQueueExecLoop(std::mutex * mtx_DenseLatestBuffers) 
    : PredefinedDenseQueueExecLoop(), mtx_DenseLatestBuffers_(mtx_DenseLatestBuffers) {}

  void predefined_work(DenseTask task) override;

  std::mutex * mtx_DenseLatestBuffers_;
};

class CPUReduceQueueExecLoop : public PredefinedDenseQueueExecLoop {
 public:
  static CPUReduceQueueExecLoop* init_loop(::byteps::sparse::CpuReducer* denseReducer,
                                           std::mutex * mtx_DenseLatestBuffers);
  ~CPUReduceQueueExecLoop() override {
    stop_executors();
    delete mtx_DenseLatestBuffers_;
    delete _loopdenseReducer;
  }

 private:
  CPUReduceQueueExecLoop(::byteps::sparse::CpuReducer * denseReducer, std::mutex * mtx_DenseLatestBuffers)
    : PredefinedDenseQueueExecLoop(), _loopdenseReducer(denseReducer), mtx_DenseLatestBuffers_(mtx_DenseLatestBuffers) {}

  void predefined_work(DenseTask task) override;

  ::byteps::sparse::CpuReducer* _loopdenseReducer;
  std::mutex * mtx_DenseLatestBuffers_;
};

class MemcpyD2HQueueExecLoop : public PredefinedDenseQueueExecLoop{
 public:
  static MemcpyD2HQueueExecLoop* init_loop();
  ~MemcpyD2HQueueExecLoop() override { stop_executors();}

 private:
  MemcpyD2HQueueExecLoop() : PredefinedDenseQueueExecLoop() {}

  void predefined_work(DenseTask task) override;
};


} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_QUEUE_EXEC_H