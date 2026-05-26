#include "tasksys.h"
#include <iostream>
#include <cassert>
#include <atomic>
#ifdef _WIN32
// no unistd.h on Windows
#else
#include <unistd.h>
#endif
#include <chrono>


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads), num_threads_(num_threads), shutdown(false), queue_head(0), next_task_id(0) {
    //
    // DONE: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    // 拍脑袋预分配足够大的连续空间，物理隔绝任何运行时内存拷贝开销
    ready_queue.reserve(500000);
    task_registry.reserve(100000);
    task_completed.reserve(100000);
    dynamic_graph.reserve(100000);
    incoming_edges.reserve(100000);

    for (int i = 0; i < num_threads_; ++i) {
        workers.emplace_back(&TaskSystemParallelThreadPoolSleeping::workerLoop, this);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // DONE: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
   {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown = true;
    }
    cv_worker.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) worker.join();
    }
    for (auto* status : task_registry) {
        delete status;
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    //
    // DONE: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    std::vector<TaskID> no_deps;
    runAsyncWithDeps(runnable, num_total_tasks, no_deps);
    sync();
}


TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks, const std::vector<TaskID>& deps) {
    //
    // TODO: CS149 students will implement this method in Part B.
    //
    std::unique_lock<std::mutex> lock(mutex_);
    
    TaskID tid = next_task_id++;
    active_tasks_count++;

    TaskStatus* status = new TaskStatus();
    status->runnable = runnable;
    status->total_elements = num_total_tasks;
    
    task_registry.push_back(status);
    task_completed.push_back(false);
    dynamic_graph.push_back(std::vector<TaskID>());
    incoming_edges.push_back(0);

    int current_deps_count = 0;
    for (TaskID dep : deps) {
        if (!task_completed[dep]) {
            current_deps_count++;
            dynamic_graph[dep].push_back(tid);
        }
    }
    incoming_edges[tid] = current_deps_count;

    if (current_deps_count == 0) {
        // 记录投递前的队列大小
        size_t old_size = ready_queue.size();
        
        for (int i = 0; i < num_total_tasks; ++i) {
            ready_queue.push_back({runnable, i, num_total_tasks, tid});
        }
        
        // 【💡 唤醒调优】：只有在队列原本是干涸空置的状态下，才去触发激烈的 notify
        if (queue_head >= (int)old_size) {
            if (num_total_tasks == 1) {
                cv_worker.notify_one(); // 单个任务只叫醒一个人，防止惊群
            } else {
                cv_worker.notify_all(); // 批量大任务叫醒大家一起干
            }
        }
    }

    return tid;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    std::unique_lock<std::mutex> lock(mutex_);
    cv_master.wait(lock, [this]() {
        return active_tasks_count == 0;
    });
}

// 工人循环：无死锁、有限自旋退避、高性能去中心化内核
void TaskSystemParallelThreadPoolSleeping::workerLoop() {
    while (true) {
        ReadySlice my_slice;
        bool has_work = false;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // 【💡 临界用户态阻尼】：队列没活时，先不急着去执行沉重的 cv.wait() 掉进内核态
            // 因为主线程可能正在以几微秒的间隔疯狂高频扔进来下一个 async 任务！
            if (!shutdown && queue_head >= (int)ready_queue.size()) {
                lock.unlock();
                
                // 进行约 150 次的 CPU 硬件暂停自旋，拖延进入内核态的时间
                for (int spin = 0; spin < 150; ++spin) {
                    #if defined(__x86_64__) || defined(_M_X64)
                    __builtin_ia32_pause();
                    #else
                    asm volatile("" : : : "memory");
                    #endif
                }
                
                lock.lock();
            }

            // 如果阻尼空转完还是没等到新活，那说明真空期真的到了，此时心安理得地睡觉
            while (!shutdown && queue_head >= (int)ready_queue.size()) {
                cv_worker.wait(lock);
            }

            if (shutdown) return;

            if (queue_head < (int)ready_queue.size()) {
                my_slice = ready_queue[queue_head++];
                has_work = true;
            }
        }

        if (has_work) {
            my_slice.runnable->runTask(my_slice.element_id, my_slice.total_elements);

            TaskStatus* status = task_registry[my_slice.task_id];
            int prev_completed = status->completed_elements.fetch_add(1);

            if (prev_completed + 1 == my_slice.total_elements) {
                std::unique_lock<std::mutex> lock(mutex_);
                
                task_completed[my_slice.task_id] = true;
                active_tasks_count--;

                bool spawned_any = false;
                for (TaskID next_tid : dynamic_graph[my_slice.task_id]) {
                    incoming_edges[next_tid]--;
                    if (incoming_edges[next_tid] == 0) {
                        TaskStatus* next_status = task_registry[next_tid];
                        for (int i = 0; i < next_status->total_elements; ++i) {
                            ready_queue.push_back({next_status->runnable, i, next_status->total_elements, next_tid});
                        }
                        spawned_any = true;
                    }
                }

                if (active_tasks_count == 0) {
                    cv_master.notify_all();
                }
                
                // 【💡 唤醒调优】：解开依赖时同理，只在必要时通知
                if (spawned_any) {
                    cv_worker.notify_all();
                }
            }
        }
    }
}