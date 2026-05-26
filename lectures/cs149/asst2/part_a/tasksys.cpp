#include "tasksys.h"


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
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
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

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads), num_threads(num_threads), shutdown(false), current_runnable(nullptr), total_elements(0), next_element(0), completed_elements(0) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    for (int i = 0; i < num_threads; ++i){
        workers.emplace_back(&TaskSystemParallelSpawn::worker_loop, this, i);
    }
}

void TaskSystemParallelSpawn::worker_loop(int thread_id){
    while (true) {
        int my_element = -1;
        IRunnable* my_runnable = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);

            while (!shutdown && (current_runnable == nullptr || next_element >= total_elements)) {
                cv_worker.wait(lock);
            }

            if (shutdown){
                return;
            }

            if (next_element < total_elements){
                my_element = next_element++;
                my_runnable = current_runnable;
            }
        }

        if (my_runnable != nullptr && my_element != -1) {
            my_runnable->runTask(my_element, total_elements);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                completed_elements++;

                if (completed_elements >= total_elements){
                    cv_master.notify_all();
                }
            }
        }
    }
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown = true;
    }

    cv_worker.notify_all();

    for (std::thread& worker:workers){
        if (worker.joinable()){
            worker.join();
        }
    }
}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }

    {
        std::unique_lock<std::mutex> lock(mutex_);
        current_runnable = runnable;
        total_elements = num_total_tasks;
        next_element = 0;
        completed_elements = 0;
    }

    cv_worker.notify_all();

    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (completed_elements < total_elements){
            cv_master.wait(lock);
        }

        current_runnable = nullptr;
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads), num_threads(num_threads), shutdown(false), current_runnable(nullptr), total_elements(0), next_element(0), completed_elements(0) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(&TaskSystemParallelThreadPoolSpinning::worker_loop, this, i);
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() 
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown = true;
    }
    for (std::thread& worker: workers){
        if (worker.joinable()){
            worker.join();
        }
    }
}

void TaskSystemParallelThreadPoolSpinning::worker_loop(int thread_id) {
    while (true) {
        int my_element = -1;
        IRunnable* my_runnable = nullptr;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (shutdown){
                return;
            }

            if (current_runnable != nullptr && next_element < total_elements) {
                my_element = next_element++;
                my_runnable = current_runnable;
            }
        }

        if (my_runnable != nullptr && my_element != -1) {
            my_runnable->runTask(my_element, total_elements);

            {
                std::unique_lock<std::mutex> lock(mutex_);
                completed_elements++;
            }
        }
        else{
            std::this_thread::yield();
        }
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    {
        std::unique_lock<std::mutex> lock(mutex_);
        current_runnable = runnable;
        total_elements = num_total_tasks;
        next_element = 0;
        completed_elements = 0;
    }

    while (true){
        std::unique_lock<std::mutex> lock(mutex_);
        if (completed_elements >= total_elements) {
            current_runnable = nullptr;
            break;
        }
    }

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks, const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
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

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads), num_threads(num_threads), shutdown(false), current_runnable(nullptr), total_elements(0), next_element(0), completed_elements(0) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    for (int i = 0; i < num_threads; ++i){
        workers.emplace_back(&TaskSystemParallelThreadPoolSleeping::worker_loop, this, i);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown = true;
    }
    cv_worker.notify_all();
    for (std::thread& worker: workers){
        if (worker.joinable()){
            worker.join();
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::worker_loop(int thread_id) {
    while (true) {
        int my_element = -1;
        IRunnable* my_runnable = nullptr;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // 【重点】：如果车间没倒闭，并且（还没有分配任务，或者活已经被抢光了）
            // 工人们就不准往下走，老老实实躺下睡觉
            while (!shutdown && (current_runnable == nullptr || next_element >= total_elements)){
                cv_worker.wait(lock);
            }

            if (shutdown){
                return;
            }
            
            // 醒来后，安全地领走一个切片
            if (next_element < total_elements) {
                my_element = next_element++;
                my_runnable = current_runnable;
            }

        }// 解锁

        // 【裸奔干活】
        if (my_runnable != nullptr && my_element != -1) {
            my_runnable->runTask(my_element, total_elements);
            
            // 【汇报账本】
            {
                std::unique_lock<std::mutex> lock(mutex_);
                completed_elements++;

                // 如果我是最后一个把大任务里所有切片刷完的人，去敲办公室门叫醒经理
                if (completed_elements >= total_elements){
                    cv_master.notify_all();
                }
            }
        }
    }
}


void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }

    {
        std::unique_lock<std::mutex> lock(mutex_);
        current_runnable = runnable;
        total_elements = num_total_tasks;
        next_element = 0;
        completed_elements = 0;
    }

    cv_worker.notify_all();

    {
        std::unique_lock<std::mutex> lock(mutex_);
        while(completed_elements < total_elements){
            cv_master.wait(lock);
        }
        current_runnable = nullptr;
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
