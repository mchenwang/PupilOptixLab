#include "thread_pool.h"

#include <queue>
#include <mutex>
#include <condition_variable>

namespace {
unsigned int s_threads_num = 0;
std::unique_ptr<std::thread[]> s_threads_list;
std::queue<std::function<void()>> s_task_queue;

// synchronization
std::mutex s_queue_mutex;
std::condition_variable s_cv;
bool s_stop_flag = true;
}// namespace

namespace Pupil::util {
void ThreadPool::Init(unsigned int threads_num) noexcept {
    s_threads_num = std::max(1u, threads_num);

    s_stop_flag = false;
    s_threads_list = std::make_unique<std::thread[]>(threads_num);
    for (unsigned int i = 0; i < threads_num; i++) {
        s_threads_list[i] = std::thread([index = i]() {
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(s_queue_mutex);
                    s_cv.wait(lock, []() { return s_stop_flag || !s_task_queue.empty(); });

                    if (s_stop_flag && s_task_queue.empty()) return;
                }

                auto task = s_task_queue.front();
                s_task_queue.pop();
                task();
            }
        });
    }

    m_init_flag = true;

    // PEngine::Log::Info("Thread pool initialized with {} threads", s_threads_num);
}

void ThreadPool::Destroy() noexcept {
    {
        std::unique_lock<std::mutex> lock(s_queue_mutex);
        s_stop_flag = true;
        s_cv.notify_all();
    }

    if (s_threads_list)
        for (auto i = 0u; i < s_threads_num; i++)
            s_threads_list[i].join();

    s_threads_list.reset();
    m_init_flag = false;
}

void ThreadPool::Enqueue(std::function<void()> &&task) noexcept {
    std::unique_lock<std::mutex> lock(s_queue_mutex);

    if (!s_stop_flag) {
        s_task_queue.emplace(task);
        s_cv.notify_one();
    }
}

}// namespace Pupil::util