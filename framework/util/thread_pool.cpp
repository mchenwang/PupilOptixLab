#include "thread_pool.h"

#include <queue>
#include <mutex>
#include <condition_variable>

namespace Pupil::util {
struct ThreadPool::Impl {
    unsigned int threads_num = 0;
    std::unique_ptr<std::thread[]> threads_list;
    std::queue<std::function<void()>> task_queue;

    // synchronization
    std::atomic_uint32_t num_busy_threads;
    std::mutex task_queue_mutex;
    std::condition_variable task_cv;
    std::mutex join_mutex;
    std::condition_variable join_cv;
    bool stop_flag = true;
};
ThreadPool::Impl *ThreadPool::m_impl = nullptr;

void ThreadPool::Init(unsigned int threads_num) noexcept {
    if (m_impl) return;

    m_impl = new ThreadPool::Impl();
    m_impl->threads_num = threads_num;
    m_impl->stop_flag = false;
    m_impl->threads_list = std::make_unique<std::thread[]>(threads_num);
    m_impl->num_busy_threads = 0;

    for (unsigned int i = 0; i < threads_num; i++) {
        m_impl->threads_list[i] = std::thread([index = i]() {
            while (true) {
                std::function<void()> task = nullptr;
                {
                    std::unique_lock<std::mutex> lock(m_impl->task_queue_mutex);
                    m_impl->task_cv.wait(lock, []() { return m_impl->stop_flag || !m_impl->task_queue.empty(); });

                    if (m_impl->stop_flag && m_impl->task_queue.empty()) return;
                    ++m_impl->num_busy_threads;

                    task = m_impl->task_queue.front();
                    m_impl->task_queue.pop();
                }

                if (task) task();
                --m_impl->num_busy_threads;

                if (m_impl->num_busy_threads == 0 && m_impl->task_queue.empty()) {
                    std::unique_lock<std::mutex> lock(m_impl->join_mutex);
                    m_impl->join_cv.notify_all();
                }
            }
        });
    }
}

void ThreadPool::JoinAll() noexcept {
    std::unique_lock<std::mutex> lock(m_impl->join_mutex);
    m_impl->join_cv.wait(lock, []() { return m_impl->num_busy_threads == 0 && m_impl->task_queue.empty(); });
}

void ThreadPool::Destroy() noexcept {
    {
        std::unique_lock<std::mutex> lock(m_impl->task_queue_mutex);
        m_impl->stop_flag = true;
        m_impl->task_cv.notify_all();
    }

    if (m_impl->threads_list)
        for (auto i = 0u; i < m_impl->threads_num; i++)
            m_impl->threads_list[i].join();

    m_impl->threads_list.reset();
    delete m_impl;
}

void ThreadPool::Enqueue(std::function<void()> &&task) noexcept {
    std::unique_lock<std::mutex> lock(m_impl->task_queue_mutex);

    if (!m_impl->stop_flag) {
        m_impl->task_queue.emplace(std::forward<std::function<void()>>(task));
        m_impl->task_cv.notify_one();
    }
}
}// namespace Pupil::util