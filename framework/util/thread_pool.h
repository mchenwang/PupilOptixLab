#pragma once

#include "util.h"

#include <thread>
#include <future>
#include <functional>

namespace Pupil::util {
class ThreadPool : public util::Singleton<ThreadPool> {
public:
    void Init(unsigned int threads_num = std::thread::hardware_concurrency() / 2) noexcept;
    void Destroy() noexcept;

    template<typename Func, typename... Ts>
        requires(std::invocable<Func, Ts...> && std::is_same_v<void, std::invoke_result_t<Func, Ts && ...>>)
    void AddTask(Func &&func, Ts &&...args) noexcept {
        std::function<void()> task_function = std::bind(std::forward<Func>(func), std::forward<Ts>(args)...);
        Enqueue(std::move(task_function));
    }

    template<typename Func, typename... Ts>
        requires std::invocable<Func, Ts...>
    auto AddTaskWithReturnValue(Func &&func, Ts &&...args) noexcept {
        using ReturnType = std::invoke_result_t<Func, Ts &&...>;
        auto shared_promise = std::make_shared<std::promise<ReturnType>>();
        auto task = [&, promise = shared_promise]() {
            promise->set_value(func(std::forward<Ts>(args)...));
        };

        auto ret_futre = shared_promise->get_future();
        Enqueue(std::move(task));
        return ret_futre;
    }

private:
    void Enqueue(std::function<void()> &&) noexcept;
};
}// namespace Pupil::util