#pragma once

#include <functional>
#include <vector>
#include <mutex>
#include <map>

namespace Pupil {
template<typename ET>
concept EnumType = std::is_enum_v<ET>;

template<EnumType ET, ET enum_v>
class Event {
public:
    static void Bind(std::function<void(void *)> &&op) noexcept {
        std::scoped_lock<std::mutex> lock(m_mutex);
        m_ops.push_back(op);
    }

    static void Respond(void *p = nullptr) noexcept {
        for (auto &&op : m_ops) op(p);
    }

private:
    static std::mutex m_mutex;
    static std::vector<std::function<void(void *)>> m_ops;
};

template<EnumType ET, ET enum_v>
std::mutex Event<ET, enum_v>::m_mutex;
template<EnumType ET, ET enum_v>
std::vector<std::function<void(void *)>> Event<ET, enum_v>::m_ops;

template<auto enum_v, typename Func>
    requires(std::invocable<Func, void *>)
inline void EventBinder(Func &&f) {
    Event<decltype(enum_v), enum_v>::Bind(std::forward<Func>(f));
}

template<auto enum_v>
inline void EventDispatcher() {
    Event<decltype(enum_v), enum_v>::Respond();
}
template<auto enum_v>
inline void EventDispatcher(void *p) {
    Event<decltype(enum_v), enum_v>::Respond(p);
}
template<auto enum_v, typename T>
    requires(!std::is_pointer_v<T>)
inline void EventDispatcher(T p) {
    Event<decltype(enum_v), enum_v>::Respond((void *)&p);
}
}// namespace Pupil