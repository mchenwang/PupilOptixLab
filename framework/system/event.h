#pragma once

#include <functional>
#include <vector>
#include <mutex>
#include <map>

namespace Pupil {
template<typename ET>
concept EnumType = std::is_enum_v<ET>;

template<EnumType ET, ET enum_v, typename PT>
class Event {
public:
    static void Bind(std::function<void(PT)> &&op) noexcept {
        std::scoped_lock<std::mutex> lock(m_mutex);
        m_ops[enum_v].push_back(op);
    }

    static void Respond(PT p) noexcept {
        for (auto &&op : m_ops[enum_v]) op(p);
    }

private:
    static std::mutex m_mutex;
    static std::map<ET, std::vector<std::function<void(PT)>>> m_ops;
};

template<EnumType ET, ET enum_v, typename PT>
std::mutex Event<ET, enum_v, PT>::m_mutex;
template<EnumType ET, ET enum_v, typename PT>
std::map<ET, std::vector<std::function<void(PT)>>> Event<ET, enum_v, PT>::m_ops;

template<EnumType ET, ET enum_v>
class Event<ET, enum_v, void> {
public:
    static void Bind(std::function<void()> &&op) noexcept {
        std::scoped_lock<std::mutex> lock(m_mutex);
        m_ops[enum_v].push_back(op);
    }

    static void Respond() noexcept {
        for (auto &&op : m_ops[enum_v]) op();
    }

private:
    static std::mutex m_mutex;
    static std::map<ET, std::vector<std::function<void()>>> m_ops;
};

template<EnumType ET, ET enum_v>
std::mutex Event<ET, enum_v, void>::m_mutex;
template<EnumType ET, ET enum_v>
std::map<ET, std::vector<std::function<void()>>> Event<ET, enum_v, void>::m_ops;

namespace detail {
template<typename T>
concept class_callable =
    requires(T t) {
        std::invocable<decltype(T::operator())>;
    };

template<typename T>
struct deduce_type {
    static constexpr bool is_callable = false;
};

template<typename Return, typename Class, typename Arg>
struct deduce_type<Return (Class::*)(Arg) const> {
    static constexpr bool is_callable = true;
    using arg = Arg;
    using ret = Return;
};
template<typename Return, typename Class>
struct deduce_type<Return (Class::*)() const> {
    static constexpr bool is_callable = true;
    using arg = void;
    using ret = Return;
};
template<typename Return, typename Arg>
struct deduce_type<Return (&)(Arg)> {
    static constexpr bool is_callable = true;
    using arg = Arg;
    using ret = Return;
};
template<typename Return>
struct deduce_type<Return (&)()> {
    static constexpr bool is_callable = true;
    using arg = void;
    using ret = Return;
};
}// namespace detail

template<auto enum_v, detail::class_callable Func>
constexpr inline void EventBinder(Func &&f) {
    Event<decltype(enum_v), enum_v,
          typename detail::deduce_type<
              decltype(&Func::operator())>::arg>::Bind(std::forward<Func>(f));
}
template<auto enum_v, typename Func>
    requires(!detail::class_callable<Func> && detail::deduce_type<Func>::is_callable)
constexpr inline void EventBinder(Func &&f) {
    Event<decltype(enum_v), enum_v,
          typename detail::deduce_type<
              decltype(&Func::operator())>::arg>::Bind(std::forward<Func>(f));
}

template<auto enum_v>
constexpr inline void EventDispatcher() {
    Event<decltype(enum_v), enum_v, void>::Respond();
}
template<auto enum_v, typename PT>
constexpr inline void EventDispatcher(PT p) {
    Event<decltype(enum_v), enum_v, PT>::Respond(p);
}
}// namespace Pupil