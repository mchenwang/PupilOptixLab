#pragma once

#include "util/util.h"

#include <any>
#include <vector>
#include <queue>
#include <memory>
#include <functional>
#include <unordered_map>

namespace Pupil::Event {
    class Handler {
    public:
        using Args = std::vector<std::any>;

        virtual void Handle(const Args& args) noexcept = 0;
    };

    class Handler0A final : public Handler {
    public:
        Handler0A(std::function<void()>&& func) noexcept : m_func(func) {}

        virtual void Handle(const Args& args) noexcept override {
            m_func();
        }

    private:
        std::function<void()> m_func;
    };

    template<typename T>
    class Handler1A final : public Handler {
    public:
        Handler1A(std::function<void(T)>&& func) noexcept : m_func(func) {}

        virtual void Handle(const Args& args) noexcept override {
            if (args.size() < 1) return;
            if (const T* arg = std::any_cast<const T>(&args[0]); arg)
                m_func(*arg);
        }

    private:
        std::function<void(T)> m_func;
    };

    template<typename T1, typename T2>
    class Handler2A final : public Handler {
    public:
        Handler2A(std::function<void(T1, T2)>&& func) noexcept : m_func(func) {}

        virtual void Handle(const Args& args) noexcept override {
            if (args.size() < 2) return;
            if (const T1* arg1 = std::any_cast<const T1>(&args[0]); arg1)
                if (const T2* arg2 = std::any_cast<const T2>(&args[1]); arg2)
                    m_func(*arg1, *arg2);
        }

    private:
        std::function<void(T1, T2)> m_func;
    };

    class Dispatcher {
    public:
        using HandlerList = std::vector<std::unique_ptr<Handler>>;

        Dispatcher() noexcept;
        ~Dispatcher() noexcept;

        void BindEvent(const char* event_id, Handler* handler) noexcept;

        void Push(const char* event_id) noexcept;
        void Push(const char* event_id, const Handler::Args& args) noexcept;

        void Dispatch() noexcept;
        void DispatchImmediately(const char* event_id, const Handler::Args& args) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };

    class Center : public util::Singleton<Center> {
    public:
        Center() noexcept;
        ~Center() noexcept;

        /**
         * bind event with dispatcher and handler
        */
        void BindEvent(const char* dispatcher_id, const char* event_id, Handler* handler) noexcept;

        void Dispatch(const char* dispatcher_id) noexcept;
        void DispatchImmediately(const char* event_id) noexcept;
        void DispatchImmediately(const char* event_id, const Handler::Args& args) noexcept;
        void DispatchImmediately(const char* dispatcher_id, const char* event_id, const Handler::Args& args) noexcept;

        void Send(const char* event_id) noexcept;
        void Send(const char* event_id, const Handler::Args& args) noexcept;

        void        RegisterDispatcher(const char* dispatcher_id) noexcept;
        Dispatcher* GetDispatcher(const char* dispatcher_id) noexcept;

        void Destroy() noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil::Event