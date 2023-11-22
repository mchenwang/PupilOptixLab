#include "event.h"

namespace Pupil::Event {
    struct Dispatcher::Impl {
        std::atomic<int>                                  flip_index = 0;
        std::queue<std::pair<const char*, Handler::Args>> fs[2];
        std::unordered_map<const char*, HandlerList>      handles;
    };

    Dispatcher::Dispatcher() noexcept {
        m_impl = new Impl();
    }

    Dispatcher::~Dispatcher() noexcept {
        if (m_impl) delete m_impl;
        m_impl = nullptr;
    }

    void Dispatcher::BindEvent(const char* event_id, Handler* handler) noexcept {
        if (m_impl->handles.find(event_id) == m_impl->handles.end())
            m_impl->handles.emplace(event_id, std::vector<std::unique_ptr<Handler>>());
        m_impl->handles[event_id].emplace_back(handler);
    }

    void Dispatcher::Push(const char* event_id) noexcept {
        int i = m_impl->flip_index.load();
        m_impl->fs[i].emplace(event_id, Handler::Args{});
    }

    void Dispatcher::Push(const char* event_id, const Handler::Args& args) noexcept {
        int i = m_impl->flip_index.load();
        m_impl->fs[i].emplace(event_id, args);
    }

    void Dispatcher::Dispatch() noexcept {
        int i = m_impl->flip_index.fetch_xor(1);
        while (!m_impl->fs[i].empty()) {
            auto it = m_impl->handles.find(m_impl->fs[i].front().first);
            if (it != m_impl->handles.end()) {
                for (auto&& handler : it->second) {
                    handler->Handle(m_impl->fs[i].front().second);
                }
            }
            m_impl->fs[i].pop();
        }
    }

    void Dispatcher::DispatchImmediately(const char* event_id, const Handler::Args& args) noexcept {
        if (auto it = m_impl->handles.find(event_id);
            it != m_impl->handles.end()) {
            for (auto&& handler : it->second) {
                handler->Handle(args);
            }
        }
    }

    struct Center::Impl {
        std::unordered_map<const char*, std::unique_ptr<Dispatcher>> dispatchers;
        std::unordered_map<const char*, std::vector<Dispatcher*>>    event_dispatcher_bind_map;
    };

    Center::Center() noexcept {
        m_impl = new Impl();
    }

    Center::~Center() noexcept {
        if (m_impl) delete m_impl;
        m_impl = nullptr;
    }

    void Center::Destroy() noexcept {
        m_impl->dispatchers.clear();
        m_impl->event_dispatcher_bind_map.clear();
    }

    void Center::BindEvent(const char* dispatcher_id, const char* event_id, Handler* handler) noexcept {
        if (auto dispatcher = GetDispatcher(dispatcher_id); dispatcher) {
            if (auto it = m_impl->event_dispatcher_bind_map.find(event_id);
                it == m_impl->event_dispatcher_bind_map.end()) {
                m_impl->event_dispatcher_bind_map.emplace(event_id, std::vector<Dispatcher*>());
            }
            m_impl->event_dispatcher_bind_map[event_id].emplace_back(dispatcher);
            dispatcher->BindEvent(event_id, handler);
        }
    }

    void Center::Dispatch(const char* dispatcher_id) noexcept {
        if (auto dispatcher = GetDispatcher(dispatcher_id); dispatcher) {
            dispatcher->Dispatch();
        }
    }

    void Center::DispatchImmediately(const char* event_id) noexcept {
        if (auto it = m_impl->event_dispatcher_bind_map.find(event_id);
            it != m_impl->event_dispatcher_bind_map.end()) {
            for (auto&& dispatcher : it->second) {
                dispatcher->DispatchImmediately(event_id, {});
            }
        }
    }

    void Center::DispatchImmediately(const char* event_id, const Handler::Args& args) noexcept {
        if (auto it = m_impl->event_dispatcher_bind_map.find(event_id);
            it != m_impl->event_dispatcher_bind_map.end()) {
            for (auto&& dispatcher : it->second) {
                dispatcher->DispatchImmediately(event_id, args);
            }
        }
    }

    void Center::DispatchImmediately(const char* dispatcher_id, const char* event_id, const Handler::Args& args) noexcept {
        if (auto dispatcher = GetDispatcher(dispatcher_id); dispatcher) {
            dispatcher->DispatchImmediately(event_id, args);
        }
    }

    void Center::Send(const char* event_id) noexcept {
        if (auto it = m_impl->event_dispatcher_bind_map.find(event_id);
            it != m_impl->event_dispatcher_bind_map.end()) {
            for (auto&& dispatcher : it->second) {
                dispatcher->Push(event_id);
            }
        }
    }

    void Center::Send(const char* event_id, const Handler::Args& args) noexcept {
        if (auto it = m_impl->event_dispatcher_bind_map.find(event_id);
            it != m_impl->event_dispatcher_bind_map.end()) {
            for (auto&& dispatcher : it->second) {
                dispatcher->Push(event_id, args);
            }
        }
    }

    void Center::RegisterDispatcher(const char* dispatcher_id) noexcept {
        if (GetDispatcher(dispatcher_id) == nullptr) {
            m_impl->dispatchers[dispatcher_id] = std::make_unique<Dispatcher>();
        }
    }

    Dispatcher* Center::GetDispatcher(const char* dispatcher_id) noexcept {
        if (auto it = m_impl->dispatchers.find(dispatcher_id);
            it != m_impl->dispatchers.end())
            return it->second.get();
        return nullptr;
    }
}// namespace Pupil::Event