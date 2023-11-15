#pragma once

#include <atomic>
#include <memory>

#include <assert.h>

namespace Pupil::util {
    class Counter {
    public:
        Counter() noexcept : count(0) {}
        ~Counter() noexcept { assert(count == 0); }

        Counter(const Counter&)            = delete;
        Counter& operator=(const Counter&) = delete;

        void Increase() noexcept { ++count; }
        void Decrease() noexcept {
            assert(count > 0);
            --count;
        }

        uint32_t GetCount() const noexcept { return count; }

    private:
        std::atomic_uint32_t count;
    };

    template<typename T>
    class CountableRef {
    public:
        CountableRef() noexcept : m_ptr(nullptr), m_cnter(nullptr) {}
        CountableRef(nullptr_t) noexcept : m_ptr(nullptr), m_cnter(nullptr) {}

        CountableRef(void* ptr, Counter* cnter) noexcept
            : m_ptr(static_cast<T*>(ptr)), m_cnter(cnter) { AddRef(); }

        CountableRef(CountableRef const& other) noexcept
            : m_ptr(other.m_ptr), m_cnter(other.m_cnter) { AddRef(); }

        template<typename U>
        CountableRef(CountableRef<U> const& other) noexcept
            : m_ptr(static_cast<T*>(other.m_ptr)), m_cnter(other.m_cnter) { AddRef(); }

        template<typename U>
        CountableRef(CountableRef<U>&& other) noexcept
            : m_cnter(std::exchange(other.m_cnter, {})) {
            m_ptr = static_cast<T*>(std::exchange(other.m_ptr, {}));
        }

        ~CountableRef() noexcept { ReleaseRef(); }

        CountableRef& operator=(nullptr_t) noexcept {
            Reset();
            return *this;
        }

        CountableRef& operator=(CountableRef const& other) noexcept {
            CopyRef(other);
            return *this;
        }

        CountableRef& operator=(CountableRef&& other) noexcept {
            if (this == &other) return *this;
            ReleaseRef();
            m_ptr   = std::exchange(other.m_ptr, {});
            m_cnter = std::exchange(other.m_cnter, {});

            return *this;
        }

        template<typename U>
        CountableRef& operator=(CountableRef<U> const& other) noexcept {
            ReleaseRef();
            m_ptr   = static_cast<T*>(other.m_ptr);
            m_cnter = other.m_cnter;
            AddRef();
            return *this;
        }

        template<typename U>
        CountableRef& operator=(CountableRef<U>&& other) noexcept {
            ReleaseRef();
            m_ptr   = static_cast<T*>(std::exchange(other.m_ptr, {}));
            m_cnter = std::exchange(other.m_cnter, {});
            return *this;
        }

        operator bool() const noexcept { return m_ptr != nullptr; }

        auto operator->() const noexcept { return m_ptr; }
        T&   operator*() const noexcept { return *m_ptr; }
        T*   Get() const noexcept { return m_ptr; }

        template<typename U>
        U* As() const noexcept { return static_cast<U*>(m_ptr); }

        void Reset(void* ptr = nullptr, Counter* cnter = nullptr) noexcept {
            ReleaseRef();
            m_ptr   = static_cast<T*>(ptr);
            m_cnter = cnter;
            AddRef();
        }

        uint32_t GetRefCount() const noexcept { return m_cnter ? m_cnter->GetCount() : 0; }

    private:
        void AddRef() noexcept {
            if (!m_ptr) return;
            if (!m_cnter) return;
            m_cnter->Increase();
        }

        void ReleaseRef() noexcept {
            if (!m_ptr) return;
            if (!m_cnter) return;
            m_cnter->Decrease();
        }

        void CopyRef(CountableRef const& other) noexcept {
            if (m_ptr == other.m_ptr) return;
            ReleaseRef();
            m_ptr   = other.m_ptr;
            m_cnter = other.m_cnter;
            AddRef();
        }

    private:
        template<typename U>
        friend class CountableRef;

        T*       m_ptr;
        Counter* m_cnter;
    };

    template<typename T>
    class Data {
    public:
        Data(void* data) noexcept {
            m_data.reset(static_cast<T*>(data));
            m_cnter = std::make_unique<Counter>();
        }

        template<typename U = T>
        Data(CountableRef<U> ref) noexcept {
            m_data.reset(ref.m_ptr);
            m_cnter.reset(ref.m_cnter);
        }

        template<typename U = T>
        Data(std::unique_ptr<U> data) noexcept {
            m_data.reset(data.release());
            m_cnter = std::make_unique<Counter>();
        }

        Data(Data&& other) noexcept {
            m_data  = std::move(other.m_data);
            m_cnter = std::move(other.m_cnter);
        }

        Data& operator=(Data&& other) noexcept {
            m_data  = std::move(other.m_data);
            m_cnter = std::move(other.m_cnter);
            return *this;
        }

        ~Data() noexcept { Release(); }

        void Release() noexcept {
            assert(m_data == nullptr || m_cnter->GetCount() == 0);
            m_data.reset();
            m_cnter.reset();
        }

        template<typename U = T>
        Data<U> Clone() const noexcept {
            Data<U> clone = m_data->Clone();
            return std::move(clone);
        }

        CountableRef<T> GetRef() const noexcept { return CountableRef<T>(m_data.get(), m_cnter.get()); }
        uint32_t        GetRefCount() const noexcept { return m_cnter->GetCount(); }
        auto            operator->() const noexcept { return m_data.get(); }
        T&              operator*() const noexcept { return *m_data; }
        T*              Get() const noexcept { return m_data.get(); }

    private:
        template<typename U>
        friend class Data;

        std::unique_ptr<T>       m_data;
        std::unique_ptr<Counter> m_cnter;
    };
}// namespace Pupil::util