#include "stream.h"

#include "util.h"
#include "util/log.h"

#include <array>
#include <queue>
#include <mutex>

namespace Pupil::cuda {
    Stream::Stream(std::string_view name) noexcept : m_name(name) {
        // CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreate(&m_stream));
        //CUDA_CHECK(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
    }
    Stream::~Stream() noexcept {
        if (m_stream) {
            Synchronize();
            CUDA_CHECK(cudaStreamDestroy(m_stream));
        }
        m_stream = nullptr;
        //CUDA_CHECK(cudaEventDestroy(m_event));
    }

    void Stream::Synchronize() noexcept { CUDA_CHECK(cudaStreamSynchronize(m_stream)); }

    Event::Event(bool use_timer) noexcept
        : m_stream(nullptr) {
        if (use_timer)
            CUDA_CHECK(cudaEventCreate(&m_event));
        else
            CUDA_CHECK(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
    }

    Event::~Event() noexcept {
        CUDA_CHECK(cudaEventDestroy(m_event));
    }

    void Event::Reset(Stream* stream) noexcept {
        CUDA_CHECK(cudaEventRecord(m_event, *stream));
        m_stream = stream;
    }

    void Event::Synchronize() noexcept {
        if (m_stream) {
            CUDA_CHECK(cudaStreamWaitEvent(*m_stream, m_event));
            m_stream = nullptr;
        }
    }

    static inline unsigned int GenericFfs(unsigned int x) {
        unsigned int r = 0;
        if (!x) return 0;

        if (!(x & 0xffff)) x >>= 16, r += 16;
        if (!(x & 0xff)) x >>= 8, r += 8;
        if (!(x & 0xf)) x >>= 4, r += 4;
        if (!(x & 3)) x >>= 2, r += 2;
        if (!(x & 1)) x >>= 1, r += 1;

        return r;
    }

    struct StreamManager::Impl {
        std::mutex mtx;

        std::list<util::Data<Stream>> stream_pool;

        std::list<util::Data<Stream>> stream_groups[32];

        util::Data<Stream> default_stream = nullptr;
    };

    StreamManager::StreamManager() noexcept {
        if (m_impl) return;

        m_impl = new Impl();

        m_impl->default_stream = std::make_unique<Stream>();
    }

    StreamManager::~StreamManager() noexcept {
    }

    util::CountableRef<Stream> StreamManager::Alloc(EStreamTaskType task_type) noexcept {
        return m_impl->default_stream.GetRef();
        std::unique_lock lock(m_impl->mtx);
        auto             group_idx = GenericFfs(static_cast<unsigned int>(task_type));

        if (m_impl->stream_pool.empty())
            m_impl->stream_groups[group_idx].emplace_back(std::make_unique<Stream>());
        else {
            m_impl->stream_groups[group_idx]
                .emplace_back(std::move(m_impl->stream_pool.front()));
            m_impl->stream_pool.pop_front();
        }
        return m_impl->stream_groups[group_idx].back().GetRef();
    }

    void StreamManager::Synchronize(EStreamTaskType task_type) noexcept {
        m_impl->default_stream->Synchronize();
        return;
        std::unique_lock lock(m_impl->mtx);
        for (auto i = 0u; i < 32u; ++i) {
            if (task_type & static_cast<EStreamTaskType>(1 << i)) {
                auto& st_gp = m_impl->stream_groups[i];
                for (auto it = st_gp.begin(); it != st_gp.end();) {
                    (*it)->Synchronize();
                    if (it->GetRefCount() == 0) {
                        m_impl->stream_pool.emplace_back(std::move(*it));
                        it = st_gp.erase(it);
                    } else
                        ++it;
                }
            }
        }
    }

    void StreamManager::Destroy() noexcept {
        Synchronize(EStreamTaskType::ALL);
        for (auto i = 0u; i < 32u; ++i) {
            auto& st_gp = m_impl->stream_groups[i];
            for (auto it = st_gp.begin(); it != st_gp.end();) {
                it = st_gp.erase(it);
            }
        }

        m_impl->stream_pool.clear();
        delete m_impl;
        m_impl = nullptr;
    }

}// namespace Pupil::cuda