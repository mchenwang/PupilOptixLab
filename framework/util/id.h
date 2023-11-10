#pragma once

#include <vector>
#include <queue>
#include <mutex>

namespace Pupil::util {
class UintIdAllocator {
public:
    UintIdAllocator() noexcept : m_next_id(0) {}

    UintIdAllocator(const UintIdAllocator &) = delete;
    UintIdAllocator &operator=(const UintIdAllocator &) = delete;

    uint64_t Allocate() noexcept {
        std::unique_lock lock(m_mtx);
        if (m_cycle_pool.empty()) return m_next_id++;
        auto id = m_cycle_pool.front();
        m_cycle_pool.pop();
        return id;
    }
    void Recycle(uint64_t id) noexcept {
        std::unique_lock lock(m_mtx);
        m_cycle_pool.push(id);
    }

private:
    std::mutex m_mtx;
    std::queue<uint64_t> m_cycle_pool;
    uint64_t m_next_id;
};
}// namespace Pupil::util