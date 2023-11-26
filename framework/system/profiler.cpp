#include "profiler.h"
#include "util/hash.h"

#include "cuda/timer.h"

#include "imgui.h"

#include <unordered_map>
#include <thread>

namespace Pupil {
    struct ProfilerUnit {
        std::unique_ptr<Timer> timer;
        bool                   enabled;
        int                    max_buffer_size;
        int                    buffer_index;
        std::mutex             buffer_mtx;
        std::vector<float>     time_buffer;
    };

    struct Profiler::Impl {
        std::unordered_map<std::string, size_t, util::StringHash, std::equal_to<>> timers_map;

        std::vector<std::unique_ptr<ProfilerUnit>> units;
    };

    Profiler::Profiler() noexcept {
        m_impl = new Impl();
    }

    Profiler::~Profiler() noexcept {
        if (m_impl) delete m_impl;
        m_impl = nullptr;
    }

    void Profiler::Run() noexcept {
        int size = m_impl->units.size();
        for (int i = 0; i < size; ++i) {
            auto& unit = m_impl->units[i];
            if (float ms; unit->timer->TryGetElapsedMilliseconds(ms)) {
                std::unique_lock lock(unit->buffer_mtx);
                if (unit->buffer_index == unit->max_buffer_size)
                    unit->buffer_index = 0;
                unit->time_buffer[unit->buffer_index++] = ms;
            }
        }
    }

    Timer* Profiler::AllocTimer(std::string_view name, int buffer_size) noexcept {
        auto unit             = std::make_unique<ProfilerUnit>();
        unit->timer           = std::make_unique<Pupil::TracedCpuTimer>();
        unit->enabled         = true;
        unit->max_buffer_size = buffer_size;
        unit->buffer_index    = 0;
        unit->time_buffer.resize(unit->max_buffer_size, 0.f);

        m_impl->timers_map.emplace(name, m_impl->units.size());
        m_impl->units.emplace_back(std::move(unit));
        return m_impl->units.back()->timer.get();
    }

    Timer* Profiler::AllocTimer(std::string_view name, const util::CountableRef<cuda::Stream>& stream, int buffer_size) noexcept {
        auto unit             = std::make_unique<ProfilerUnit>();
        unit->timer           = std::make_unique<Pupil::cuda::TracedGpuTimer>(stream);
        unit->enabled         = true;
        unit->max_buffer_size = buffer_size;
        unit->buffer_index    = 0;
        unit->time_buffer.resize(unit->max_buffer_size, 0.f);

        m_impl->timers_map.emplace(name, m_impl->units.size());
        m_impl->units.emplace_back(std::move(unit));
        return m_impl->units.back()->timer.get();
    }

    void Profiler::Enable(std::string_view name) noexcept {
        if (auto it = m_impl->timers_map.find(name);
            it != m_impl->timers_map.end()) {
            m_impl->units[it->second]->enabled = true;
        }
    }

    void Profiler::Disable(std::string_view name) noexcept {
        if (auto it = m_impl->timers_map.find(name);
            it != m_impl->timers_map.end()) {
            m_impl->units[it->second]->enabled = false;
        }
    }

    Profiler::Entry Profiler::GetEntry(std::string_view name) noexcept {
        Profiler::Entry entry{
            .datas  = nullptr,
            .count  = 0,
            .offset = 0};

        if (auto it = m_impl->timers_map.find(name);
            it != m_impl->timers_map.end()) {
            auto& unit  = m_impl->units[it->second];
            entry.datas = unit->time_buffer.data();
            entry.count = unit->max_buffer_size;

            std::unique_lock lock(unit->buffer_mtx);
            entry.offset = unit->buffer_index;
        }
        return entry;
    }

    void Profiler::ShowPlot(std::string_view name) noexcept {
        auto profiler_entry = GetEntry(name);
        if (profiler_entry.datas == nullptr) return;

        ImGui::Text("speed:");
        ImGui::PlotLines("##plot", profiler_entry.datas, profiler_entry.count, profiler_entry.offset, NULL, FLT_MAX, FLT_MAX, ImVec2{0.f, 20.f});
        ImGui::SameLine();
        float max_speed = *std::max_element(profiler_entry.datas, profiler_entry.datas + profiler_entry.count);
        float min_speed = *std::min_element(profiler_entry.datas, profiler_entry.datas + profiler_entry.count);
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX(), ImGui::GetCursorPosY() - 6.f));
        ImGui::Text("min: %-3.4f ms\nmax: %-3.4f ms", min_speed, max_speed);
    }
}// namespace Pupil