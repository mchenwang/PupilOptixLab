#pragma once

#include "util/util.h"
#include "pass.h"

#include <unordered_map>
#include <functional>
#include <memory>
#include <array>
#include <mutex>

namespace Pupil {
struct Buffer;
enum class WindowEvent {
    Quit,
    Resize,
    Minimized,
    // MouseLeftButtonMove,
    // MouseRightButtonMove,
    // MouseWheel,
    // KeyboardMove
};

class GuiPass : public Pass, public util::Singleton<GuiPass> {
public:
    constexpr static uint32_t SWAP_BUFFER_NUM = 2;
    constexpr static std::array<std::string_view, SWAP_BUFFER_NUM>
        RENDER_OUTPUT_BUFFER = {
            "render output buffer0", "render output buffer1"
        };

    GuiPass() noexcept : Pass("GUI") {}

    virtual void Run() noexcept override;
    virtual void SetScene(scene::Scene *) noexcept override;
    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void Init() noexcept;
    void Destroy() noexcept;
    void Resize(uint32_t, uint32_t) noexcept;
    void AdjustWindowSize() noexcept;

    using CustomInspector = std::function<void()>;
    void RegisterInspector(std::string_view, CustomInspector &&) noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }

    void FlipSwapBuffer() noexcept {
        std::scoped_lock lock{ m_flip_model_mutex, m_render_output_buffer_mutex[m_ready_buffer_index] };
        m_ready_buffer_index = m_current_buffer_index;
        m_current_buffer_index = (m_current_buffer_index + 1) % SWAP_BUFFER_NUM;
    }
    [[nodiscard]] uint32_t GetCurrentRenderOutputBufferIndex() const noexcept { return m_current_buffer_index; }
    [[nodiscard]] Buffer *GetCurrentRenderOutputBuffer() const noexcept { return m_render_output_buffers[m_current_buffer_index]; }
    [[nodiscard]] uint32_t GetReadyOutputBufferIndex() const noexcept { return m_ready_buffer_index; }
    [[nodiscard]] Buffer *GetReadyOutputBuffer() const noexcept { return m_render_output_buffers[m_ready_buffer_index]; }

protected:
    void OnDraw() noexcept;

    std::unordered_map<std::string, CustomInspector> m_inspectors;
    bool m_init_flag = false;

    // one for rendering output, the other for showing on gui
    Buffer *m_render_output_buffers[2]{};
    uint64_t m_render_output_srvs[2]{};
    uint32_t m_current_buffer_index = 0;
    uint32_t m_ready_buffer_index = 1;
    std::mutex m_render_output_buffer_mutex[2]{};
    std::mutex m_flip_model_mutex;
    float m_render_output_show_w = 0.f;
    float m_render_output_show_h = 0.f;
};
}// namespace Pupil