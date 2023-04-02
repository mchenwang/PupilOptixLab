#pragma once

#include "util/util.h"
#include "pass.h"
#include "resource.h"

#include <d3d12.h>
#include <winrt/base.h>

#include <unordered_map>
#include <functional>
#include <memory>
#include <array>
#include <mutex>

namespace Pupil {
struct Buffer;
enum class EWindowEvent {
    Quit,
    Resize,
    Minimized
};

enum class ECanvasEvent {
    MouseDragging,
    MouseWheel,
    CameraMove
};

class GuiPass : public Pass, public util::Singleton<GuiPass> {
public:
    constexpr static uint32_t SWAP_BUFFER_NUM = 2;
    constexpr static std::array<std::string_view, SWAP_BUFFER_NUM>
        OUTPUT_FLIP_BUFFER = {
            "output flip buffer0", "output flip buffer1"
        };
    constexpr static std::array<std::string_view, SWAP_BUFFER_NUM>
        OUTPUT_FLIP_TEXTURE = {
            "output flip texture0", "output flip texture1"
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

    void FlipSwapBuffer() noexcept;
    [[nodiscard]] uint32_t GetCurrentRenderOutputBufferIndex() const noexcept { return m_current_buffer_index; }
    [[nodiscard]] auto &GetCurrentRenderOutputBuffer() const noexcept { return m_flip_buffers[m_current_buffer_index]; }
    [[nodiscard]] uint32_t GetReadyOutputBufferIndex() const noexcept { return m_ready_buffer_index; }
    [[nodiscard]] auto &GetReadyOutputBuffer() const noexcept { return m_flip_buffers[m_ready_buffer_index]; }

protected:
    void OnDraw() noexcept;
    void InitRenderToTexturePipeline() noexcept;
    void RenderFlipBufferToTexture(winrt::com_ptr<ID3D12GraphicsCommandList>) noexcept;

    std::unordered_map<std::string, CustomInspector> m_inspectors;
    bool m_init_flag = false;
    bool m_copy_after_flip_flag = false;

    // one for rendering output, the other for showing on gui
    struct FlipBuffer {
        winrt::com_ptr<ID3D12Resource> res = nullptr;
        D3D12_GPU_DESCRIPTOR_HANDLE output_buffer_srv{};
        D3D12_GPU_DESCRIPTOR_HANDLE output_texture_srv{};
        D3D12_CPU_DESCRIPTOR_HANDLE output_rtv{};
        SharedBuffer shared_buffer{};
    };

    FlipBuffer m_flip_buffers[SWAP_BUFFER_NUM];

    uint32_t m_current_buffer_index = 0;
    uint32_t m_ready_buffer_index = 1;
    std::mutex m_flip_model_mutex;
    uint32_t m_output_w = 0;
    uint32_t m_output_h = 0;

    // render buffer to texture
    winrt::com_ptr<ID3D12RootSignature> m_root_signature;
    winrt::com_ptr<ID3D12PipelineState> m_pipeline_state;

    winrt::com_ptr<ID3D12Resource> m_vb;
    D3D12_VERTEX_BUFFER_VIEW m_vbv;

    winrt::com_ptr<ID3D12Resource> m_frame_constant_buffer;
    void *m_frame_constant_buffer_mapped_ptr = nullptr;
};
}// namespace Pupil