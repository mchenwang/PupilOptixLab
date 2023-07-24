#pragma once

#include "util/util.h"

#include "../pass.h"
#include "../buffer.h"
#include "cuda/stream.h"

#include <d3d12.h>
#include <winrt/base.h>

#include <vector>
#include <functional>
#include <memory>
#include <array>
#include <mutex>
#include <atomic>

namespace Pupil {
struct Buffer;
enum class EWindowEvent {
    Quit,
    Resize,
    Minimized
};

enum class ECanvasEvent {
    Resize,
    Display,
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
    struct {
        bool console = true;
        bool scene = true;
        bool bottom = false;
    } show_window;

    GuiPass() noexcept : Pass("GUI") {}

    virtual void OnRun() noexcept override;

    void Init() noexcept;
    void Destroy() noexcept;
    void Resize(uint32_t, uint32_t) noexcept;
    void ResizeCanvas(uint32_t w, uint32_t h) noexcept;
    void UpdateCanvasOutput() noexcept;
    void AdjustWindowSize() noexcept;

    using CustomInspector = std::function<void()>;
    void RegisterInspector(std::string_view, CustomInspector &&) noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }

    void FlipSwapBuffer() noexcept;
    [[nodiscard]] uint32_t GetCurrentRenderOutputBufferIndex() const noexcept { return m_current_buffer_index.load(); }
    [[nodiscard]] auto &GetCurrentRenderOutputBuffer() const noexcept { return m_flip_buffers[m_current_buffer_index.load()]; }
    [[nodiscard]] uint32_t GetReadyOutputBufferIndex() const noexcept { return m_ready_buffer_index.load(); }
    [[nodiscard]] auto &GetReadyOutputBuffer() const noexcept { return m_flip_buffers[m_ready_buffer_index.load()]; }

protected:
    void OnDraw() noexcept;
    void Docking() noexcept;
    void Menu(bool show = true) noexcept;
    void Console(bool show = true) noexcept;
    void Canvas(bool show = true) noexcept;
    void Scene(bool show = true) noexcept;
    void Bottom(bool show = true) noexcept;

    void InitRenderToTexturePipeline() noexcept;
    void RenderFlipBufferToTexture(winrt::com_ptr<ID3D12GraphicsCommandList>) noexcept;

    std::vector<std::pair<std::string, CustomInspector>> m_inspectors;
    bool m_init_flag = false;
    std::atomic_bool m_render_flip_buffer_to_texture_flag = false;
    std::unique_ptr<cuda::Stream> m_memcpy_stream = nullptr;

    // one for rendering output, the other for showing on gui
    struct FlipBuffer {
        winrt::com_ptr<ID3D12Resource> res = nullptr;
        Buffer *system_buffer = nullptr;
        D3D12_GPU_DESCRIPTOR_HANDLE output_buffer_srv{};
        D3D12_GPU_DESCRIPTOR_HANDLE output_texture_srv{};
        D3D12_CPU_DESCRIPTOR_HANDLE output_rtv{};
    };

    FlipBuffer m_flip_buffers[SWAP_BUFFER_NUM];

    std::atomic<uint32_t> m_current_buffer_index = 0;
    std::atomic<uint32_t> m_ready_buffer_index = 1;
    std::mutex m_flip_model_mutex;

    // render buffer to texture
    winrt::com_ptr<ID3D12RootSignature> m_root_signature;
    winrt::com_ptr<ID3D12PipelineState> m_pipeline_state;

    winrt::com_ptr<ID3D12Resource> m_vb;
    D3D12_VERTEX_BUFFER_VIEW m_vbv;

    winrt::com_ptr<ID3D12Resource> m_frame_constant_buffer;
    void *m_frame_constant_buffer_mapped_ptr = nullptr;
};
}// namespace Pupil