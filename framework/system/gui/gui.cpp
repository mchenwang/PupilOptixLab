#include "gui.h"
#include "../system.h"
#include "../buffer.h"

#include "dx12/context.h"
#include "dx12/d3dx12.h"

#include "util/event.h"
#include "util/camera.h"
#include "util/thread_pool.h"
#include "util/texture.h"
#include "resource/scene.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include "imfilebrowser.h"
#include "ImGuizmo/ImGuizmo.h"
#include "world/render_object.h"
#include "world/world.h"

#include "cuda/util.h"

#include "buffer_to_canvas.cuh"

#include "static.h"

#include <format>
#include <d3dcompiler.h>

namespace Pupil {
HWND g_window_handle;
uint32_t g_window_w = 1280;
uint32_t g_window_h = 720;
}// namespace Pupil

namespace {
const std::wstring WND_NAME = L"PupilOptixLab";
const std::wstring WND_CLASS_NAME = L"PupilOptixLab_CLASS";
HINSTANCE m_instance;

ImGui::FileBrowser m_scene_file_browser;

std::atomic_bool m_waiting_scene_load = false;
bool m_render_flag = false;
double m_flip_rate = 1.;

int m_canvas_display_buffer_index = 0;
std::string_view m_canvas_display_buffer_name = Pupil::BufferManager::DEFAULT_FINAL_RESULT_BUFFER_NAME;

std::vector<Pupil::world::RenderObject *> m_render_objects;
Pupil::world::RenderObject *m_selected_ro = nullptr;
ImGuizmo::MODE m_zmo_mode = ImGuizmo::WORLD;
ImGuizmo::OPERATION m_zmo_operation = ImGuizmo::TRANSLATE;

struct CanvasDisplayDesc {
    uint32_t w = 1;
    uint32_t h = 1;
    uint32_t tone_mapping = 0;
    uint32_t gamma_correct = 1;
} m_canvas_desc;
}// namespace

namespace {
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// void OnMouseDown(WPARAM, LONG, LONG) noexcept;
// void OnMouseUp(WPARAM, LONG, LONG) noexcept;
// void OnMouseMove(WPARAM, LONG, LONG) noexcept;
// void OnMouseWheel(short) noexcept;
// void OnKeyDown(WPARAM) noexcept;
// void OnKeyUp(WPARAM) noexcept;
}// namespace

namespace Pupil {
void GuiPass::Init() noexcept {
    // create window
    {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(WNDCLASSEXW);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = &WndProc;
        wc.cbClsExtra = 0;
        wc.cbWndExtra = 0;
        wc.hInstance = GetModuleHandleW(NULL);
        wc.hIcon = NULL;
        wc.hCursor = ::LoadCursorW(NULL, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wc.lpszMenuName = NULL;
        wc.lpszClassName = WND_CLASS_NAME.data();

        ::RegisterClassExW(&wc);

        RECT window_rect{ 0, 0, static_cast<LONG>(g_window_w), static_cast<LONG>(g_window_h) };
        ::AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, FALSE);

        int screen_w = ::GetSystemMetrics(SM_CXSCREEN);
        int screen_h = ::GetSystemMetrics(SM_CYSCREEN);

        int window_w = window_rect.right - window_rect.left;
        int window_h = window_rect.bottom - window_rect.top;

        // Center the window within the screen. Clamp to 0, 0 for the top-left corner.
        int window_x = std::max<int>(0, (screen_w - window_w) / 2);
        int window_y = std::max<int>(0, (screen_h - window_h) / 2);

        m_instance = GetModuleHandleW(NULL);
        g_window_handle = ::CreateWindowExW(
            NULL,
            WND_CLASS_NAME.data(),
            WND_NAME.data(),
            WS_OVERLAPPEDWINDOW,
            window_x, window_y, window_w, window_h,
            NULL, NULL, wc.hInstance, NULL);

        ::ShowWindow(g_window_handle, SW_SHOW);
        ::UpdateWindow(g_window_handle);
    }

    // event binding
    {
        EventBinder<EWindowEvent::Resize>([this](void *param) {
            struct {
                uint32_t w, h;
            } size;
            size = *static_cast<decltype(size) *>(param);
            this->Resize(size.w, size.h);
        });

        EventBinder<ESystemEvent::StartRendering>([](void *) {
            m_waiting_scene_load = false;
            m_render_flag = true;
        });

        EventBinder<ESystemEvent::StopRendering>([](void *) {
            m_render_flag = false;
        });

        EventBinder<ESystemEvent::FrameFinished>([this](void *p) {
            double time_count = *(double *)p;
            m_flip_rate = 1000. / time_count;
        });

        EventBinder<ECanvasEvent::Display>([this](void *p) {
            auto buffer_name = reinterpret_cast<std::string_view *>(p);
            m_canvas_display_buffer_name = *buffer_name;
        });

        EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
            auto world = reinterpret_cast<world::World *>(p);
            m_selected_ro = nullptr;
            if (world)
                m_render_objects = world->GetRenderobjects();
        });
    }

    // init imgui
    {
        auto dx_ctx =
            util::Singleton<DirectX::Context>::instance();
        dx_ctx->Init(g_window_w, g_window_h, g_window_handle);

        ImGui::CreateContext();
        auto &io = ImGui::GetIO();
        (void)io;
        ImGui::StyleColorsDark();
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        ImGui_ImplWin32_Init(g_window_handle);

        ImGui_ImplDX12_Init(
            dx_ctx->device.get(),
            dx_ctx->FRAMES_NUM,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            dx_ctx->srv_heap.get(),
            dx_ctx->srv_heap->GetCPUDescriptorHandleForHeapStart(),
            dx_ctx->srv_heap->GetGPUDescriptorHandleForHeapStart());

        m_scene_file_browser.SetTitle("scene browser");
        m_scene_file_browser.SetTypeFilters({ ".xml" });

        std::filesystem::path scene_data_path{ DATA_DIR };
        m_scene_file_browser.SetPwd(scene_data_path);
    }

    InitRenderToTexturePipeline();

    m_memcpy_stream = std::make_unique<cuda::Stream>();

    m_init_flag = true;
}

void GuiPass::ResizeCanvas(uint32_t w, uint32_t h) noexcept {
    if (w == m_canvas_desc.w && h == m_canvas_desc.h) return;
    m_canvas_desc.h = h;
    m_canvas_desc.w = w;

    auto dx_ctx = util::Singleton<DirectX::Context>::instance();
    dx_ctx->Flush();
    // init render output buffers
    {
        auto buffer_mngr = util::Singleton<BufferManager>::instance();
        uint64_t size = static_cast<uint64_t>(h) * w * sizeof(float) * 4;

        auto srv_descriptor_handle_size = dx_ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto rtv_descriptor_handle_size = dx_ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        auto srv_cpu_handle = dx_ctx->srv_heap->GetCPUDescriptorHandleForHeapStart();
        auto srv_gpu_handle = dx_ctx->srv_heap->GetGPUDescriptorHandleForHeapStart();
        auto rtv_cpu_handle = dx_ctx->rtv_heap->GetCPUDescriptorHandleForHeapStart();
        auto rtv_gpu_handle = dx_ctx->rtv_heap->GetGPUDescriptorHandleForHeapStart();

        rtv_cpu_handle.ptr += rtv_descriptor_handle_size * 2;
        rtv_gpu_handle.ptr += rtv_descriptor_handle_size * 2;

        for (auto i = 0u; i < SWAP_BUFFER_NUM; ++i) {
            auto d3d12_res_desc = CD3DX12_RESOURCE_DESC::Tex2D(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                static_cast<UINT>(w), static_cast<UINT>(h),
                1, 0, 1, 0,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            auto properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            winrt::com_ptr<ID3D12Resource> temp_res;
            DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
                &properties, D3D12_HEAP_FLAG_NONE, &d3d12_res_desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr,
                winrt::guid_of<ID3D12Resource>(), temp_res.put_void()));
            m_flip_buffers[i].res = temp_res;
            m_flip_buffers[i].res->SetName(std::wstring{ OUTPUT_FLIP_TEXTURE[i].begin(), OUTPUT_FLIP_TEXTURE[i].end() }.c_str());

            srv_gpu_handle.ptr += srv_descriptor_handle_size;
            srv_cpu_handle.ptr += srv_descriptor_handle_size;
            m_flip_buffers[i].output_texture_srv = srv_gpu_handle;

            D3D12_SHADER_RESOURCE_VIEW_DESC tex_srv_desc{};
            tex_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            tex_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            tex_srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            tex_srv_desc.Texture2D.MipLevels = 1;
            dx_ctx->device->CreateShaderResourceView(m_flip_buffers[i].res.get(), &tex_srv_desc, srv_cpu_handle);

            rtv_cpu_handle.ptr += rtv_descriptor_handle_size;
            rtv_gpu_handle.ptr += rtv_descriptor_handle_size;
            m_flip_buffers[i].output_rtv = rtv_cpu_handle;

            dx_ctx->device->CreateRenderTargetView(m_flip_buffers[i].res.get(), nullptr, rtv_cpu_handle);

            srv_gpu_handle.ptr += srv_descriptor_handle_size;
            srv_cpu_handle.ptr += srv_descriptor_handle_size;
            m_flip_buffers[i].output_buffer_srv = srv_gpu_handle;

            BufferDesc buf_desc{
                .name = OUTPUT_FLIP_BUFFER[i].data(),
                .flag = EBufferFlag::SharedWithDX12,
                .width = w,
                .height = h,
                .stride_in_byte = sizeof(float) * 4
            };
            m_flip_buffers[i].system_buffer = buffer_mngr->AllocBuffer(buf_desc);

            D3D12_SHADER_RESOURCE_VIEW_DESC buf_srv_desc{};
            buf_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            buf_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            buf_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
            buf_srv_desc.Buffer.NumElements = static_cast<UINT>(m_canvas_desc.h * m_canvas_desc.w);
            buf_srv_desc.Buffer.StructureByteStride = sizeof(float) * 4;
            buf_srv_desc.Buffer.FirstElement = 0;
            buf_srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            dx_ctx->device->CreateShaderResourceView(m_flip_buffers[i].system_buffer->dx12_ptr.get(), &buf_srv_desc, srv_cpu_handle);
        }
    }

    Pupil::Log::Info("Canvas resize to {}x{}", m_canvas_desc.w, m_canvas_desc.h);
}

void GuiPass::UpdateCanvasOutput() noexcept {
    auto buf_mngr = util::Singleton<BufferManager>::instance();
    auto canvas_output = buf_mngr->GetBuffer(m_canvas_display_buffer_name);
    if (canvas_output->desc.height != m_canvas_desc.h ||
        canvas_output->desc.width != m_canvas_desc.w) {
        ResizeCanvas(canvas_output->desc.width, canvas_output->desc.height);
    }

    if (canvas_output->desc.stride_in_byte == sizeof(float4)) {
        auto dst_buf = GetReadyOutputBuffer().system_buffer;
        cudaMemcpyAsync(reinterpret_cast<void *>(dst_buf->cuda_ptr),
                        reinterpret_cast<void *>(canvas_output->cuda_ptr),
                        static_cast<size_t>(dst_buf->desc.width * dst_buf->desc.height * dst_buf->desc.stride_in_byte),
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice, *m_memcpy_stream.get());
    } else if (canvas_output->desc.stride_in_byte == sizeof(float3)) {
        auto dst_buf = GetReadyOutputBuffer().system_buffer;
        Pupil::CopyFloat3BufferToCanvas(dst_buf->cuda_ptr, canvas_output->cuda_ptr,
                                        dst_buf->desc.width * dst_buf->desc.height, m_memcpy_stream.get());
    } else if (canvas_output->desc.stride_in_byte == sizeof(float2)) {
        auto dst_buf = GetReadyOutputBuffer().system_buffer;
        Pupil::CopyFloat2BufferToCanvas(dst_buf->cuda_ptr, canvas_output->cuda_ptr,
                                        dst_buf->desc.width * dst_buf->desc.height, m_memcpy_stream.get());
    } else if (canvas_output->desc.stride_in_byte == sizeof(float)) {
        auto dst_buf = GetReadyOutputBuffer().system_buffer;
        Pupil::CopyFloat1BufferToCanvas(dst_buf->cuda_ptr, canvas_output->cuda_ptr,
                                        dst_buf->desc.width * dst_buf->desc.height, m_memcpy_stream.get());
    }
}

void GuiPass::Destroy() noexcept {
    if (!IsInitialized()) return;

    util::Singleton<DirectX::Context>::instance()->Flush();
    m_memcpy_stream.reset();

    if (m_frame_constant_buffer_mapped_ptr) {
        m_frame_constant_buffer_mapped_ptr = nullptr;
        m_frame_constant_buffer->Unmap(0, nullptr);
    }

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    for (auto &&buffer : m_flip_buffers) buffer.res = nullptr;
    m_root_signature = nullptr;
    m_pipeline_state = nullptr;
    m_vb = nullptr;
    m_frame_constant_buffer = nullptr;

    ::DestroyWindow(g_window_handle);
    ::UnregisterClassW(WND_CLASS_NAME.data(), m_instance);
    m_init_flag = false;
}

void GuiPass::Resize(uint32_t w, uint32_t h) noexcept {
    if (!IsInitialized()) return;

    if (w != g_window_w || h != g_window_h) {
        g_window_w = w;
        g_window_h = h;
        util::Singleton<DirectX::Context>::instance()->Resize(w, h);
    }
}

void GuiPass::AdjustWindowSize() noexcept {
    if (!IsInitialized()) return;

    RECT window_rect{ 0, 0, static_cast<LONG>(g_window_w), static_cast<LONG>(g_window_h) };
    ::AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, FALSE);

    int window_w = window_rect.right - window_rect.left;
    int window_h = window_rect.bottom - window_rect.top;

    ::SetWindowPos(g_window_handle, 0, 0, 0, window_w, window_h, SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
}

void GuiPass::RegisterInspector(std::string_view name, CustomInspector &&inspector) noexcept {
    m_inspectors.emplace_back(name, inspector);
}

void GuiPass::FlipSwapBuffer() noexcept {
    std::scoped_lock lock{ m_flip_model_mutex };
    if (m_waiting_scene_load) return;

    m_current_buffer_index = m_ready_buffer_index.exchange(m_current_buffer_index);
    UpdateCanvasOutput();
    m_render_flip_buffer_to_texture_flag = true;
}

void GuiPass::OnRun() noexcept {
    if (!IsInitialized()) return;

    MSG msg = {};
    if (::PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
    }
    OnDraw();

    if (msg.message == WM_QUIT)
        EventDispatcher<EWindowEvent::Quit>();
}

void GuiPass::RenderFlipBufferToTexture(winrt::com_ptr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    auto dx_ctx = util::Singleton<DirectX::Context>::instance();

    cmd_list->SetGraphicsRootSignature(m_root_signature.get());
    cmd_list->SetPipelineState(m_pipeline_state.get());

    auto &buffer = GetReadyOutputBuffer();
    // ID3D12DescriptorHeap *heaps[] = { dx_ctx->srv_heap.get() };
    // cmd_list->SetDescriptorHeaps(1, heaps);
    cmd_list->SetGraphicsRootDescriptorTable(0, buffer.output_buffer_srv);
    memcpy(m_frame_constant_buffer_mapped_ptr, &m_canvas_desc, sizeof(CanvasDisplayDesc));
    cmd_list->SetGraphicsRootConstantBufferView(1, m_frame_constant_buffer->GetGPUVirtualAddress());

    D3D12_VIEWPORT viewport{ 0.f, 0.f, (FLOAT)m_canvas_desc.w, (FLOAT)m_canvas_desc.h, D3D12_MIN_DEPTH, D3D12_MAX_DEPTH };
    cmd_list->RSSetViewports(1, &viewport);
    D3D12_RECT rect{ 0, 0, (LONG)m_canvas_desc.w, (LONG)m_canvas_desc.h };
    cmd_list->RSSetScissorRects(1, &rect);

    auto rt = buffer.res;
    auto rtv = buffer.output_rtv;

    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = rt.get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd_list->ResourceBarrier(1, &barrier);
    }

    cmd_list->OMSetRenderTargets(1, &rtv, TRUE, nullptr);
    const FLOAT clear_color[4]{ 0.f, 0.f, 0.f, 1.f };
    cmd_list->ClearRenderTargetView(rtv, clear_color, 0, nullptr);

    cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmd_list->IASetVertexBuffers(0, 1, &m_vbv);
    cmd_list->DrawInstanced(4, 1, 0, 0);

    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = rt.get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd_list->ResourceBarrier(1, &barrier);
    }
}

void GuiPass::Docking() noexcept {
    static bool first_draw_call = true;
    ImGuiID main_node_id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    if (first_draw_call) {
        first_draw_call = false;

        ImGui::DockBuilderRemoveNode(main_node_id);
        ImGui::DockBuilderAddNode(main_node_id, ImGuiDockNodeFlags_None);

        // Make the dock node's size and position to match the viewport
        ImGui::DockBuilderSetNodeSize(main_node_id, ImGui::GetMainViewport()->WorkSize);
        ImGui::DockBuilderSetNodePos(main_node_id, ImGui::GetMainViewport()->WorkPos);

        ImGuiID dock_main_id = main_node_id;
        ImGuiID dock_left_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.2f, nullptr, &dock_main_id);
        ImGuiID dock_bottom_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.3f, nullptr, &dock_main_id);
        ImGuiID dock_right_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.26f, nullptr, &dock_main_id);

        ImGui::DockBuilderDockWindow("Console", dock_left_id);
        ImGui::DockBuilderDockWindow("Scene", dock_right_id);
        ImGui::DockBuilderDockWindow("Canvas", dock_main_id);
        ImGui::DockBuilderDockWindow("Bottom", dock_bottom_id);

        ImGui::DockBuilderFinish(dock_main_id);
    }
}

void GuiPass::Menu(bool show) noexcept {
    if (!show) return;

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Menu")) {
            if (ImGui::MenuItem("Load Scene")) {
                m_scene_file_browser.Open();
            }
            if (ImGui::MenuItem("Screenshot")) {
                std::filesystem::path path{ ROOT_DIR };

                auto time = std::chrono::zoned_time{
                    std::chrono::current_zone(),
                    std::chrono::system_clock::now()
                };

                auto time_str = std::format(
                    "{:%H.%M.%S.%m.%d}",
                    floor<std::chrono::seconds>(time.get_local_time()));
                path /= "screenshot(" + time_str + ").exr";

                size_t size = m_canvas_desc.h * m_canvas_desc.w * 4;
                auto image = std::make_unique<float[]>(size);
                memset(image.get(), 0, size);
                auto &buffer = GetCurrentRenderOutputBuffer();
                cuda::CudaMemcpyToHost(image.get(), buffer.system_buffer->cuda_ptr, size * sizeof(float));
                util::BitmapTexture::Save(image.get(), m_canvas_desc.w, m_canvas_desc.h, path.string(), util::BitmapTexture::FileFormat::EXR);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Console", NULL, show_window.console)) {
                show_window.console ^= true;
            }
            if (ImGui::MenuItem("Scene", NULL, show_window.scene)) {
                show_window.scene ^= true;
            }
            if (ImGui::MenuItem("Bottom", NULL, show_window.bottom)) {
                show_window.bottom ^= true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            ImGui::SeparatorText("Keyboard Op");
            ImGui::Text("[W] Move Forward");
            ImGui::Text("[S] Move Backward");
            ImGui::Text("[A] Move Left");
            ImGui::Text("[D] Move Right");
            ImGui::Text("[Q] Move Up");
            ImGui::Text("[E] Move Down");
            ImGui::SeparatorText("Mouse Op");
            ImGui::Text("Press the left mouse button and drag to change camera forward direction.");
            ImGui::Text("Rolling the mouse wheel to scale camera fov.");
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void GuiPass::Console(bool show) noexcept {
    if (!show) return;

    if (bool open = false;
        ImGui::Begin("Console", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {

        ImGui::PushTextWrapPos(0.f);

        if (ImGui::CollapsingHeader("Frame", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (auto &flag = util::Singleton<System>::instance()->render_flag;
                ImGui::Button(flag ? "Stop" : "Continue")) {
                if (flag ^= 1) {
                    EventDispatcher<ESystemEvent::StartRendering>();
                } else {
                    EventDispatcher<ESystemEvent::StopRendering>();
                }
            }
            ImGui::Text("Rendering average %.3lf ms/frame (%.1lf FPS)", 1000.0f / m_flip_rate, m_flip_rate);
            ImGui::Text("Frame size: %d x %d", m_canvas_desc.w, m_canvas_desc.h);
            ImGui::Text("Render output flip buffer index: %d", m_ready_buffer_index.load());
            if (bool flag = m_canvas_desc.tone_mapping;
                ImGui::Checkbox("Tone mapping", &flag)) {
                m_canvas_desc.tone_mapping = static_cast<uint32_t>(flag);
            }
            if (bool flag = m_canvas_desc.gamma_correct;
                ImGui::Checkbox("Gamma Correction", &flag)) {
                m_canvas_desc.gamma_correct = static_cast<uint32_t>(flag);
            }
            if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode("buffers")) {
                static int selected = -1;
                static int displayable_buffer_num = 0;
                auto buffer_mngr = util::Singleton<BufferManager>::instance();
                const auto &buffer_list = buffer_mngr->GetBufferNameList();
                if (selected == -1) {
                    displayable_buffer_num = 0;
                    for (auto &&buffer_name : buffer_list) {
                        auto buffer = buffer_mngr->GetBuffer(buffer_name);
                        if (buffer && buffer->desc.flag & EBufferFlag::AllowDisplay) {
                            if (buffer->desc.name == m_canvas_display_buffer_name) {
                                selected = displayable_buffer_num;
                            }
                            ++displayable_buffer_num;
                        }
                    }
                }

                ImGui::BeginChild("Console", ImVec2(0.f, ImGui::GetTextLineHeightWithSpacing() * min(displayable_buffer_num, 10)), false);
                for (int i = 0; auto &&buffer_name : buffer_list) {
                    auto buffer = buffer_mngr->GetBuffer(buffer_name);
                    if (buffer && buffer->desc.flag & EBufferFlag::AllowDisplay) {
                        if (ImGui::Selectable(buffer_name.data(), selected == i)) {
                            if (m_canvas_display_buffer_name != buffer_name) {
                                selected = i;
                                m_canvas_display_buffer_name = buffer_name;
                                if (!m_render_flag && !m_waiting_scene_load) {
                                    UpdateCanvasOutput();
                                    m_render_flip_buffer_to_texture_flag.exchange(true);
                                }
                            }
                        }
                        ++i;
                    }
                }
                ImGui::EndChild();

                ImGui::TreePop();
            }
        }

        if (ImGui::CollapsingHeader("Render Pass", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto sys = util::Singleton<System>::instance();
            for (auto &&pass : sys->m_pre_passes) {
                if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode(pass->name.c_str())) {
                    ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);
                    pass->Inspector();
                    ImGui::PopItemWidth();
                    ImGui::TreePop();
                }
            }
            for (auto &&pass : sys->m_passes) {
                if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode(pass->name.c_str())) {
                    ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);
                    pass->Inspector();
                    ImGui::PopItemWidth();
                    ImGui::TreePop();
                }
            }
        }

        if (m_inspectors.size() > 0) {
            if (ImGui::CollapsingHeader("Custom Control", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (auto &&[title, inspector] : m_inspectors) {
                    if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode(title.c_str())) {
                        ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);
                        inspector();
                        ImGui::PopItemWidth();
                        ImGui::TreePop();
                    }
                }
            }
        }

        ImGui::PopTextWrapPos();
    }
    ImGui::End();
}

void GuiPass::Canvas(bool show) noexcept {
    if (!show) return;

    if (bool open = true;
        ImGui::Begin("Canvas", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
        if (!m_waiting_scene_load) {
            if (auto buffer = GetReadyOutputBuffer();
                buffer.res) {
                float screen_w = ImGui::GetContentRegionAvail().x;
                float screen_h = ImGui::GetContentRegionAvail().y;
                float ratio_x = screen_w / m_canvas_desc.w;
                float ratio_y = screen_h / m_canvas_desc.h;
                float ratio = std::min(ratio_x, ratio_y);
                if (ratio == 0.f) ratio = 1.f;

                float show_w = m_canvas_desc.w * ratio;
                float show_h = m_canvas_desc.h * ratio;

                float cursor_x = (screen_w - show_w) * 0.5f + ImGui::GetCursorPosX();
                float cursor_y = (screen_h - show_h) * 0.5f + ImGui::GetCursorPosY();

                ImGui::SetCursorPos(ImVec2(cursor_x, cursor_y));
                ImGui::Image((ImTextureID)buffer.output_texture_srv.ptr,
                             ImVec2(show_w, show_h));

                bool disable_imguizmo = false;
                // This will catch our interactions
                if (!ImGuizmo::IsOver()) {
                    ImGui::SetCursorPos(ImVec2(cursor_x, cursor_y));
                    ImGui::InvisibleButton("canvas", ImVec2(show_w, show_h), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
                    ImGui::SetItemUsingMouseWheel();
                    const bool is_hovered = ImGui::IsItemHovered();// Hovered
                    const bool is_active = ImGui::IsItemActive();  // Held

                    if (is_hovered && is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                        ImGuiIO &io = ImGui::GetIO();
                        const struct {
                            float x, y;
                        } delta{ io.MouseDelta.x, io.MouseDelta.y };
                        EventDispatcher<ECanvasEvent::MouseDragging>(delta);
                        disable_imguizmo = true;
                    }

                    if (is_hovered) {
                        ImGuiIO &io = ImGui::GetIO();
                        if (io.MouseWheel != 0.f) {
                            EventDispatcher<ECanvasEvent::MouseWheel>(io.MouseWheel);
                            disable_imguizmo = true;
                        }

                        util::Float3 delta_pos;
                        if (ImGui::IsKeyDown(ImGuiKey_A)) delta_pos -= util::Camera::X;
                        if (ImGui::IsKeyDown(ImGuiKey_D)) delta_pos += util::Camera::X;
                        if (ImGui::IsKeyDown(ImGuiKey_W)) delta_pos -= util::Camera::Z;
                        if (ImGui::IsKeyDown(ImGuiKey_S)) delta_pos += util::Camera::Z;
                        if (ImGui::IsKeyDown(ImGuiKey_E)) delta_pos -= util::Camera::Y;
                        if (ImGui::IsKeyDown(ImGuiKey_Q)) delta_pos += util::Camera::Y;
                        if (delta_pos.x != 0.f || delta_pos.y != 0.f || delta_pos.z != 0.f) {
                            EventDispatcher<ECanvasEvent::CameraMove>(delta_pos);
                            disable_imguizmo = true;
                        }
                    }
                }

                if (auto world = util::Singleton<Pupil::world::World>::instance(); !disable_imguizmo && m_selected_ro && world->camera) {
                    ImGuizmo::SetDrawlist();
                    ImGuizmo::SetRect(ImGui::GetWindowPos().x + cursor_x, ImGui::GetWindowPos().y + cursor_y, show_w, show_h);

                    auto camera = world->camera.get();
                    auto proj = camera->GetProjectionMatrix().GetTranspose();
                    auto view = camera->GetViewMatrix().GetTranspose();
                    auto transform_matrix = m_selected_ro->transform.matrix.GetTranspose();
                    ImGuizmo::Manipulate(view.e, proj.e, m_zmo_operation, m_zmo_mode, transform_matrix.e, nullptr, nullptr);
                    if (auto new_transform = transform_matrix.GetTranspose();
                        !new_transform.ApproxEqualTo(m_selected_ro->transform.matrix, 1e-5)) {
                        m_selected_ro->UpdateTransform(new_transform);
                    }
                }
            }
        } else {
            ImGui::Text("Loading %c", "|/-\\"[(int)(ImGui::GetTime() / 0.05f) & 3]);
        }
    }
    ImGui::End();
}

void GuiPass::Scene(bool show) noexcept {
    if (!show) return;

    if (bool open = false;
        ImGui::Begin("Scene", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {

        ImGui::PushTextWrapPos(0.f);

        ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);

        auto world = util::Singleton<world::World>::instance();
        if (auto camera = world->camera.get(); camera) {
            if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode("Camera")) {
                auto desc = camera->GetDesc();

                float fov = desc.fov_y;
                ImGui::InputFloat("fov", &fov, 0.f, 0.f, "%.1f");
                if (fov != desc.fov_y) camera->SetFov(fov);

                float near_clip = desc.near_clip;
                float far_clip = desc.far_clip;
                // ImGui::InputFloat("near clip", &near_clip, 0.f, 0.f, "%.2f");
                near_clip = clamp(near_clip, 0.01f, desc.far_clip - 0.0001f);
                if (near_clip != desc.near_clip) camera->SetNearClip(near_clip);
                // ImGui::InputFloat("far clip", &far_clip, 0.f, 0.f, "%.2f");
                far_clip = clamp(far_clip, desc.near_clip + 0.0001f, 100000.f);
                if (far_clip != desc.far_clip) camera->SetFarClip(far_clip);

                if (fov != desc.fov_y || near_clip != desc.near_clip || far_clip != desc.far_clip)
                    EventDispatcher<EWorldEvent::CameraChange>();

                ImGui::InputFloat("sensitivity scale", &util::Camera::sensitivity_scale, 0.1f, 1.0f, "%.1f");

                auto transform = desc.to_world.matrix.GetTranspose();
                float translation[3], rotation[3], scale[3];
                ImGuizmo::DecomposeMatrixToComponents(transform.e, translation, rotation, scale);
                bool flag = false;
                ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.75f);
                flag |= ImGui::InputFloat3("Tr", translation);
                flag |= ImGui::InputFloat3("Rt", rotation);
                flag |= ImGui::InputFloat3("Sc", scale);
                ImGui::PopItemWidth();
                if (flag) {
                    ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, transform.e);
                    auto new_transform = transform.GetTranspose();
                    camera->SetWorldTransform(new_transform);
                    EventDispatcher<EWorldEvent::CameraChange>();
                }

                ImGui::TreePop();
            }
        }

        if (!m_waiting_scene_load && m_render_objects.size() > 0) {
            if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode("Render Objects")) {
                if (ImGui::Button("Unselect")) m_selected_ro = nullptr;
                ImGui::BeginChild("Scene", ImVec2(0.f, ImGui::GetTextLineHeightWithSpacing() * min((int)m_render_objects.size(), 10)), false);
                for (int selectable_index = 0; auto &&ro : m_render_objects) {
                    if (!ro) continue;
                    std::string ro_name = ro->name;
                    if (ro_name.empty()) ro_name = "(anonymous)" + std::to_string(selectable_index++);
                    if (ImGui::Selectable(ro_name.data(), m_selected_ro == ro))
                        m_selected_ro = ro;
                }
                ImGui::EndChild();

                if (m_selected_ro) {
                    ImGui::SeparatorText("Properties");

                    if (bool visibility = m_selected_ro->visibility_mask;
                        ImGui::Checkbox("visibility", &visibility)) {
                        m_selected_ro->visibility_mask = visibility;
                        EventDispatcher<EWorldEvent::RenderInstanceUpdate>(m_selected_ro);
                    }

                    ImGui::SeparatorText("Transform");
                    if (ImGui::RadioButton("Translate", m_zmo_operation == ImGuizmo::TRANSLATE))
                        m_zmo_operation = ImGuizmo::TRANSLATE;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Rotate", m_zmo_operation == ImGuizmo::ROTATE))
                        m_zmo_operation = ImGuizmo::ROTATE;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Scale", m_zmo_operation == ImGuizmo::SCALE))
                        m_zmo_operation = ImGuizmo::SCALE;

                    auto transform = m_selected_ro->transform.matrix.GetTranspose();
                    float translation[3], rotation[3], scale[3];
                    ImGuizmo::DecomposeMatrixToComponents(transform.e, translation, rotation, scale);
                    bool flag = false;
                    ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.75f);
                    flag |= ImGui::InputFloat3("Tr", translation);
                    flag |= ImGui::InputFloat3("Rt", rotation);
                    flag |= ImGui::InputFloat3("Sc", scale);
                    ImGui::PopItemWidth();
                    ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, transform.e);
                    if (flag) {
                        auto new_transform = transform.GetTranspose();
                        m_selected_ro->UpdateTransform(new_transform);
                    }

                    ImGui::SeparatorText("Material");
                }

                ImGui::TreePop();
            }
        }

        ImGui::PopItemWidth();

        ImGui::PopTextWrapPos();
    }
    ImGui::End();
}

void GuiPass::Bottom(bool show) noexcept {
    if (!show) return;

    if (bool open = false;
        ImGui::Begin("Bottom", &open)) {
        ImGui::Text("todo");
    }
    ImGui::End();
}

void GuiPass::OnDraw() noexcept {
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    Docking();

    ImGuizmo::SetOrthographic(false);

    // lock flip model to ensure the ready buffer is unchanged during the current frame
    std::scoped_lock lock{ m_flip_model_mutex };
    Menu();
    Console(show_window.console);
    Scene(show_window.scene);
    Bottom(show_window.bottom);

    // display scene loading browser
    m_scene_file_browser.Display();
    if (m_scene_file_browser.IsOpened()) {
        if (m_scene_file_browser.HasSelected()) {
            EventDispatcher<ESystemEvent::StopRendering>();
            m_waiting_scene_load = true;

            if (auto canvas_buffer = GetReadyOutputBuffer().system_buffer; canvas_buffer) {
                auto size = canvas_buffer->desc.width * canvas_buffer->desc.height * canvas_buffer->desc.stride_in_byte;
                cudaMemsetAsync(reinterpret_cast<void *>(canvas_buffer->cuda_ptr), 0, size, *m_memcpy_stream.get());
            }
            m_render_flip_buffer_to_texture_flag = true;

            util::Singleton<util::ThreadPool>::instance()->AddTask(
                [](std::filesystem::path path) {
                    util::Singleton<System>::instance()->SetScene(path);
                },
                m_scene_file_browser.GetSelected());
            m_scene_file_browser.ClearSelected();
        }
    }

    auto dx_ctx = util::Singleton<DirectX::Context>::instance();
    auto cmd_list = dx_ctx->GetCmdList();

    ID3D12DescriptorHeap *heaps[] = { dx_ctx->srv_heap.get() };
    cmd_list->SetDescriptorHeaps(1, heaps);
    // render buffer to texture
    if (auto buffer = GetReadyOutputBuffer();
        buffer.res && m_render_flip_buffer_to_texture_flag) {
        RenderFlipBufferToTexture(cmd_list);
        m_render_flip_buffer_to_texture_flag = false;
    }
    // show on canvas
    Canvas();

    ImGui::Render();
    dx_ctx->StartRenderScreen(cmd_list);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list.get());

    if (auto &io = ImGui::GetIO(); io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault(NULL, (void *)cmd_list.get());
    }

    m_memcpy_stream->Synchronize();
    dx_ctx->Present(cmd_list);
}

void GuiPass::InitRenderToTexturePipeline() noexcept {
    auto dx_ctx = util::Singleton<DirectX::Context>::instance();
    // root signature
    {
        D3D12_FEATURE_DATA_ROOT_SIGNATURE feat_data{};
        feat_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

        if (FAILED(dx_ctx->device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feat_data, sizeof(feat_data))))
            assert(false);

        D3D12_DESCRIPTOR_RANGE1 ranges[1]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;// TODO
        ranges[0].NumDescriptors = 1;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].RegisterSpace = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;

        CD3DX12_ROOT_PARAMETER1 root_params[2]{};
        root_params[0].InitAsDescriptorTable(1, ranges, D3D12_SHADER_VISIBILITY_PIXEL);
        root_params[1].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_PIXEL);

        D3D12_ROOT_SIGNATURE_FLAGS root_sign_flags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_sign_desc{};
        root_sign_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        root_sign_desc.Desc_1_1.Flags = root_sign_flags;
        root_sign_desc.Desc_1_1.NumParameters = 2;
        root_sign_desc.Desc_1_1.NumStaticSamplers = 0;
        root_sign_desc.Desc_1_1.pParameters = root_params;// TODO
        // root_sign_desc.Desc_1_1.pStaticSamplers = &sampler_desc;

        winrt::com_ptr<ID3DBlob> signature;
        winrt::com_ptr<ID3DBlob> error;
        DirectX::StopIfFailed(D3D12SerializeVersionedRootSignature(&root_sign_desc, signature.put(), error.put()));
        DirectX::StopIfFailed(dx_ctx->device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                                  winrt::guid_of<ID3D12RootSignature>(), m_root_signature.put_void()));
    }

    // pso
    {
        winrt::com_ptr<ID3DBlob> vs;
        winrt::com_ptr<ID3DBlob> ps;

#if defined(_DEBUG)
        // Enable better shader debugging with the graphics debugging tools.
        UINT compile_flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
        UINT compile_flags = 0;
#endif

        std::filesystem::path file_path = (std::filesystem::path{ ROOT_DIR } / "framework/system/gui/output.hlsl").make_preferred();
        std::wstring w_file_path = file_path.wstring();
        LPCWSTR result = w_file_path.data();
        //StopIfFailed(D3DCompileFromFile(result, 0, 0, "VSMain", "vs_5_1", compile_flags, 0, &vs, 0));
        //StopIfFailed(D3DCompileFromFile(result, 0, 0, "PSMain", "ps_5_1", compile_flags, 0, &ps, 0));
        winrt::com_ptr<ID3DBlob> errors;
        auto hr = D3DCompileFromFile(result, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "VSMain", "vs_5_1", compile_flags, 0, vs.put(), errors.put());
        if (errors != nullptr)
            OutputDebugStringA((char *)errors->GetBufferPointer());
        DirectX::StopIfFailed(hr);
        hr = D3DCompileFromFile(result, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "PSMain", "ps_5_1", compile_flags, 0, ps.put(), errors.put());
        if (errors != nullptr)
            OutputDebugStringA((char *)errors->GetBufferPointer());
        DirectX::StopIfFailed(hr);

        // Define the vertex input layout.
        D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
        };

        // Describe and create the graphics pipeline state object (PSO).
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = {};
        pso_desc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
        pso_desc.pRootSignature = m_root_signature.get();
        pso_desc.VS.BytecodeLength = vs->GetBufferSize();
        pso_desc.VS.pShaderBytecode = vs->GetBufferPointer();
        pso_desc.PS.BytecodeLength = ps->GetBufferSize();
        pso_desc.PS.pShaderBytecode = ps->GetBufferPointer();
        pso_desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        pso_desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        pso_desc.DepthStencilState.DepthEnable = FALSE;
        pso_desc.DepthStencilState.StencilEnable = FALSE;
        pso_desc.SampleMask = UINT_MAX;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        DirectX::StopIfFailed(dx_ctx->device->CreateGraphicsPipelineState(&pso_desc, winrt::guid_of<ID3D12PipelineState>(), m_pipeline_state.put_void()));
    }

    // upload vb
    auto cmd_list = dx_ctx->GetCmdList();
    winrt::com_ptr<ID3D12Resource> vb_upload;
    {
        struct TriVertex {
            float x, y, z;
            float u, v;
            TriVertex(float x, float y, float z, float u, float v) noexcept
                : x(x), y(y), z(z), u(u), v(v) {}
        };
        TriVertex quad[] = {
            { -1.f, -1.f, 0.f, 0.f, 0.f },
            { -1.f, 1.f, 0.f, 0.f, 1.f },
            { 1.f, -1.f, 0.f, 1.f, 0.f },
            { 1.f, 1.f, 0.f, 1.f, 1.f }
        };

        constexpr auto vb_size = sizeof(quad);
        D3D12_HEAP_PROPERTIES heap_properties{};
        heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_properties.CreationNodeMask = 1;
        heap_properties.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC vb_desc{};
        vb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        vb_desc.Alignment = 0;
        vb_desc.Width = vb_size;
        vb_desc.Height = 1;
        vb_desc.DepthOrArraySize = 1;
        vb_desc.MipLevels = 1;
        vb_desc.Format = DXGI_FORMAT_UNKNOWN;
        vb_desc.SampleDesc.Count = 1;
        vb_desc.SampleDesc.Quality = 0;
        vb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        vb_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &vb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            winrt::guid_of<ID3D12Resource>(), m_vb.put_void()));

        heap_properties.Type = D3D12_HEAP_TYPE_UPLOAD;
        DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &vb_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            winrt::guid_of<ID3D12Resource>(), vb_upload.put_void()));

        D3D12_SUBRESOURCE_DATA vertex_data{};
        vertex_data.pData = quad;
        vertex_data.RowPitch = vb_size;
        vertex_data.SlicePitch = vertex_data.RowPitch;

        UpdateSubresources<1>(cmd_list.get(), m_vb.get(), vb_upload.get(), 0, 0, 1, &vertex_data);
        D3D12_RESOURCE_BARRIER barrier =
            CD3DX12_RESOURCE_BARRIER::Transition(
                m_vb.get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

        cmd_list->ResourceBarrier(1, &barrier);

        // Initialize the vertex buffer view.
        m_vbv.BufferLocation = m_vb->GetGPUVirtualAddress();
        m_vbv.StrideInBytes = sizeof(TriVertex);
        m_vbv.SizeInBytes = vb_size;
    }

    dx_ctx->ExecuteCommandLists(cmd_list);
    dx_ctx->Flush();

    {
        CD3DX12_HEAP_PROPERTIES properties(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(CanvasDisplayDesc));
        DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
            &properties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            winrt::guid_of<ID3D12Resource>(),
            m_frame_constant_buffer.put_void()));

        m_frame_constant_buffer->Map(0, nullptr, &m_frame_constant_buffer_mapped_ptr);
    }
}
}// namespace Pupil

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
namespace {
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return ::DefWindowProc(hWnd, msg, wParam, lParam);

    POINT cursor_pos;
    switch (msg) {
        case WM_SIZE:
            if (auto dx_ctx = Pupil::util::Singleton<Pupil::DirectX::Context>::instance();
                dx_ctx->IsInitialized()) {
                if (wParam == SIZE_MINIMIZED) {
                    Pupil::EventDispatcher<Pupil::EWindowEvent::Minimized>();
                } else {
                    struct {
                        uint32_t w, h;
                    } size{ static_cast<uint32_t>(LOWORD(lParam)), static_cast<uint32_t>(HIWORD(lParam)) };
                    Pupil::EventDispatcher<Pupil::EWindowEvent::Resize>(size);
                }
            }
            return 0;
        case WM_EXITSIZEMOVE:
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU)// Disable ALT application menu
                return 0;
            break;
        case WM_DESTROY:
            Pupil::EventDispatcher<Pupil::EWindowEvent::Quit>();
            ::PostQuitMessage(0);
            return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
}// namespace