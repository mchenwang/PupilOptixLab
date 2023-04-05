#include "gui.h"
#include "system.h"
#include "resource.h"

#include "dx12/context.h"
#include "dx12/d3dx12.h"

#include "util/event.h"
#include "util/camera.h"
#include "util/thread_pool.h"
#include "scene/scene.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"
#include "imfilebrowser.h"

#include "static.h"

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

bool m_waiting_scene_load = false;
double m_flip_rate = 1.;

struct FrameInfo {
    uint32_t w = 1;
    uint32_t h = 1;
};
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
        });

        EventBinder<ESystemEvent::FrameFinished>([this](void *p) {
            double time_count = *(double *)p;
            m_flip_rate = 1000. / time_count;
        });

        EventBinder<ECanvasEvent::Resize>([this](void *p) {
            struct {
                uint32_t w, h;
            } size = *static_cast<decltype(size) *>(p);
            ResizeCanvas(size.w, size.h);
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

    m_init_flag = true;
}

void GuiPass::ResizeCanvas(uint32_t w, uint32_t h) noexcept {
    auto dx_ctx = util::Singleton<DirectX::Context>::instance();
    dx_ctx->Flush();
    // init render output buffers
    {
        auto buffer_mngr = util::Singleton<BufferManager>::instance();
        uint64_t size = static_cast<uint64_t>(h) * w * sizeof(float) * 4;
        m_output_h = h;
        m_output_w = w;

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

            // D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
            // rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            // rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            // rtv_desc.Texture2D.MipSlice = 1;
            // dx_ctx->device->CreateRenderTargetView(m_flip_buffers[i].res.get(), &rtv_desc, rtv_cpu_handle);
            dx_ctx->device->CreateRenderTargetView(m_flip_buffers[i].res.get(), nullptr, rtv_cpu_handle);

            srv_gpu_handle.ptr += srv_descriptor_handle_size;
            srv_cpu_handle.ptr += srv_descriptor_handle_size;
            m_flip_buffers[i].output_buffer_srv = srv_gpu_handle;

            BufferDesc buf_desc{
                .type = EBufferType::SharedCudaWithDX12,
                .name = std::string{ OUTPUT_FLIP_BUFFER[i] },
                .size = size
            };
            m_flip_buffers[i].shared_buffer = buffer_mngr->AllocBuffer(buf_desc)->shared_res;

            D3D12_SHADER_RESOURCE_VIEW_DESC buf_srv_desc{};
            buf_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            buf_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            buf_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
            buf_srv_desc.Buffer.NumElements = m_output_h * m_output_w;
            buf_srv_desc.Buffer.StructureByteStride = sizeof(float) * 4;
            buf_srv_desc.Buffer.FirstElement = 0;
            buf_srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            dx_ctx->device->CreateShaderResourceView(m_flip_buffers[i].shared_buffer.dx12_ptr.get(), &buf_srv_desc, srv_cpu_handle);
        }
    }

    {
        FrameInfo init_frame_info{
            .w = m_output_w,
            .h = m_output_h
        };
        memcpy(m_frame_constant_buffer_mapped_ptr, &init_frame_info, sizeof(FrameInfo));
    }
}

void GuiPass::Destroy() noexcept {
    if (!IsInitialized()) return;

    util::Singleton<DirectX::Context>::instance()->Flush();

    if (m_frame_constant_buffer_mapped_ptr) {
        m_frame_constant_buffer_mapped_ptr = nullptr;
        m_frame_constant_buffer->Unmap(0, nullptr);
    }

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    util::Singleton<DirectX::Context>::instance()->Destroy();
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
    m_inspectors.emplace(name, inspector);
}

void GuiPass::FlipSwapBuffer() noexcept {
    std::scoped_lock lock{ m_flip_model_mutex };
    m_ready_buffer_index = m_current_buffer_index;
    m_current_buffer_index = (m_current_buffer_index + 1) % SWAP_BUFFER_NUM;
    m_copy_after_flip_flag = true;
}

void GuiPass::Run() noexcept {
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
    cmd_list->SetGraphicsRootConstantBufferView(1, m_frame_constant_buffer->GetGPUVirtualAddress());

    D3D12_VIEWPORT viewport{ 0.f, 0.f, (FLOAT)m_output_w, (FLOAT)m_output_h, D3D12_MIN_DEPTH, D3D12_MAX_DEPTH };
    cmd_list->RSSetViewports(1, &viewport);
    D3D12_RECT rect{ 0, 0, (LONG)m_output_w, (LONG)m_output_h };
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

void GuiPass::OnDraw() noexcept {
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

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
        ImGuiID dock_inspector_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.3f, nullptr, &dock_main_id);

        ImGui::DockBuilderDockWindow("Inspector", dock_inspector_id);
        ImGui::DockBuilderDockWindow("Canvas", dock_main_id);

        ImGui::DockBuilderFinish(dock_main_id);
    }

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Menu")) {
            if (ImGui::MenuItem("load scene")) {
                m_scene_file_browser.Open();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (bool open = true;
        ImGui::Begin("Inspector", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {

        ImGui::PushTextWrapPos(0.f);

        if (ImGui::CollapsingHeader("Application", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("GUI average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Canvas Rendering:");
            ImGui::Text("Render output flip buffer index: %d", m_ready_buffer_index);
            ImGui::Text("Rendering average %.3lf ms/frame (%.1lf FPS)", 1000.0f / m_flip_rate, m_flip_rate);
            if (auto &flag = util::Singleton<System>::instance()->render_flag;
                ImGui::Button(flag ? "Stop" : "Continue")) {
                if (flag ^= 1) {
                    EventDispatcher<ESystemEvent::StartRendering>();
                } else {
                    EventDispatcher<ESystemEvent::StopRendering>();
                }
            }

            ImGui::Text("Camera:");
            ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);
            ImGui::InputFloat("sensitivity scale", &util::Camera::sensitivity_scale, 0.1f, 1.0f, "%.1f");
            ImGui::PopItemWidth();

            ImGui::Text("Save rendering screen shot:");
            // save image
            {
                ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.15f);
                static char file_name[256]{};
                ImGui::InputText("file name", file_name, 256);
                ImGui::SameLine();
                // constexpr auto image_file_format = std::array{ "hdr", "jpg", "png" };
                constexpr auto image_file_format = std::array{ "hdr" };
                static int item_current = 0;
                ImGui::Combo("format", &item_current, image_file_format.data(), (int)image_file_format.size());
                ImGui::SameLine();
                if (ImGui::Button("Save")) {
                    std::filesystem::path path{ ROOT_DIR };
                    path /= std::string{ file_name } + "." + image_file_format[item_current];

                    size_t size = g_window_h * g_window_w * 4;
                    // auto image = new float[size];
                    // memset(image, 0, size);
                    // cuda::CudaMemcpyToHost(image, m_backend->GetCurrentFrameResource().src->cuda_buffer_ptr, size * sizeof(float));

                    // stbi_flip_vertically_on_write(true);
                    // stbi_write_hdr(path.string().c_str(), g_window_w, g_window_h, 4, image);
                    // delete[] image;

                    // printf("image was saved successfully in [%ws].\n", path.wstring().data());
                }
                ImGui::PopItemWidth();
            }
        }

        for (auto &&[title, inspector] : m_inspectors) {
            if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                inspector();
            }
        }
        ImGui::PopTextWrapPos();
    }
    ImGui::End();

    auto dx_ctx = util::Singleton<DirectX::Context>::instance();
    auto cmd_list = dx_ctx->GetCmdList();

    ID3D12DescriptorHeap *heaps[] = { dx_ctx->srv_heap.get() };
    cmd_list->SetDescriptorHeaps(1, heaps);

    std::scoped_lock lock{ m_flip_model_mutex };
    if (bool open = true;
        ImGui::Begin("Canvas", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
        if (!m_waiting_scene_load) {
            if (auto buffer = GetReadyOutputBuffer();
                buffer.res) {
                if (m_copy_after_flip_flag) {
                    RenderFlipBufferToTexture(cmd_list);
                    m_copy_after_flip_flag = false;
                }
                float screen_w = ImGui::GetContentRegionAvail().x;
                float screen_h = ImGui::GetContentRegionAvail().y;
                float ratio_x = screen_w / m_output_w;
                float ratio_y = screen_h / m_output_h;
                float ratio = std::min(ratio_x, ratio_y);
                if (ratio == 0.f) ratio = 1.f;

                float show_w = m_output_w * ratio;
                float show_h = m_output_h * ratio;

                float cursor_x = (screen_w - show_w) * 0.5f + ImGui::GetCursorPosX();
                float cursor_y = (screen_h - show_h) * 0.5f + ImGui::GetCursorPosY();

                ImGui::SetCursorPos(ImVec2(cursor_x, cursor_y));
                ImGui::Image((ImTextureID)buffer.output_texture_srv.ptr,
                             ImVec2(show_w, show_h));

                // This will catch our interactions
                ImGui::SetCursorPos(ImVec2(cursor_x, cursor_y));
                ImGui::InvisibleButton("canvas", ImVec2(show_w, show_h), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
                ImGui::SetItemUsingMouseWheel();
                const bool is_hovered = ImGui::IsItemHovered();// Hovered
                const bool is_active = ImGui::IsItemActive();  // Held

                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    ImGuiIO &io = ImGui::GetIO();
                    const struct {
                        float x, y;
                    } delta{ io.MouseDelta.x, io.MouseDelta.y };
                    EventDispatcher<ECanvasEvent::MouseDragging>(delta);
                }

                if (is_hovered) {
                    ImGuiIO &io = ImGui::GetIO();
                    if (io.MouseWheel != 0.f)
                        EventDispatcher<ECanvasEvent::MouseWheel>(io.MouseWheel);

                    util::Float3 delta_pos;
                    if (ImGui::IsKeyDown(ImGuiKey_A)) delta_pos += util::Camera::X;
                    if (ImGui::IsKeyDown(ImGuiKey_D)) delta_pos -= util::Camera::X;
                    if (ImGui::IsKeyDown(ImGuiKey_W)) delta_pos += util::Camera::Z;
                    if (ImGui::IsKeyDown(ImGuiKey_S)) delta_pos -= util::Camera::Z;
                    if (ImGui::IsKeyDown(ImGuiKey_Q)) delta_pos += util::Camera::X;
                    if (ImGui::IsKeyDown(ImGuiKey_E)) delta_pos -= util::Camera::X;
                    if (delta_pos.x != 0.f || delta_pos.y != 0.f || delta_pos.z != 0.f)
                        EventDispatcher<ECanvasEvent::CameraMove>(delta_pos);
                }
            }
        } else {
            ImGui::Text("Loading %c", "|/-\\"[(int)(ImGui::GetTime() / 0.05f) & 3]);
        }
    }
    ImGui::End();

    m_scene_file_browser.Display();
    {
        if (m_scene_file_browser.HasSelected()) {
            EventDispatcher<ESystemEvent::StopRendering>();
            m_waiting_scene_load = true;
            util::Singleton<util::ThreadPool>::instance()->AddTask(
                [](std::filesystem::path path) {
                    util::Singleton<System>::instance()->SetScene(path);
                },
                m_scene_file_browser.GetSelected());
            m_scene_file_browser.ClearSelected();
        }
    }

    ImGui::Render();

    dx_ctx->StartRenderScreen(cmd_list);

    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list.get());

    auto &io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault(NULL, (void *)cmd_list.get());
    }

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

        std::filesystem::path file_path = (std::filesystem::path{ ROOT_DIR } / "framework/system/output.hlsl").make_preferred();
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
        CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(FrameInfo));
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
        // case WM_KEYDOWN:
        //     OnKeyDown(wParam);
        //     break;
        // case WM_KEYUP:
        //     OnKeyUp(wParam);
        //     break;
        // case WM_LBUTTONDOWN:
        // case WM_MBUTTONDOWN:
        // case WM_RBUTTONDOWN:
        //     if (GetCursorPos(&cursor_pos)) {
        //         OnMouseDown(wParam, cursor_pos.x, cursor_pos.y);
        //     }
        //     break;
        // case WM_LBUTTONUP:
        // case WM_MBUTTONUP:
        // case WM_RBUTTONUP:
        //     if (GetCursorPos(&cursor_pos)) {
        //         OnMouseUp(wParam, cursor_pos.x, cursor_pos.y);
        //     }
        //     break;
        // case WM_MOUSEMOVE:
        //     if (GetCursorPos(&cursor_pos)) {
        //         OnMouseMove(wParam, cursor_pos.x, cursor_pos.y);
        //         // m_last_mouse_pos.x = cursor_pos.x;
        //         // m_last_mouse_pos.y = cursor_pos.y;
        //     }
        //     break;
        // case WM_MOUSEWHEEL:
        //     OnMouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));
        //     break;
        case WM_DESTROY:
            Pupil::EventDispatcher<Pupil::EWindowEvent::Quit>();
            ::PostQuitMessage(0);
            return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
}// namespace