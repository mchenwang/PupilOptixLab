#include "pass.h"
#include "buffer_to_canvas.cuh"
#include "system/event.h"
#include "system/system.h"
#include "system/buffer.h"
#include "system/world.h"

#include "util/log.h"
#include "util/timer.h"

#include "scene/loader/util.h"
#include "resource/texture/image.h"
#include "cuda/check.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include "imfilebrowser.h"
#include "ImGuizmo/ImGuizmo.h"

#include "dx12/context.h"
#include "dx12/d3dx12.h"

#include "static.h"

#include <d3dcompiler.h>
#include <atomic>

namespace Pupil::Gui {
    LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
}

namespace Pupil::Gui {
    struct Pass::Impl {
        HWND     window_handle;
        uint32_t window_w = 1280;
        uint32_t window_h = 720;

        const std::wstring WND_NAME       = L"Pupil Optix Lab";
        const std::wstring WND_CLASS_NAME = L"PupilOptixLab_CLASS";
        HINSTANCE          instance;
        bool               init_flag     = false;
        bool               scene_loading = false;

        util::CountableRef<cuda::Stream> copy_stream = nullptr;

        ImGui::FileBrowser scene_file_browser;

        std::vector<std::pair<std::string, CustomConsole>> custom_consoles;

        bool show_console = true;
        bool show_scene   = true;
        bool show_bottom  = false;

        int    frame_cnt = 0;
        Timer  frame_rate_timer;
        double last_frame_time_cost = 0.;

        int selected_object_index = -1;

        // std::atomic_bool waiting_scene_load = false;
        // bool             render_flag        = false;
        // double           flip_rate          = 1.;

        // int              canvas_display_buffer_index = 0;

        // ImGuizmo::MODE      zmo_mode      = ImGuizmo::WORLD;
        // ImGuizmo::OPERATION zmo_operation = ImGuizmo::TRANSLATE;

        // render output buffer to texture
        winrt::com_ptr<ID3D12RootSignature> canvas_root_signature;
        winrt::com_ptr<ID3D12PipelineState> canvas_pipeline_state;
        winrt::com_ptr<ID3D12Resource>      canvas_vb;
        D3D12_VERTEX_BUFFER_VIEW            canvas_vbv;

        struct {
            uint32_t w             = 1;
            uint32_t h             = 1;
            uint32_t tone_mapping  = 0;
            uint32_t gamma_correct = 1;
        } canvas_desc;
        std::mutex                     canvas_mtx;
        winrt::com_ptr<ID3D12Resource> canvas_cb;
        void*                          canvas_cb_mapped_ptr = nullptr;
        std::string                    canvas_display_buffer_name{Pupil::BufferManager::DEFAULT_FINAL_RESULT_BUFFER_NAME};

        struct FlipModel {
            // index for displaying
            // index ^ 1 for rendering buffer to texture
            std::mutex       flip_mtx;
            int              index;
            std::atomic_bool ready_flip;

            std::mutex                     mtx[2];
            Buffer*                        system_buffer[2];
            winrt::com_ptr<ID3D12Resource> canvas_buffer[2];

            D3D12_GPU_DESCRIPTOR_HANDLE system_buffer_srv[2];
            D3D12_GPU_DESCRIPTOR_HANDLE canvas_srv[2];
            D3D12_CPU_DESCRIPTOR_HANDLE canvas_rtv[2];
        };

        FlipModel flip_model;

        constexpr static std::array<std::string_view, 2>
            OUTPUT_FLIP_BUFFER = {
                "output flip buffer0", "output flip buffer1"};

        void InitCanvasPipeline() noexcept;
        void ResizeCanvas(uint32_t w, uint32_t h) noexcept;
        bool CopyFromOutput(int index) noexcept;
        void RenderToCanvas(winrt::com_ptr<ID3D12GraphicsCommandList> cmd_list) noexcept;
        void OnDraw() noexcept;
        void Docking() noexcept;
        void Menu() noexcept;
        void Console() noexcept;
        void Canvas() noexcept;
        void Scene() noexcept;
        void Bottom() noexcept;
    };

    Pass::Pass() noexcept : Pupil::Pass("GUI") {
        m_impl = new Impl();
    }

    Pass::~Pass() noexcept {
        if (m_impl) delete m_impl;
        m_impl = nullptr;
    }

    void Pass::Init() noexcept {
        if (m_impl->init_flag) return;

        // create window
        {
            WNDCLASSEXW wc{};
            wc.cbSize        = sizeof(WNDCLASSEXW);
            wc.style         = CS_HREDRAW | CS_VREDRAW;
            wc.lpfnWndProc   = &Pupil::Gui::WndProc;
            wc.cbClsExtra    = 0;
            wc.cbWndExtra    = 0;
            wc.hInstance     = GetModuleHandleW(NULL);
            wc.hIcon         = NULL;
            wc.hCursor       = ::LoadCursorW(NULL, IDC_ARROW);
            wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
            wc.lpszMenuName  = NULL;
            wc.lpszClassName = m_impl->WND_CLASS_NAME.data();

            ::RegisterClassExW(&wc);

            RECT window_rect{0, 0, static_cast<LONG>(m_impl->window_w), static_cast<LONG>(m_impl->window_h)};
            ::AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, FALSE);

            int screen_w = ::GetSystemMetrics(SM_CXSCREEN);
            int screen_h = ::GetSystemMetrics(SM_CYSCREEN);

            int window_w = window_rect.right - window_rect.left;
            int window_h = window_rect.bottom - window_rect.top;

            // Center the window within the screen. Clamp to 0, 0 for the top-left corner.
            int window_x = std::max<int>(0, (screen_w - window_w) / 2);
            int window_y = std::max<int>(0, (screen_h - window_h) / 2);

            m_impl->instance      = GetModuleHandleW(NULL);
            m_impl->window_handle = ::CreateWindowExW(
                NULL,
                m_impl->WND_CLASS_NAME.data(),
                m_impl->WND_NAME.data(),
                WS_OVERLAPPEDWINDOW,
                window_x, window_y, window_w, window_h,
                NULL, NULL, wc.hInstance, NULL);

            ::ShowWindow(m_impl->window_handle, SW_SHOW);
            ::UpdateWindow(m_impl->window_handle);
        }

        // event
        {
            auto event_center = util::Singleton<Pupil::Event::Center>::instance();
            event_center->BindEvent(
                Pupil::Event::DispatcherMain, Gui::Event::WindowResized,
                new Pupil::Event::Handler2A<uint32_t, uint32_t>([this](uint32_t w, uint32_t h) {
                    if (!m_impl->init_flag) return;

                    if (w != m_impl->window_w || h != m_impl->window_h) {
                        m_impl->window_w = w;
                        m_impl->window_h = h;
                        util::Singleton<DirectX::Context>::instance()->Resize(w, h);
                    }
                }));

            event_center->BindEvent(
                Pupil::Event::DispatcherRender, Pupil::Event::FrameDone,
                new Pupil::Event::Handler1A<size_t>([this](size_t frame_cnt) {
                    int next_index;
                    {
                        std::scoped_lock lock(m_impl->flip_model.flip_mtx);
                        next_index = m_impl->flip_model.index ^ 1;
                        m_impl->flip_model.mtx[next_index].lock();
                    }

                    if (m_impl->CopyFromOutput(next_index)) {
                        m_impl->flip_model.ready_flip.store(true);
                        m_impl->frame_cnt = frame_cnt;
                    }

                    m_impl->flip_model.mtx[next_index].unlock();

                    m_impl->last_frame_time_cost = m_impl->frame_rate_timer.ElapsedMilliseconds();
                    m_impl->frame_rate_timer.Start();
                }));

            event_center->BindEvent(
                Pupil::Event::DispatcherRender, Pupil::Event::SceneReset,
                new Pupil::Event::Handler0A([this]() {
                    m_impl->frame_cnt = 0;
                    m_impl->frame_rate_timer.Start();
                }));

            event_center->BindEvent(
                Pupil::Event::DispatcherMain, Pupil::Event::SceneReset,
                new Pupil::Event::Handler0A([this]() {
                    m_impl->scene_loading = false;
                    auto buf_mngr         = util::Singleton<BufferManager>::instance();
                    auto output           = buf_mngr->GetBuffer(m_impl->canvas_display_buffer_name);
                    if (output) {
                        if (output->desc.height != m_impl->canvas_desc.h ||
                            output->desc.width != m_impl->canvas_desc.w) {
                            m_impl->ResizeCanvas(output->desc.width, output->desc.height);
                        }
                    }
                }));

            event_center->BindEvent(
                Pupil::Event::DispatcherMain, Pupil::Event::SceneLoading,
                new Pupil::Event::Handler0A([this]() {
                    m_impl->scene_loading = true;
                }));

            event_center->BindEvent(
                Pupil::Event::DispatcherMain, Pupil::Event::RenderContinue,
                new Pupil::Event::Handler0A([this]() {
                    m_impl->scene_loading = false;
                }));

            event_center->BindEvent(
                Pupil::Event::DispatcherMain, Pupil::Gui::Event::CanvasDisplayTargetChange,
                new Pupil::Event::Handler1A<std::string>([this](const std::string name) {
                    auto buf_mngr = util::Singleton<BufferManager>::instance();
                    auto output   = buf_mngr->GetBuffer(m_impl->canvas_display_buffer_name);
                    if (output) {
                        m_impl->canvas_display_buffer_name = name;

                        if (output->desc.height != m_impl->canvas_desc.h ||
                            output->desc.width != m_impl->canvas_desc.w) {
                            m_impl->ResizeCanvas(output->desc.width, output->desc.height);
                        }
                    }
                }));
        }

        // init imgui
        {
            auto dx_ctx = util::Singleton<DirectX::Context>::instance();
            dx_ctx->Init(m_impl->window_w, m_impl->window_h, m_impl->window_handle);

            ImGui::CreateContext();
            auto& io = ImGui::GetIO();
            (void)io;
            ImGui::StyleColorsDark();
            io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

            ImGui_ImplWin32_Init(m_impl->window_handle);

            ImGui_ImplDX12_Init(
                dx_ctx->device.get(),
                dx_ctx->FRAMES_NUM,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                dx_ctx->srv_heap.get(),
                dx_ctx->srv_heap->GetCPUDescriptorHandleForHeapStart(),
                dx_ctx->srv_heap->GetGPUDescriptorHandleForHeapStart());

            m_impl->scene_file_browser.SetTitle("scene browser");
            m_impl->scene_file_browser.SetTypeFilters(Pupil::util::g_supported_scene_format);

            std::filesystem::path scene_data_path{DATA_DIR};
            m_impl->scene_file_browser.SetPwd(scene_data_path);
        }

        m_impl->InitCanvasPipeline();

        m_impl->init_flag = true;
    }

    void Pass::OnRun() noexcept {
        if (!m_impl->init_flag) return;

        if (m_impl->copy_stream.Get() == nullptr) {
            m_impl->copy_stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::GUIInterop);
        }

        MSG msg = {};
        if (::PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
        }

        if (msg.message == WM_QUIT || msg.message == WM_DESTROY) {
            Pupil::util::Singleton<Pupil::Event::Center>::instance()->Send(Pupil::Event::RequestQuit);
        } else
            m_impl->OnDraw();
    }

    void Pass::Impl::OnDraw() noexcept {
        if (!init_flag) return;

        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        Docking();

        ImGuizmo::SetOrthographic(false);

        bool locked = false;
        if (flip_model.ready_flip.load()) {
            std::scoped_lock lock(flip_model.flip_mtx);
            flip_model.index ^= 1;
            flip_model.mtx[flip_model.index].lock();
            flip_model.ready_flip.store(false);
            locked = true;
        }

        Menu();
        Console();
        Scene();
        Bottom();

        // display scene loading browser
        scene_file_browser.Display();
        if (scene_file_browser.IsOpened()) {
            if (scene_file_browser.HasSelected()) {
                util::Singleton<Pupil::Event::Center>::instance()->Send(Pupil::Event::RequestSceneLoad, {scene_file_browser.GetSelected().string()});

                scene_file_browser.ClearSelected();
            }
        }

        auto dx_ctx   = util::Singleton<DirectX::Context>::instance();
        auto cmd_list = dx_ctx->GetCmdList();

        ID3D12DescriptorHeap* heaps[] = {dx_ctx->srv_heap.get()};
        cmd_list->SetDescriptorHeaps(1, heaps);
        // render buffer to texture
        if (frame_cnt > 0 && !scene_loading)
            RenderToCanvas(cmd_list);
        // show on canvas
        Canvas();

        ImGui::Render();
        dx_ctx->StartRenderScreen(cmd_list);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list.get());

        if (auto& io = ImGui::GetIO(); io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault(NULL, (void*)cmd_list.get());
        }

        copy_stream->Synchronize();
        dx_ctx->Present(cmd_list);
        if (locked)
            flip_model.mtx[flip_model.index].unlock();
    }

    void Pass::Impl::ResizeCanvas(uint32_t w, uint32_t h) noexcept {
        if (w == canvas_desc.w && h == canvas_desc.h) return;
        canvas_mtx.lock();

        canvas_desc.h = h;
        canvas_desc.w = w;

        auto dx_ctx = util::Singleton<DirectX::Context>::instance();
        dx_ctx->Flush();
        copy_stream->Synchronize();
        // init render output buffers
        {
            auto     buffer_mngr = util::Singleton<BufferManager>::instance();
            uint64_t size        = static_cast<uint64_t>(h) * w * sizeof(float) * 4;

            auto srv_descriptor_handle_size = dx_ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            auto rtv_descriptor_handle_size = dx_ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            auto srv_cpu_handle             = dx_ctx->srv_heap->GetCPUDescriptorHandleForHeapStart();
            auto srv_gpu_handle             = dx_ctx->srv_heap->GetGPUDescriptorHandleForHeapStart();
            auto rtv_cpu_handle             = dx_ctx->rtv_heap->GetCPUDescriptorHandleForHeapStart();

            rtv_cpu_handle.ptr += rtv_descriptor_handle_size * 2;

            for (auto i = 0u; i < 2; ++i) {
                auto d3d12_res_desc = CD3DX12_RESOURCE_DESC::Tex2D(
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    static_cast<UINT>(w), static_cast<UINT>(h),
                    1, 0, 1, 0,
                    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
                auto                           properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
                winrt::com_ptr<ID3D12Resource> temp_res;
                DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
                    &properties, D3D12_HEAP_FLAG_NONE, &d3d12_res_desc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    winrt::guid_of<ID3D12Resource>(), temp_res.put_void()));

                flip_model.canvas_buffer[i] = temp_res;
                flip_model.canvas_buffer[i]->SetName((L"canvas buffer " + std::to_wstring(i)).c_str());

                srv_gpu_handle.ptr += srv_descriptor_handle_size;
                srv_cpu_handle.ptr += srv_descriptor_handle_size;
                flip_model.canvas_srv[i] = srv_gpu_handle;

                D3D12_SHADER_RESOURCE_VIEW_DESC tex_srv_desc{};
                tex_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                tex_srv_desc.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
                tex_srv_desc.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
                tex_srv_desc.Texture2D.MipLevels     = 1;
                dx_ctx->device->CreateShaderResourceView(flip_model.canvas_buffer[i].get(), &tex_srv_desc, srv_cpu_handle);

                rtv_cpu_handle.ptr += rtv_descriptor_handle_size;
                flip_model.canvas_rtv[i] = rtv_cpu_handle;

                dx_ctx->device->CreateRenderTargetView(flip_model.canvas_buffer[i].get(), nullptr, rtv_cpu_handle);

                srv_gpu_handle.ptr += srv_descriptor_handle_size;
                srv_cpu_handle.ptr += srv_descriptor_handle_size;
                flip_model.system_buffer_srv[i] = srv_gpu_handle;

                BufferDesc buf_desc{
                    .name           = OUTPUT_FLIP_BUFFER[i].data(),
                    .flag           = EBufferFlag::SharedWithDX12,
                    .width          = w,
                    .height         = h,
                    .stride_in_byte = sizeof(float) * 4};
                flip_model.system_buffer[i] = buffer_mngr->AllocBuffer(buf_desc);

                D3D12_SHADER_RESOURCE_VIEW_DESC buf_srv_desc{};
                buf_srv_desc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                buf_srv_desc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                buf_srv_desc.Format                     = DXGI_FORMAT_UNKNOWN;
                buf_srv_desc.Buffer.NumElements         = static_cast<UINT>(canvas_desc.h * canvas_desc.w);
                buf_srv_desc.Buffer.StructureByteStride = sizeof(float) * 4;
                buf_srv_desc.Buffer.FirstElement        = 0;
                buf_srv_desc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                dx_ctx->device->CreateShaderResourceView(flip_model.system_buffer[i]->dx12_ptr.get(), &buf_srv_desc, srv_cpu_handle);
            }
        }

        Pupil::Log::Info("Canvas resize to {}x{}", canvas_desc.w, canvas_desc.h);
        canvas_mtx.unlock();
    }

    bool Pass::Impl::CopyFromOutput(int dst_index) noexcept {
        if (!canvas_mtx.try_lock()) return false;
        auto buf_mngr = util::Singleton<BufferManager>::instance();
        auto output   = buf_mngr->GetBuffer(canvas_display_buffer_name);

        auto dst_buf = flip_model.system_buffer[dst_index];
        if (dst_buf) {
            if (output->desc.stride_in_byte == sizeof(float4)) {
                cudaMemcpyAsync(reinterpret_cast<void*>(dst_buf->cuda_ptr),
                                reinterpret_cast<void*>(output->cuda_ptr),
                                static_cast<size_t>(dst_buf->desc.width * dst_buf->desc.height * dst_buf->desc.stride_in_byte),
                                cudaMemcpyKind::cudaMemcpyDeviceToDevice, *copy_stream.Get());
            } else if (output->desc.stride_in_byte == sizeof(float3)) {
                Pupil::CopyFromFloat3(dst_buf->cuda_ptr, output->cuda_ptr,
                                      dst_buf->desc.width * dst_buf->desc.height, copy_stream.Get());
            } else if (output->desc.stride_in_byte == sizeof(float2)) {
                Pupil::CopyFromFloat2(dst_buf->cuda_ptr, output->cuda_ptr,
                                      dst_buf->desc.width * dst_buf->desc.height, copy_stream.Get());
            } else if (output->desc.stride_in_byte == sizeof(float)) {
                Pupil::CopyFromFloat1(dst_buf->cuda_ptr, output->cuda_ptr,
                                      dst_buf->desc.width * dst_buf->desc.height, copy_stream.Get());
            }
        }
        canvas_mtx.unlock();
        return true;
    }

    void Pass::Impl::Docking() noexcept {
        static bool first_draw_call = true;
        ImGuiID     main_node_id    = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        if (first_draw_call) {
            first_draw_call = false;

            ImGui::DockBuilderRemoveNode(main_node_id);
            ImGui::DockBuilderAddNode(main_node_id, ImGuiDockNodeFlags_None);

            // Make the dock node's size and position to match the viewport
            ImGui::DockBuilderSetNodeSize(main_node_id, ImGui::GetMainViewport()->WorkSize);
            ImGui::DockBuilderSetNodePos(main_node_id, ImGui::GetMainViewport()->WorkPos);

            ImGuiID dock_main_id   = main_node_id;
            ImGuiID dock_left_id   = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.2f, nullptr, &dock_main_id);
            ImGuiID dock_bottom_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.3f, nullptr, &dock_main_id);
            ImGuiID dock_right_id  = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.26f, nullptr, &dock_main_id);

            ImGui::DockBuilderDockWindow("Console", dock_left_id);
            ImGui::DockBuilderDockWindow("Scene", dock_right_id);
            ImGui::DockBuilderDockWindow("Canvas", dock_main_id);
            ImGui::DockBuilderDockWindow("Bottom", dock_bottom_id);

            ImGui::DockBuilderFinish(dock_main_id);
        }
    }

    void Pass::Impl::Menu() noexcept {
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("Tool")) {
                if (ImGui::MenuItem("Load Scene")) {
                    scene_file_browser.Open();
                }

                if (ImGui::MenuItem("Screenshot")) {
                    std::filesystem::path path{ROOT_DIR};

                    auto time = std::chrono::zoned_time{
                        std::chrono::current_zone(),
                        std::chrono::system_clock::now()};

                    auto time_str = std::format(
                        "{:%H.%M.%S.%m.%d}",
                        floor<std::chrono::seconds>(time.get_local_time()));
                    path /= "screenshot(" + time_str + ").exr";

                    size_t size = canvas_desc.h * canvas_desc.w * 4;

                    resource::Image image{
                        .w    = canvas_desc.w,
                        .h    = canvas_desc.h,
                        .data = new float[size],
                    };
                    memset(image.data, 0, size * sizeof(float));

                    auto buffer = flip_model.system_buffer[flip_model.index];

                    auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::GUIInterop).Get();
                    CUDA_CHECK(cudaMemcpyAsync(image.data,
                                               reinterpret_cast<const void*>(buffer->cuda_ptr),
                                               size * sizeof(float), cudaMemcpyDeviceToHost, *stream));
                    stream->Synchronize();
                    resource::Image::Save(image, path.string().data());
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Setting")) {
                if (ImGui::BeginMenu("Limit Frame Rate")) {
                    static bool enable         = false;
                    static int  max_frame_rate = 60;
                    ImGui::PushItemWidth(100);
                    if (ImGui::MenuItem("Enable", NULL, &enable)) {
                        util::Singleton<Pupil::Event::Center>::instance()
                            ->Send(Pupil::Event::LimitRenderRate, {enable ? max_frame_rate : -1});
                    }
                    if (ImGui::SliderInt("max frame rate", &max_frame_rate, 10, 240)) {
                        if (enable) {
                            util::Singleton<Pupil::Event::Center>::instance()
                                ->Send(Pupil::Event::LimitRenderRate, {max_frame_rate});
                        }
                    }
                    ImGui::PopItemWidth();
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Window")) {
                if (ImGui::MenuItem("Console", NULL, show_console)) {
                    show_console ^= true;
                }
                if (ImGui::MenuItem("Scene", NULL, show_scene)) {
                    show_scene ^= true;
                }
                if (ImGui::MenuItem("Bottom", NULL, show_bottom)) {
                    show_bottom ^= true;
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

    void Pass::RegisterConsole(std::string_view name, CustomConsole&& console) noexcept {
        m_impl->custom_consoles.emplace_back(name, console);
    }

    void Pass::Impl::Console() noexcept {
        if (!show_console) return;

        if (bool open = false;
            ImGui::Begin("Console", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {

            ImGui::PushTextWrapPos(0.f);

            if (ImGui::CollapsingHeader("Frame", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Rendering average %.3lf ms/frame (%d FPS)", last_frame_time_cost, (int)(1000.0f / last_frame_time_cost));
                ImGui::Text("Frame size: %d x %d", canvas_desc.w, canvas_desc.h);
                ImGui::Text("sample count: %d", frame_cnt);
                ImGui::Text("Render output flip buffer index: %d", flip_model.index);
                if (bool flag = canvas_desc.tone_mapping;
                    ImGui::Checkbox("Tone mapping", &flag)) {
                    canvas_desc.tone_mapping = static_cast<uint32_t>(flag);
                }
                if (bool flag = canvas_desc.gamma_correct;
                    ImGui::Checkbox("Gamma Correction", &flag)) {
                    canvas_desc.gamma_correct = static_cast<uint32_t>(flag);
                }
                if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode("buffers")) {
                    static int  selected               = -1;
                    static int  displayable_buffer_num = 0;
                    auto        buffer_mngr            = util::Singleton<BufferManager>::instance();
                    const auto& buffer_list            = buffer_mngr->GetBufferNameList();
                    if (selected == -1) {
                        displayable_buffer_num = 0;
                        for (auto&& buffer_name : buffer_list) {
                            auto buffer = buffer_mngr->GetBuffer(buffer_name);
                            if (buffer && buffer->desc.flag & EBufferFlag::AllowDisplay) {
                                if (buffer->desc.name == canvas_display_buffer_name) {
                                    selected = displayable_buffer_num;
                                }
                                ++displayable_buffer_num;
                            }
                        }
                    }

                    if (buffer_list.size() > 0) {
                        ImGui::BeginChild("Console", ImVec2(0.f, ImGui::GetTextLineHeightWithSpacing() * min(displayable_buffer_num, 10)), false);
                        for (int i = 0; auto&& buffer_name : buffer_list) {
                            auto buffer = buffer_mngr->GetBuffer(buffer_name);
                            if (buffer && buffer->desc.flag & EBufferFlag::AllowDisplay) {
                                if (ImGui::Selectable(buffer_name.data(), selected == i)) {
                                    if (canvas_display_buffer_name != buffer_name) {
                                        selected                   = i;
                                        canvas_display_buffer_name = buffer_name;
                                    }
                                }
                                ++i;
                            }
                        }
                        ImGui::EndChild();
                    }

                    ImGui::TreePop();
                }
            }

            if (custom_consoles.size() > 0) {
                if (ImGui::CollapsingHeader("Custom Control", ImGuiTreeNodeFlags_DefaultOpen)) {
                    for (auto&& [title, console] : custom_consoles) {
                        if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode(title.c_str())) {
                            ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);
                            console();
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

    void Pass::Impl::Canvas() noexcept {
        if (bool open = true;
            ImGui::Begin("Canvas", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
            if (!scene_loading) {
                if (auto buffer = flip_model.canvas_buffer[flip_model.index];
                    buffer) {
                    float screen_w = ImGui::GetContentRegionAvail().x;
                    float screen_h = ImGui::GetContentRegionAvail().y;
                    float ratio_x  = screen_w / canvas_desc.w;
                    float ratio_y  = screen_h / canvas_desc.h;
                    float ratio    = std::min(ratio_x, ratio_y);
                    if (ratio == 0.f) ratio = 1.f;

                    float show_w = canvas_desc.w * ratio;
                    float show_h = canvas_desc.h * ratio;

                    float cursor_x = (screen_w - show_w) * 0.5f + ImGui::GetCursorPosX();
                    float cursor_y = (screen_h - show_h) * 0.5f + ImGui::GetCursorPosY();

                    ImGui::SetCursorPos(ImVec2(cursor_x, cursor_y));
                    ImGui::Image((ImTextureID)flip_model.canvas_srv[flip_model.index].ptr,
                                 ImVec2(show_w, show_h));

                    bool disable_imguizmo = false;
                    // This will catch our interactions
                    if (!ImGuizmo::IsOver()) {
                        ImGui::SetCursorPos(ImVec2(cursor_x, cursor_y));
                        ImGui::InvisibleButton("canvas", ImVec2(show_w, show_h), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
                        ImGui::SetItemUsingMouseWheel();
                        const bool is_hovered = ImGui::IsItemHovered();// Hovered
                        const bool is_active  = ImGui::IsItemActive(); // Held

                        if (is_hovered && is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                            ImGuiIO& io = ImGui::GetIO();
                            util::Singleton<World>::instance()->CameraRotate(io.MouseDelta.x, io.MouseDelta.y);
                            disable_imguizmo = true;
                        }

                        if (is_hovered) {
                            ImGuiIO& io = ImGui::GetIO();
                            if (io.MouseWheel != 0.f) {
                                util::Singleton<World>::instance()->SetCameraFovDelta(io.MouseWheel);
                                disable_imguizmo = true;
                            }

                            Float3 delta_pos;
                            if (ImGui::IsKeyDown(ImGuiKey_A)) delta_pos -= Pupil::Camera::X;
                            if (ImGui::IsKeyDown(ImGuiKey_D)) delta_pos += Pupil::Camera::X;
                            if (ImGui::IsKeyDown(ImGuiKey_W)) delta_pos -= Pupil::Camera::Z;
                            if (ImGui::IsKeyDown(ImGuiKey_S)) delta_pos += Pupil::Camera::Z;
                            if (ImGui::IsKeyDown(ImGuiKey_E)) delta_pos -= Pupil::Camera::Y;
                            if (ImGui::IsKeyDown(ImGuiKey_Q)) delta_pos += Pupil::Camera::Y;
                            if (delta_pos.x != 0.f || delta_pos.y != 0.f || delta_pos.z != 0.f) {
                                util::Singleton<World>::instance()->CameraMove(delta_pos);
                                disable_imguizmo = true;
                            }
                        }
                    }

                    // if (auto world = util::Singleton<Pupil::World>::instance();
                    //     !disable_imguizmo && m_selected_ro && world->camera) {
                    //     ImGuizmo::SetDrawlist();
                    //     ImGuizmo::SetRect(ImGui::GetWindowPos().x + cursor_x, ImGui::GetWindowPos().y + cursor_y, show_w, show_h);

                    //     auto camera           = world->camera.get();
                    //     auto proj             = camera->GetProjectionMatrix().GetTranspose();
                    //     auto view             = camera->GetViewMatrix().GetTranspose();
                    //     auto transform_matrix = m_selected_ro->transform.matrix.GetTranspose();
                    //     ImGuizmo::Manipulate(view.e, proj.e, m_zmo_operation, m_zmo_mode, transform_matrix.e, nullptr, nullptr);
                    //     if (auto new_transform = transform_matrix.GetTranspose();
                    //         !new_transform.ApproxEqualTo(m_selected_ro->transform.matrix, 1e-5)) {
                    //         m_selected_ro->UpdateTransform(new_transform);
                    //     }
                    // }
                }
            } else {
                ImGui::Text("Loading %c", "|/-\\"[(int)(ImGui::GetTime() / 0.05f) & 3]);
            }
        }
        ImGui::End();
    }

    void Pass::Impl::Scene() noexcept {
        if (!show_scene) return;

        if (bool open = false;
            ImGui::Begin("Scene", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {

            ImGui::PushTextWrapPos(0.f);

            ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);

            auto world = util::Singleton<World>::instance();
            if (auto camera = world->GetScene()->GetCamera(); camera) {
                if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode("Camera")) {

                    if (float fov = camera->GetFovY().GetDegree();
                        ImGui::InputFloat("fov y", &fov, 0.f, 0.f, "%.1f")) {
                        world->SetCameraFov(fov);
                    }

                    if (float near_clip = camera->GetNearClip();
                        ImGui::InputFloat("near clip", &near_clip, 0.f, 0.f, "%.2f")) {
                        near_clip = clamp(near_clip, 0.01f, camera->GetFarClip() - 0.01f);
                        world->SetCameraNearClip(near_clip);
                    }

                    if (float far_clip = camera->GetFarClip();
                        ImGui::InputFloat("far clip", &far_clip, 0.f, 0.f, "%.2f")) {
                        far_clip = clamp(far_clip, camera->GetNearClip() + 0.01f, 10000.f);
                        world->SetCameraFarClip(far_clip);
                    }

                    auto to_world                           = Pupil::Transform(camera->GetToWorldMatrix());
                    auto [translation, scaling, quaternion] = to_world.AffineDecomposition();
                    bool flag                               = false;
                    ImGui::Text("position:");
                    {
                        static float trans_x = translation.x;
                        static float trans_y = translation.y;
                        static float trans_z = translation.z;
                        static bool  changed = false;
                        ImGui::InputFloat("x##position", &trans_x, 0.f, 0.f, "%.2f");
                        bool xb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool xb2 = ImGui::IsItemActive();
                        ImGui::InputFloat("y##position", &trans_y, 0.f, 0.f, "%.2f");
                        bool yb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool yb2 = ImGui::IsItemActive();
                        ImGui::InputFloat("z##position", &trans_z, 0.f, 0.f, "%.2f");
                        bool zb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool zb2 = ImGui::IsItemActive();

                        if ((xb1 && !yb2 && !zb2) || (!xb2 && yb1 && !zb2) || (!xb2 && !yb2 && zb1) ||
                            (changed && !xb2 && !yb2 && !zb2)) {
                            flag        = true;
                            changed     = false;
                            translation = Vector3f(trans_x, trans_y, trans_z);
                        } else {
                            changed |= xb1 || yb1 || zb1;
                        }
                    }

                    bool        rotate_flag         = false;
                    const char* rotate_modes[]      = {"Quaternion", "Axis angle"};
                    static int  current_rotate_mode = 0;
                    ImGui::Combo("Rotate mode", &current_rotate_mode, rotate_modes, 2);
                    if (current_rotate_mode == 0) {
                        ImGui::Text("quaternion:");
                        static float quat_w = quaternion.vec.w;
                        static float quat_x = quaternion.vec.x;
                        static float quat_y = quaternion.vec.y;
                        static float quat_z = quaternion.vec.z;
                        ImGui::InputFloat("w##quaternion", &quat_w, 0.f, 0.f, "%.2f");
                        bool wb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool wb2 = ImGui::IsItemActivated();
                        ImGui::InputFloat("x##quaternion", &quat_x, 0.f, 0.f, "%.2f");
                        bool xb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool xb2 = ImGui::IsItemActivated();
                        ImGui::InputFloat("y##quaternion", &quat_y, 0.f, 0.f, "%.2f");
                        bool yb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool yb2 = ImGui::IsItemActivated();
                        ImGui::InputFloat("z##quaternion", &quat_z, 0.f, 0.f, "%.2f");
                        bool zb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool zb2 = ImGui::IsItemActivated();

                        if ((xb1 && !yb2 && !zb2 && !wb2) || (!xb2 && yb1 && !zb2 && !wb2) ||
                            (!xb2 && !yb2 && zb1 && !wb2) || (!xb2 && !yb2 && !zb2 && wb1)) {
                            rotate_flag = true;
                            quaternion  = Quaternion(quat_w, quat_x, quat_y, quat_z);
                        }

                    } else if (current_rotate_mode == 1) {
                        ImGui::Text("axis angle:");
                        static auto [axis, angle] = quaternion.GetAxisAngle();
                        static float axis_x       = axis.x;
                        static float axis_y       = axis.y;
                        static float axis_z       = axis.z;
                        static float angle_rad    = angle.radian;

                        ImGui::InputFloat("axis x", &axis_x, 0.f, 0.f, "%.2f");
                        bool xb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool xb2 = ImGui::IsItemActivated();
                        ImGui::InputFloat("axis y", &axis_y, 0.f, 0.f, "%.2f");
                        bool yb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool yb2 = ImGui::IsItemActivated();
                        ImGui::InputFloat("axis z", &axis_z, 0.f, 0.f, "%.2f");
                        bool zb1 = ImGui::IsItemDeactivatedAfterEdit();
                        bool zb2 = ImGui::IsItemActivated();
                        ImGui::SliderAngle("angle", &angle_rad);
                        bool angle_changed = ImGui::IsItemEdited();

                        if ((xb1 && !yb2 && !zb2) || (!xb2 && yb1 && !zb2) || (!xb2 && !yb2 && zb1) ||
                            angle_changed) {
                            rotate_flag = true;
                            quaternion  = Quaternion(Vector3f(axis_x, axis_y, axis_z), Angle(angle_rad));
                        }
                    }
                    flag |= rotate_flag;

                    if (flag)
                        world->SetCameraWorldTransform(Transform(translation, scaling, quaternion));

                    ImGui::TreePop();
                }
            }

            // if (auto& instances = world->GetScene()->GetInstances();
            //     !scene_loading && instances.size() > 0) {
            //     if (ImGui::SetNextItemOpen(true, ImGuiCond_Once); ImGui::TreeNode("Render Objects")) {
            //         if (ImGui::Button("Unselect")) m_selected_ro = nullptr;
            //         ImGui::BeginChild("Scene", ImVec2(0.f, ImGui::GetTextLineHeightWithSpacing() * min((int)m_render_objects.size(), 10)), false);
            //         for (int selectable_index = 0; auto&& ro : m_render_objects) {
            //             if (!ro) continue;
            //             std::string ro_name = ro->name;
            //             if (ro_name.empty()) ro_name = "(anonymous)" + std::to_string(selectable_index++);
            //             if (ImGui::Selectable(ro_name.data(), m_selected_ro == ro))
            //                 m_selected_ro = ro;
            //         }
            //         ImGui::EndChild();

            //         ImGui::TreePop();
            //     }
            // }

            ImGui::PopItemWidth();

            ImGui::PopTextWrapPos();
        }
        ImGui::End();
    }

    void Pass::Impl::Bottom() noexcept {
        if (!show_bottom) return;

        if (bool open = false;
            ImGui::Begin("Bottom", &open)) {
            ImGui::Text("todo");
        }
        ImGui::End();
    }

    void Pass::Impl::RenderToCanvas(winrt::com_ptr<ID3D12GraphicsCommandList> cmd_list) noexcept {
        auto dx_ctx = util::Singleton<DirectX::Context>::instance();

        cmd_list->SetGraphicsRootSignature(canvas_root_signature.get());
        cmd_list->SetPipelineState(canvas_pipeline_state.get());

        // auto& buffer = GetReadyOutputBuffer();
        // ID3D12DescriptorHeap *heaps[] = { dx_ctx->srv_heap.get() };
        // cmd_list->SetDescriptorHeaps(1, heaps);
        cmd_list->SetGraphicsRootDescriptorTable(0, flip_model.system_buffer_srv[flip_model.index]);
        memcpy(canvas_cb_mapped_ptr, &canvas_desc, sizeof(canvas_desc));
        cmd_list->SetGraphicsRootConstantBufferView(1, canvas_cb->GetGPUVirtualAddress());

        D3D12_VIEWPORT viewport{0.f, 0.f, (FLOAT)canvas_desc.w, (FLOAT)canvas_desc.h, D3D12_MIN_DEPTH, D3D12_MAX_DEPTH};
        cmd_list->RSSetViewports(1, &viewport);
        D3D12_RECT rect{0, 0, (LONG)canvas_desc.w, (LONG)canvas_desc.h};
        cmd_list->RSSetScissorRects(1, &rect);

        auto rt  = flip_model.canvas_buffer[flip_model.index];
        auto rtv = flip_model.canvas_rtv[flip_model.index];

        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource   = rt.get();
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd_list->ResourceBarrier(1, &barrier);
        }

        cmd_list->OMSetRenderTargets(1, &rtv, TRUE, nullptr);
        const FLOAT clear_color[4]{0.f, 0.f, 0.f, 1.f};
        cmd_list->ClearRenderTargetView(rtv, clear_color, 0, nullptr);

        cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        cmd_list->IASetVertexBuffers(0, 1, &canvas_vbv);
        cmd_list->DrawInstanced(4, 1, 0, 0);

        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource   = rt.get();
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd_list->ResourceBarrier(1, &barrier);
        }
    }

    void Pass::Destroy() noexcept {
        if (!m_impl->init_flag) return;

        m_impl->copy_stream->Synchronize();
        util::Singleton<DirectX::Context>::instance()->Flush();

        if (m_impl->canvas_cb_mapped_ptr) {
            m_impl->canvas_cb_mapped_ptr = nullptr;
            m_impl->canvas_cb->Unmap(0, nullptr);
        }

        ImGui_ImplDX12_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();

        // for (auto&& buffer : m_flip_buffers) buffer.res = nullptr;
        m_impl->canvas_root_signature = nullptr;
        m_impl->canvas_pipeline_state = nullptr;
        m_impl->canvas_vb             = nullptr;
        m_impl->canvas_cb             = nullptr;

        ::DestroyWindow(m_impl->window_handle);
        ::UnregisterClassW(m_impl->WND_CLASS_NAME.data(), m_impl->instance);
        m_impl->init_flag = false;
    }

    void Pass::Impl::InitCanvasPipeline() noexcept {
        auto dx_ctx = util::Singleton<DirectX::Context>::instance();
        // root signature
        {
            D3D12_FEATURE_DATA_ROOT_SIGNATURE feat_data{};
            feat_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

            if (FAILED(dx_ctx->device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feat_data, sizeof(feat_data))))
                assert(false);

            D3D12_DESCRIPTOR_RANGE1 ranges[1]{};
            ranges[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
            ranges[0].NumDescriptors                    = 1;
            ranges[0].BaseShaderRegister                = 0;
            ranges[0].RegisterSpace                     = 0;
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
            root_sign_desc.Version                    = D3D_ROOT_SIGNATURE_VERSION_1_1;
            root_sign_desc.Desc_1_1.Flags             = root_sign_flags;
            root_sign_desc.Desc_1_1.NumParameters     = 2;
            root_sign_desc.Desc_1_1.NumStaticSamplers = 0;
            root_sign_desc.Desc_1_1.pParameters       = root_params;

            winrt::com_ptr<ID3DBlob> signature;
            winrt::com_ptr<ID3DBlob> error;
            DirectX::StopIfFailed(D3D12SerializeVersionedRootSignature(&root_sign_desc, signature.put(), error.put()));
            DirectX::StopIfFailed(dx_ctx->device->CreateRootSignature(
                0, signature->GetBufferPointer(), signature->GetBufferSize(),
                winrt::guid_of<ID3D12RootSignature>(), canvas_root_signature.put_void()));
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

            std::filesystem::path file_path   = (std::filesystem::path{ROOT_DIR} / "framework/system/gui/output.hlsl").make_preferred();
            std::wstring          w_file_path = file_path.wstring();
            LPCWSTR               result      = w_file_path.data();

            winrt::com_ptr<ID3DBlob> errors;

            auto hr = D3DCompileFromFile(result, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "VSMain", "vs_5_1", compile_flags, 0, vs.put(), errors.put());
            if (errors != nullptr)
                OutputDebugStringA((char*)errors->GetBufferPointer());
            DirectX::StopIfFailed(hr);

            hr = D3DCompileFromFile(result, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "PSMain", "ps_5_1", compile_flags, 0, ps.put(), errors.put());
            if (errors != nullptr)
                OutputDebugStringA((char*)errors->GetBufferPointer());
            DirectX::StopIfFailed(hr);

            // Define the vertex input layout.
            D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

            // Describe and create the graphics pipeline state object (PSO).
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = {};
            pso_desc.InputLayout                        = {inputElementDescs, _countof(inputElementDescs)};
            pso_desc.pRootSignature                     = canvas_root_signature.get();
            pso_desc.VS.BytecodeLength                  = vs->GetBufferSize();
            pso_desc.VS.pShaderBytecode                 = vs->GetBufferPointer();
            pso_desc.PS.BytecodeLength                  = ps->GetBufferSize();
            pso_desc.PS.pShaderBytecode                 = ps->GetBufferPointer();
            pso_desc.RasterizerState                    = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            pso_desc.BlendState                         = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            pso_desc.DepthStencilState.DepthEnable      = FALSE;
            pso_desc.DepthStencilState.StencilEnable    = FALSE;
            pso_desc.SampleMask                         = UINT_MAX;
            pso_desc.PrimitiveTopologyType              = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso_desc.NumRenderTargets                   = 1;
            pso_desc.RTVFormats[0]                      = DXGI_FORMAT_R8G8B8A8_UNORM;
            pso_desc.SampleDesc.Count                   = 1;
            DirectX::StopIfFailed(dx_ctx->device->CreateGraphicsPipelineState(
                &pso_desc, winrt::guid_of<ID3D12PipelineState>(), canvas_pipeline_state.put_void()));
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
                {-1.f, -1.f, 0.f, 0.f, 0.f},
                {-1.f, 1.f, 0.f, 0.f, 1.f},
                {1.f, -1.f, 0.f, 1.f, 0.f},
                {1.f, 1.f, 0.f, 1.f, 1.f}};

            constexpr auto        vb_size = sizeof(quad);
            D3D12_HEAP_PROPERTIES heap_properties{};
            heap_properties.Type                 = D3D12_HEAP_TYPE_DEFAULT;
            heap_properties.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            heap_properties.CreationNodeMask     = 1;
            heap_properties.VisibleNodeMask      = 1;

            D3D12_RESOURCE_DESC vb_desc{};
            vb_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
            vb_desc.Alignment          = 0;
            vb_desc.Width              = vb_size;
            vb_desc.Height             = 1;
            vb_desc.DepthOrArraySize   = 1;
            vb_desc.MipLevels          = 1;
            vb_desc.Format             = DXGI_FORMAT_UNKNOWN;
            vb_desc.SampleDesc.Count   = 1;
            vb_desc.SampleDesc.Quality = 0;
            vb_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            vb_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;

            DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &vb_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                winrt::guid_of<ID3D12Resource>(), canvas_vb.put_void()));

            heap_properties.Type = D3D12_HEAP_TYPE_UPLOAD;
            DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_NONE,
                &vb_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                winrt::guid_of<ID3D12Resource>(), vb_upload.put_void()));

            D3D12_SUBRESOURCE_DATA vertex_data{};
            vertex_data.pData      = quad;
            vertex_data.RowPitch   = vb_size;
            vertex_data.SlicePitch = vertex_data.RowPitch;

            UpdateSubresources<1>(cmd_list.get(), canvas_vb.get(), vb_upload.get(), 0, 0, 1, &vertex_data);
            D3D12_RESOURCE_BARRIER barrier =
                CD3DX12_RESOURCE_BARRIER::Transition(
                    canvas_vb.get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

            cmd_list->ResourceBarrier(1, &barrier);

            // Initialize the vertex buffer view.
            canvas_vbv.BufferLocation = canvas_vb->GetGPUVirtualAddress();
            canvas_vbv.StrideInBytes  = sizeof(TriVertex);
            canvas_vbv.SizeInBytes    = vb_size;
        }

        dx_ctx->ExecuteCommandLists(cmd_list);
        dx_ctx->Flush();

        {
            CD3DX12_HEAP_PROPERTIES properties(D3D12_HEAP_TYPE_UPLOAD);
            CD3DX12_RESOURCE_DESC   desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(canvas_desc));
            DirectX::StopIfFailed(dx_ctx->device->CreateCommittedResource(
                &properties,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                winrt::guid_of<ID3D12Resource>(),
                canvas_cb.put_void()));

            canvas_cb->Map(0, nullptr, &canvas_cb_mapped_ptr);
        }
    }
}// namespace Pupil::Gui

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
namespace Pupil::Gui {
    LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
            return ::DefWindowProc(hWnd, msg, wParam, lParam);

        switch (msg) {
            case WM_SIZE:
                if (wParam == SIZE_MINIMIZED) {
                    Pupil::util::Singleton<Pupil::Event::Center>::instance()->Send(Pupil::Gui::Event::WindowMinimized);
                } else {
                    if (wParam == SIZE_RESTORED) {
                        Pupil::util::Singleton<Pupil::Event::Center>::instance()->Send(Pupil::Gui::Event::WindowRestored);
                    }
                    Pupil::util::Singleton<Pupil::Event::Center>::instance()
                        ->Send(Pupil::Gui::Event::WindowResized,
                               {static_cast<uint32_t>(LOWORD(lParam)), static_cast<uint32_t>(HIWORD(lParam))});
                }
                return 0;
            case WM_EXITSIZEMOVE:
                return 0;
            case WM_SYSCOMMAND:
                if ((wParam & 0xfff0) == SC_KEYMENU)// Disable ALT application menu
                    return 0;
                break;
            case WM_DESTROY:
                ::PostQuitMessage(0);
                return 0;
        }
        return ::DefWindowProc(hWnd, msg, wParam, lParam);
    }
}// namespace Pupil::Gui