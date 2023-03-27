#include "gui.h"
#include "system.h"
#include "resource.h"

#include "dx12/context.h"

#include "util/event.h"
#include "scene/scene.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"
#include "imfilebrowser.h"

#include "static.h"

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
}// namespace

namespace {
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

inline void SetDocking() noexcept;
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
        EventBinder<WindowEvent::Resize>([this](uint64_t param) {
            uint32_t w = param & 0xffff;
            uint32_t h = (param >> 16) & 0xffff;
            this->Resize(w, h);
        });

        EventBinder<SystemEvent::SceneLoad>([this](uint64_t param) {
            EventDispatcher<WindowEvent::Resize>(param);
            this->AdjustWindowSize();
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

    m_init_flag = true;
}

void GuiPass::SetScene(scene::Scene *scene) noexcept {
    // init render output buffers
    {
        auto buffer_mngr = util::Singleton<BufferManager>::instance();
        uint64_t size =
            static_cast<uint64_t>(scene->sensor.film.h) *
            scene->sensor.film.w * sizeof(float) * 4;
        m_render_output_show_h = static_cast<float>(scene->sensor.film.h);
        m_render_output_show_w = static_cast<float>(scene->sensor.film.w);
        auto dx_ctx = util::Singleton<DirectX::Context>::instance();
        auto descriptor_handle_size = dx_ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto gpu_handle = dx_ctx->srv_heap->GetGPUDescriptorHandleForHeapStart();
        auto cpu_handle = dx_ctx->srv_heap->GetCPUDescriptorHandleForHeapStart();

        for (auto i = 0u; i < SWAP_BUFFER_NUM; ++i) {
            BufferDesc desc{
                .type = EBufferType::SharedCudaWithDX12,
                .name = std::string{ RENDER_OUTPUT_BUFFER[i] },
                .size = size
            };
            m_render_output_buffers[i] = buffer_mngr->AllocBuffer(desc);
            gpu_handle.ptr += size;
            cpu_handle.ptr += size;
            m_render_output_srvs[i] = gpu_handle.ptr;

            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;// TODO:rgba8
            srv_desc.Texture2D.MipLevels = 1;
            dx_ctx->device->CreateShaderResourceView(
                m_render_output_buffers[i]->shared_res.dx12_ptr.get(),
                &srv_desc, cpu_handle);
        }
    }
}

void GuiPass::Destroy() noexcept {
    if (!IsInitialized()) return;

    util::Singleton<DirectX::Context>::instance()->Flush();

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

void GuiPass::Run() noexcept {
    if (!IsInitialized()) return;

    MSG msg = {};
    if (::PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
    }
    OnDraw();
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
        ImGuiID dock_inspector_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.2f, nullptr, &dock_main_id);

        ImGui::DockBuilderDockWindow("Inspector", dock_inspector_id);
        ImGui::DockBuilderDockWindow("Scene", dock_main_id);

        ImGui::DockBuilderFinish(dock_main_id);
    }

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Menu")) {
            if (ImGui::MenuItem("load scene")) {
                m_scene_file_browser.Open();
            }
            if (ImGui::MenuItem("TODO")) {
                printf("test menu item\n");
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (bool open = true;
        ImGui::Begin("Inspector", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {

        ImGui::PushTextWrapPos(0.f);

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

    if (bool open = true;
        ImGui::Begin("Scene", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
        if (auto buffer = GetCurrentRenderOutputBuffer(); buffer) {

            ImGui::Image(
                (ImTextureID)m_render_output_srvs[GetCurrentRenderOutputBufferIndex()],
                ImVec2(m_render_output_show_w, m_render_output_show_h));

        } else {
            ImGui::Text("Render ouput buffer is empty.");
        }
    }
    ImGui::End();

    m_scene_file_browser.Display();

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
                    Pupil::EventDispatcher<Pupil::WindowEvent::Minimized>();
                } else {
                    Pupil::EventDispatcher<Pupil::WindowEvent::Resize>(lParam);
                    // uint32_t w = static_cast<uint32_t>(LOWORD(lParam));
                    // uint32_t h = static_cast<uint32_t>(HIWORD(lParam));
                    // util::Singleton<Window>::instance()->Resize(w, h);
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
            Pupil::EventDispatcher<Pupil::WindowEvent::Quit>();
            ::PostQuitMessage(0);
            return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

void SetDocking() noexcept {
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
        ImGuiID dock_inspector_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.2f, nullptr, &dock_main_id);

        ImGui::DockBuilderDockWindow("Inspector", dock_inspector_id);
        ImGui::DockBuilderDockWindow("Scene", dock_main_id);

        ImGui::DockBuilderFinish(dock_main_id);
    }
}
}// namespace