#include "gui.h"

#include "dx12/context.h"

#include "event.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"

namespace Pupil {
HWND g_window_handle;
uint32_t g_window_w = 1280;
uint32_t g_window_h = 720;
}// namespace Pupil

namespace {
const std::wstring WND_NAME = L"PupilOptixLab";
const std::wstring WND_CLASS_NAME = L"PupilOptixLab_CLASS";
HINSTANCE m_instance;
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

    // event binding
    {
        EventBinder<WindowEvent::Resize>([this](uint64_t param) {
            uint32_t w = param & 0xffff;
            uint32_t h = (param >> 16) & 0xffff;
            this->Resize(w, h);
        });
    }

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
    }

    m_init_flag = true;
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

void GuiPass::RegisterGui(std::string_view name, CustomGui &&gui) noexcept {
    m_guis.emplace(name, gui);
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
            if (ImGui::MenuItem("TODO")) {
                printf("test menu item\n");
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (bool open = true;
        ImGui::Begin("Inspector", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
        ImGui::Text("TODO");
        for (auto &&[title, gui] : m_guis) {
            if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                gui();
            }
        }
    }
    ImGui::End();

    if (bool open = true;
        ImGui::Begin("Scene", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
        ImGui::Text("TODO");
    }
    ImGui::End();

    ImGui::Render();

    auto dx_ctx = util::Singleton<DirectX::Context>::instance();
    auto cmd_list = dx_ctx->GetCmdList();

    ID3D12DescriptorHeap *heaps[] = { dx_ctx->srv_heap.get() };
    cmd_list->SetDescriptorHeaps(1, heaps);
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