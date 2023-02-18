#include "window.h"
#include "dx12_backend.h"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"

#include <string>
#include <vector>
#include <unordered_map>

using namespace gui;

HWND g_window_handle;
uint32_t g_window_w = 1280;
uint32_t g_window_h = 720;

GlobalMessage g_message = GlobalMessage::None;

namespace {
const std::wstring WND_NAME = L"PupilOptixLab";
const std::wstring WND_CLASS_NAME = L"PupilOptixLab_CLASS";

HINSTANCE m_instance;

Backend *m_backend = nullptr;

POINT m_last_mouse_pos;
uint32_t m_last_mouse_pos_delta_x = 0;
uint32_t m_last_mouse_pos_delta_y = 0;

short m_mouse_wheel_delta = 0;

std::unordered_map<gui::GlobalMessage, std::function<void()>> m_message_callbacks;
std::vector<std::function<void()>> m_gui_console_ops;
}// namespace

namespace {

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

inline void ImguiInit() noexcept;

void OnDraw() noexcept;

void OnMouseDown(WPARAM, LONG, LONG) noexcept;
void OnMouseUp(WPARAM, LONG, LONG) noexcept;
void OnMouseMove(WPARAM, LONG, LONG) noexcept;
void OnMouseWheel(short) noexcept;

inline void InvokeGuiEventCallback(gui::GlobalMessage msg) noexcept {
    auto it = m_message_callbacks.find(msg);
    if (it != m_message_callbacks.end())
        it->second();
}

}// namespace

void Window::Init() noexcept {
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

    util::Singleton<Backend>::instance()->Init();
    m_backend = util::Singleton<Backend>::instance();

    ::ShowWindow(g_window_handle, SW_SHOW);
    ::UpdateWindow(g_window_handle);

    ImguiInit();
}

void Window::SetWindowMessageCallback(GlobalMessage message, std::function<void()> &&callback) noexcept {
    m_message_callbacks[message] = callback;
}

void Window::AppendGuiConsoleOperations(std::function<void()> &&op) noexcept {
    m_gui_console_ops.emplace_back(op);
}

void Window::Show() noexcept {
    MSG msg = {};
    g_message = GlobalMessage::None;
    if (::PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
    }
    OnDraw();

    if (msg.message == WM_QUIT)
        g_message = GlobalMessage::Quit;

    InvokeGuiEventCallback(g_message);
}

void Window::Destroy() noexcept {
    m_backend->Flush();

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    m_backend->Destroy();
    ::DestroyWindow(g_window_handle);
    ::UnregisterClassW(WND_CLASS_NAME.data(), m_instance);
}

Backend *Window::GetBackend() const noexcept {
    return m_backend;
}

void Window::GetWindowSize(uint32_t &w, uint32_t &h) noexcept {
    w = g_window_w;
    h = g_window_h;
}

void Window::Resize(uint32_t w, uint32_t h, bool reset_window) noexcept {
    if (w != g_window_w || h != g_window_h) {
        g_window_w = w;
        g_window_h = h;
        m_backend->Resize(w, h);
    }
    if (reset_window) {
        RECT window_rect{ 0, 0, static_cast<LONG>(w), static_cast<LONG>(h) };
        ::AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, FALSE);

        int window_w = window_rect.right - window_rect.left;
        int window_h = window_rect.bottom - window_rect.top;

        ::SetWindowPos(g_window_handle, 0, 0, 0, window_w, window_h, SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
    }
}

uint32_t Window::GetMouseLastDeltaX() noexcept { return m_last_mouse_pos_delta_x; }
uint32_t Window::GetMouseLastDeltaY() noexcept { return m_last_mouse_pos_delta_y; }
short Window::GetMouseWheelDelta() noexcept { return m_mouse_wheel_delta; }

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    LONG x, y;
    switch (msg) {
        case WM_SIZE:
            if (m_backend) {
                uint32_t w = static_cast<uint32_t>(LOWORD(lParam));
                uint32_t h = static_cast<uint32_t>(HIWORD(lParam));
                util::Singleton<Window>::instance()->Resize(w, h);
            }
            return 0;
        case WM_EXITSIZEMOVE:
            g_message = GlobalMessage::Resize;
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU)// Disable ALT application menu
                return 0;
            break;
        case WM_LBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_RBUTTONDOWN:
            x = static_cast<LONG>(LOWORD(lParam));
            y = static_cast<LONG>(HIWORD(lParam));
            OnMouseDown(wParam, x, y);
            break;
        case WM_LBUTTONUP:
        case WM_MBUTTONUP:
        case WM_RBUTTONUP:
            x = static_cast<LONG>(LOWORD(lParam));
            y = static_cast<LONG>(HIWORD(lParam));
            OnMouseUp(wParam, x, y);
            break;
        case WM_MOUSEMOVE:
            x = static_cast<LONG>(LOWORD(lParam));
            y = static_cast<LONG>(HIWORD(lParam));
            OnMouseMove(wParam, x, y);

            m_last_mouse_pos.x = x;
            m_last_mouse_pos.y = y;
            break;

        case WM_MOUSEWHEEL:
            OnMouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));
            break;
        case WM_DESTROY:
            ::PostQuitMessage(0);
            return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

inline void ImguiInit() noexcept {
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui_ImplWin32_Init(g_window_handle);

    ImGui_ImplDX12_Init(
        m_backend->GetDevice()->device.Get(),
        m_backend->GetDevice()->NUM_OF_FRAMES,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        m_backend->GetDevice()->srv_heap.Get(),
        m_backend->GetDevice()->srv_heap->GetCPUDescriptorHandleForHeapStart(),
        m_backend->GetDevice()->srv_heap->GetGPUDescriptorHandleForHeapStart());
}

void OnDraw() noexcept {
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Lab Console");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        for (auto &&op : m_gui_console_ops) {
            op();
        }

        ImGui::End();
    }

    ImGui::Render();
    auto cmd_list = m_backend->GetCmdList();
    m_backend->RenderScreen(cmd_list);

    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list.Get());

    auto &io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault(NULL, (void *)cmd_list.Get());
    }

    m_backend->Present(cmd_list);
}

void OnMouseDown(WPARAM btn_state, LONG x, LONG y) noexcept {
    m_last_mouse_pos.x = x;
    m_last_mouse_pos.y = y;
    ::SetCapture(g_window_handle);
}

void OnMouseUp(WPARAM btn_state, LONG x, LONG y) noexcept {
    ::ReleaseCapture();
}

void OnMouseMove(WPARAM btn_state, LONG x, LONG y) noexcept {
    m_last_mouse_pos_delta_x = static_cast<uint32_t>(x - m_last_mouse_pos.x);
    m_last_mouse_pos_delta_y = static_cast<uint32_t>(y - m_last_mouse_pos.y);
    if ((btn_state & MK_LBUTTON) != 0) {
        InvokeGuiEventCallback(gui::GlobalMessage::MouseLeftButtonMove);
    }
    if ((btn_state & MK_RBUTTON) != 0) {
        InvokeGuiEventCallback(gui::GlobalMessage::MouseRightButtonMove);
    }
}

void OnMouseWheel(short zDelta) noexcept {
    m_mouse_wheel_delta = zDelta;
    InvokeGuiEventCallback(gui::GlobalMessage::MouseWheel);
}

}// namespace