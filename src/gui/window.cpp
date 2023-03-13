#include "window.h"
#include "dx12_backend.h"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"

#include "common/camera.h"

#include "static.h"

#include <wincodec.h>
#include "ScreenGrab.h"

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <filesystem>

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
int m_last_mouse_pos_delta_x = 0;
int m_last_mouse_pos_delta_y = 0;

short m_mouse_wheel_delta = 0;

std::array<bool, 256> m_key_is_pressed = { false };

bool m_handling_imgui_flag = false;

std::unordered_map<gui::GlobalMessage, std::function<void()>> m_message_callbacks;
std::vector<std::pair<std::string, std::function<void()>>> m_gui_console_ops;
}// namespace

namespace {

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

inline void ImguiInit() noexcept;

void DrawImGuiConsoleWindow() noexcept;

void OnDraw() noexcept;

void OnMouseDown(WPARAM, LONG, LONG) noexcept;
void OnMouseUp(WPARAM, LONG, LONG) noexcept;
void OnMouseMove(WPARAM, LONG, LONG) noexcept;
void OnMouseWheel(short) noexcept;
void OnKeyDown(WPARAM) noexcept;
void OnKeyUp(WPARAM) noexcept;

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

void Window::AppendGuiConsoleOperations(std::string title, std::function<void()> &&op) noexcept {
    m_gui_console_ops.emplace_back(title, op);
}

GlobalMessage Window::Show() noexcept {
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
    return g_message;
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

void Window::GetWindowSize(uint32_t &w, uint32_t &h) const noexcept {
    w = g_window_w;
    h = g_window_h;
}

void Window::Resize(uint32_t w, uint32_t h, bool reset_window) noexcept {
    if (w != g_window_w || h != g_window_h) {
        g_window_w = w;
        g_window_h = h;
        g_message = GlobalMessage::Resize;
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

int Window::GetMouseLastDeltaX() const noexcept { return m_last_mouse_pos_delta_x; }
int Window::GetMouseLastDeltaY() const noexcept { return m_last_mouse_pos_delta_y; }
short Window::GetMouseWheelDelta() const noexcept { return m_mouse_wheel_delta; }
bool Window::IsKeyPressed(int key) const noexcept { return key >= 0 && key < m_key_is_pressed.size() ? m_key_is_pressed[key] : false; }

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return ::DefWindowProc(hWnd, msg, wParam, lParam);

    if (m_handling_imgui_flag)
        return ::DefWindowProc(hWnd, msg, wParam, lParam);

    POINT cursor_pos;
    switch (msg) {
        case WM_SIZE:
            if (m_backend) {
                if (wParam == SIZE_MINIMIZED) {
                    g_message = gui::GlobalMessage::Minimized;
                } else {
                    uint32_t w = static_cast<uint32_t>(LOWORD(lParam));
                    uint32_t h = static_cast<uint32_t>(HIWORD(lParam));
                    util::Singleton<Window>::instance()->Resize(w, h);
                }
            }
            return 0;
        case WM_EXITSIZEMOVE:
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU)// Disable ALT application menu
                return 0;
            break;
        case WM_KEYDOWN:
            OnKeyDown(wParam);
            break;
        case WM_KEYUP:
            OnKeyUp(wParam);
            break;
        case WM_LBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_RBUTTONDOWN:
            if (GetCursorPos(&cursor_pos)) {
                OnMouseDown(wParam, cursor_pos.x, cursor_pos.y);
            }
            break;
        case WM_LBUTTONUP:
        case WM_MBUTTONUP:
        case WM_RBUTTONUP:
            if (GetCursorPos(&cursor_pos)) {
                OnMouseUp(wParam, cursor_pos.x, cursor_pos.y);
            }
            break;
        case WM_MOUSEMOVE:
            if (GetCursorPos(&cursor_pos)) {
                OnMouseMove(wParam, cursor_pos.x, cursor_pos.y);
                m_last_mouse_pos.x = cursor_pos.x;
                m_last_mouse_pos.y = cursor_pos.y;
            }
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

void DrawImGuiConsoleWindow() noexcept {
    if (ImGui::Begin("Lab Console", nullptr, ImGuiWindowFlags_MenuBar)) {
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Menu")) {
                if (ImGui::MenuItem("Test")) {
                    printf("test menu item\n");
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        if (ImGui::CollapsingHeader("Common info & op", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SeparatorText("info");
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Render Target Size(width x height): %d x %d", g_window_w, g_window_h);

            ImGui::SeparatorText("op");
            ImGui::Text("Camera:");
            ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.4f);
            ImGui::InputFloat("sensitivity scale", &util::Camera::sensitivity_scale, 0.1f, 1.0f, "%.1f");
            ImGui::PopItemWidth();

            ImGui::Text("Save rendering screen shot:");
            // save image
            {
                ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.2f);
                static char file_name[256]{};
                ImGui::InputText("file name", file_name, 256);
                ImGui::SameLine();
                constexpr auto image_file_format = std::array{ "jpg", "png" };
                static int item_current = 0;
                ImGui::Combo("format", &item_current, image_file_format.data(), (int)image_file_format.size());
                ImGui::SameLine();
                if (ImGui::Button("Save")) {
                    std::filesystem::path path{ ROOT_DIR };
                    path /= std::string{ file_name } + "." + image_file_format[item_current];
                    StopIfFailed(
                        DirectX::SaveWICTextureToFile(
                            m_backend->GetDevice()->cmd_queue.Get(),
                            m_backend->GetDevice()->GetCurrentFrame().buffer.Get(),
                            GUID_ContainerFormatJpeg, path.wstring().data(),
                            D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_PRESENT,
                            nullptr, nullptr, true));
                    printf("image was saved successfully in [%ws].\n", path.wstring().data());
                }
                ImGui::PopItemWidth();
            }
        }

        for (auto &&[title, op] : m_gui_console_ops) {
            if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                op();
            }
        }
    }

    m_handling_imgui_flag = ImGui::IsWindowFocused() && ImGui::IsWindowHovered();

    ImGui::End();
}

void OnDraw() noexcept {
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    DrawImGuiConsoleWindow();

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
    if ((btn_state & MK_LBUTTON) != 0) {
        m_last_mouse_pos_delta_x = x - m_last_mouse_pos.x;
        m_last_mouse_pos_delta_y = y - m_last_mouse_pos.y;
        InvokeGuiEventCallback(gui::GlobalMessage::MouseLeftButtonMove);
    }
    if ((btn_state & MK_RBUTTON) != 0) {
        m_last_mouse_pos_delta_x = x - m_last_mouse_pos.x;
        m_last_mouse_pos_delta_y = y - m_last_mouse_pos.y;
        InvokeGuiEventCallback(gui::GlobalMessage::MouseRightButtonMove);
    }
}

void OnMouseWheel(short zDelta) noexcept {
    m_mouse_wheel_delta = zDelta;
    InvokeGuiEventCallback(gui::GlobalMessage::MouseWheel);
}

void OnKeyUp(WPARAM key) noexcept {
    m_key_is_pressed[key] = false;
}

void OnKeyDown(WPARAM key) noexcept {
    m_key_is_pressed[key] = true;
    switch (key) {
        case VK_UP:
        case 'W':
        case VK_DOWN:
        case 'S':
        case VK_LEFT:
        case 'A':
        case VK_RIGHT:
        case 'D':
        case 'Q':
        case 'E':
            g_message = gui::GlobalMessage::KeyboardMove;
            break;
    }
}

}// namespace