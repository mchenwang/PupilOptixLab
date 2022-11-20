#include "window.h"
#include "dx12_backend.h"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"

#include <string>

using namespace gui;

HWND g_window_handle;
uint32_t g_window_w = 1280;
uint32_t g_window_h = 800;

namespace {
const std::wstring WND_NAME = L"OptixReSTIR";
const std::wstring WND_CLASS_NAME = L"OptixReSTIR_CLASS";

HINSTANCE m_instance;

Backend *m_backend = nullptr;
}// namespace

namespace {

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

inline void ImguiInit() noexcept;

void OnDraw() noexcept;

}// namespace

void Window::Init() {
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

    m_instance = GetModuleHandleW(NULL);
    g_window_handle = ::CreateWindowExW(NULL,
        WND_CLASS_NAME.data(),
        WND_NAME.data(),
        WS_OVERLAPPEDWINDOW,
        100, 100, g_window_w, g_window_h,
        NULL, NULL, wc.hInstance, NULL);

    util::Singleton<Backend>::instance()->Init();
    m_backend = util::Singleton<Backend>::instance();

    ::ShowWindow(g_window_handle, SW_SHOW);
    ::UpdateWindow(g_window_handle);

    ImguiInit();
}

void Window::Show() {
    MSG msg = {};
    while (msg.message != WM_QUIT) {
        if (::PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
        } else {
            OnDraw();
        }
    }
}

void Window::Destroy() {
    m_backend->Flush();

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    m_backend->Destroy();
    ::DestroyWindow(g_window_handle);
    ::UnregisterClassW(WND_CLASS_NAME.data(), m_instance);
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
namespace {

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            if (m_backend) {
                uint32_t w = static_cast<uint32_t>(LOWORD(lParam));
                uint32_t h = static_cast<uint32_t>(HIWORD(lParam));
                if (w != g_window_w || h != g_window_h)
                    m_backend->Resize(w, h);
            }
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

inline void ImguiInit() noexcept {
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(g_window_handle);

    ImGui_ImplDX12_Init(
        m_backend->device.Get(),
        Backend::NUM_OF_FRAMES,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        m_backend->srv_heap.Get(),
        m_backend->srv_heap->GetCPUDescriptorHandleForHeapStart(),
        m_backend->srv_heap->GetGPUDescriptorHandleForHeapStart());
}

void OnDraw() noexcept {
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!");// Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");// Display some text (you can use a format strings too)
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    ImGui::Render();
    auto cmd_list = m_backend->GetCmdList();
    m_backend->RenderScreen(cmd_list);

    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list.Get());

    m_backend->Present(cmd_list);
}

}// namespace