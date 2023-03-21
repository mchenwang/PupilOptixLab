#include "static.h"

#include "gui/dx12_backend.h"
#include "gui/window.h"
#include "imgui.h"

#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/cuda_texture.h"
#include "device/optix_wrap/module.h"
#include "device/optix_wrap/pipeline.h"
#include "device/optix_wrap/pass.h"

#include "scene/scene.h"
#include "scene/texture.h"
#include "material/optix_material.h"

#include "optix_util/emitter.h"
#include "optix_util/camera.h"

#include "ddgi/pt/pass.h"
#include "ddgi/gbuffer/pass.h"

#include <memory>
#include <iostream>
#include <fstream>

std::unique_ptr<device::Optix> g_optix_device;
std::unique_ptr<PTPass> g_pt_pass;
std::unique_ptr<GBufferPass> g_gbuffer_pass;

static char s_scene_name[256];
std::unique_ptr<scene::Scene> g_scene;

optix_util::CameraDesc g_camera_init_desc;
std::unique_ptr<optix_util::CameraHelper> g_camera;

std::unique_ptr<optix_util::EmitterHelper> g_emitters;

bool g_render_flag = true;

// void ConfigOptixPipeline() noexcept;
void LoadScene(std::string_view, std::string_view default_scene = "default.xml") noexcept;
// void InitGuiAndEventCallback() noexcept;

int main() {
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Init();

    // InitGuiAndEventCallback();

    auto backend = gui_window->GetBackend();

    g_optix_device = std::make_unique<device::Optix>(backend->GetDevice());
    g_pt_pass = std::make_unique<PTPass>(g_optix_device.get());
    g_gbuffer_pass = std::make_unique<GBufferPass>(g_optix_device.get());

    std::string scene_name;
    std::ifstream scene_config_file(std::string{ ROOT_DIR } + "/pt_config.ini", std::ios::in);
    if (scene_config_file.is_open()) {
        std::getline(scene_config_file, scene_name);
        scene_config_file.close();
    }
    LoadScene(scene_name, "default.xml");

    do {
        if (g_render_flag) {
            g_gbuffer_pass->Run();
            g_pt_pass->Run();
        }

        auto msg = gui_window->Show();
        if (msg == gui::GlobalMessage::Quit)
            break;
        else if (msg == gui::GlobalMessage::Minimized)
            g_render_flag = false;

    } while (true);

    g_emitters.reset();
    g_camera.reset();
    g_optix_device.reset();
    gui_window->Destroy();
    return 0;
}

void LoadScene(std::string_view scene_file, std::string_view default_scene) noexcept {
    std::filesystem::path scene_file_path{ DATA_DIR };
    scene_file_path /= scene_file;
    if (!std::filesystem::exists(scene_file_path)) {
        std::cout << std::format("warning: scene file [{}] does not exist.\n", scene_file_path.string());
        if (default_scene.empty()) return;
        scene_file = "default.xml";
        scene_file_path = scene_file_path.parent_path() / "default.xml";
    }
    memcpy(s_scene_name, scene_file.data(), scene_file.size() * sizeof(char));

    if (g_scene == nullptr)
        g_scene = std::make_unique<scene::Scene>();

    util::Singleton<device::CudaTextureManager>::instance()->Clear();

    g_scene->LoadFromXML(scene_file_path);
    g_optix_device->InitScene(g_scene.get());

    if (!g_emitters) {
        g_emitters = std::make_unique<optix_util::EmitterHelper>(g_scene.get());
    } else {
        g_emitters->Reset(g_scene.get());
    }
    // g_params.emitters = g_emitters->GetEmitterGroup();

    auto &&sensor = g_scene->sensor;
    optix_util::CameraDesc desc{
        .fov_y = sensor.fov,
        .aspect_ratio = static_cast<float>(sensor.film.w) / sensor.film.h,
        .near_clip = sensor.near_clip,
        .far_clip = sensor.far_clip,
        .to_world = sensor.transform
    };
    g_camera_init_desc = desc;
    g_camera = std::make_unique<optix_util::CameraHelper>(desc);

    // ConfigSBT();
    // InitLaunchParams();
    g_pt_pass->SetScene(g_scene.get(), g_camera.get(), g_emitters.get());
    g_gbuffer_pass->SetScene(g_scene.get(), g_camera.get(), g_emitters.get());

    util::Singleton<scene::ShapeDataManager>::instance()->Clear();
    util::Singleton<scene::TextureManager>::instance()->Clear();

    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Resize(g_scene->sensor.film.w, g_scene->sensor.film.h, true);
    g_optix_device->ClearSharedFrameResource();
    gui_window->GetBackend()->SetScreenResource(g_optix_device->GetSharedFrameResource());
}

// void InitGuiAndEventCallback() noexcept {
//     auto gui_window = util::Singleton<gui::Window>::instance();

//     gui_window->AppendGuiConsoleOperations(
//         "Path Tracer Option",
//         []() {
//             ImGui::SeparatorText("scene");
//             {
//                 ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.5f);
//                 ImGui::InputText("scene file", s_scene_name, 256);
//                 ImGui::SameLine();
//                 if (ImGui::Button("Load")) {
//                     LoadScene(s_scene_name, "");
//                     g_params.frame_cnt = 0;
//                     g_params.sample_cnt = 0;
//                     g_render_flag = true;
//                 }
//                 ImGui::PopItemWidth();

//                 if (ImGui::Button("Reset Camera")) {
//                     g_camera->Reset(g_camera_init_desc);
//                     g_params.frame_cnt = 0;
//                     g_params.sample_cnt = 0;
//                 }
//             }

//             ImGui::SeparatorText("render options");
//             ImGui::Text("sample count: %d", g_params.sample_cnt + 1);
//             ImGui::SameLine();
//             if (ImGui::Button(g_render_flag ? "Stop" : "Continue")) {
//                 g_render_flag ^= 1;
//                 if (g_render_flag == false) {
//                     util::Singleton<gui::Backend>::instance()->SynchronizeFrameResource();
//                 }
//             }
//             ImGui::SameLine();
//             if (ImGui::Button("Reset")) {
//                 g_params.sample_cnt = 0;
//             }

//             int depth = g_params.config.max_depth;
//             ImGui::InputInt("trace depth", &depth);
//             depth = clamp(depth, 1, 128);
//             if (g_params.config.max_depth != depth) {
//                 g_params.config.max_depth = (unsigned int)depth;
//                 g_params.frame_cnt = 0;
//                 g_params.sample_cnt = 0;
//             }

//             if (ImGui::Checkbox("accumulate radiance", &g_params.config.accumulated_flag)) {
//                 g_params.sample_cnt = 0;
//             }
//             ImGui::Checkbox("ACES tone mapping", &g_params.config.use_tone_mapping);
//         });

//     gui_window->SetWindowMessageCallback(
//         gui::GlobalMessage::Resize,
//         [&]() {
//             g_params.frame_cnt = 0;
//             g_params.sample_cnt = 0;

//             unsigned int &w = g_params.config.frame.width;
//             unsigned int &h = g_params.config.frame.height;
//             gui_window->GetWindowSize(w, h);

//             CUDA_FREE(g_params.accum_buffer);

//             CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_params.accum_buffer), w * h * sizeof(float4)));

//             g_optix_device->ClearSharedFrameResource();
//             gui_window->GetBackend()->SetScreenResource(g_optix_device->GetSharedFrameResource());

//             float aspect = static_cast<float>(w) / h;
//             g_camera->SetAspectRatio(aspect);
//         });

//     gui_window->SetWindowMessageCallback(
//         gui::GlobalMessage::MouseLeftButtonMove,
//         [&]() {
//             if (!g_render_flag) return;
//             float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
//             float dx = gui_window->GetMouseLastDeltaX() * scale;
//             float dy = gui_window->GetMouseLastDeltaY() * scale;

//             g_camera->Rotate(dx, dy);

//             g_params.frame_cnt = 0;
//             g_params.sample_cnt = 0;
//         });

//     gui_window->SetWindowMessageCallback(
//         gui::GlobalMessage::MouseWheel,
//         [&]() {
//             if (!g_render_flag) return;
//             float fov_delta = 1.f / 120.f * gui_window->GetMouseWheelDelta();
//             g_camera->SetFovDelta(fov_delta);

//             g_params.frame_cnt = 0;
//             g_params.sample_cnt = 0;
//         });

//     gui_window->SetWindowMessageCallback(
//         gui::GlobalMessage::KeyboardMove,
//         [&]() {
//             if (!g_render_flag) return;

//             auto right = util::Camera::X;
//             auto up = util::Camera::Y;
//             auto forward = util::Camera::Z;

//             util::Float3 translation{ 0.f };
//             if (gui_window->IsKeyPressed('W') || gui_window->IsKeyPressed(VK_UP)) {
//                 translation += forward;
//             }
//             if (gui_window->IsKeyPressed('S') || gui_window->IsKeyPressed(VK_DOWN)) {
//                 translation -= forward;
//             }

//             if (gui_window->IsKeyPressed('A') || gui_window->IsKeyPressed(VK_LEFT)) {
//                 translation += right;
//             }
//             if (gui_window->IsKeyPressed('D') || gui_window->IsKeyPressed(VK_RIGHT)) {
//                 translation -= right;
//             }

//             if (gui_window->IsKeyPressed('Q')) {
//                 translation += up;
//             }
//             if (gui_window->IsKeyPressed('E')) {
//                 translation -= up;
//             }

//             g_camera->Move(translation * util::Camera::sensitivity * util::Camera::sensitivity_scale);

//             g_params.frame_cnt = 0;
//             g_params.sample_cnt = 0;
//         });
// }
