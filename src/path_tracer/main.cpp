#include "static.h"

#include "path_tracer/type.h"

#include "gui/dx12_backend.h"
#include "gui/window.h"
#include "imgui.h"

#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/optix_wrap/module.h"
#include "device/optix_wrap/pipeline.h"
#include "device/optix_wrap/pass.h"

#include "scene/scene.h"
#include "material/optix_material.h"

#include "optix_util/emitter.h"
#include "optix_util/camera.h"

#include <memory>
#include <iostream>
#include <fstream>

std::unique_ptr<device::Optix> g_optix_device;

std::unique_ptr<optix_wrap::Module> g_sphere_module;
std::unique_ptr<optix_wrap::Module> g_pt_module;

std::unique_ptr<scene::Scene> g_scene;

OptixLaunchParams g_params;

optix_util::CameraDesc g_camera_init_desc;
std::unique_ptr<optix_util::CameraHelper> g_camera;

CUdeviceptr g_emitters_cuda_memory = 0;
std::vector<optix_util::Emitter> g_emitters;

bool g_render_flag = true;

struct SBTTypes {
    using RayGenDataType = RayGenData;
    using MissDataType = MissData;
    using HitGroupDataType = HitGroupData;
};

std::unique_ptr<optix_wrap::Pass<SBTTypes, OptixLaunchParams>> g_pt_pass;

void ConfigOptix();
void InitGuiAndEventCallback();

int main() {
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Init();

    InitGuiAndEventCallback();

    auto backend = gui_window->GetBackend();

    g_optix_device = std::make_unique<device::Optix>(backend->GetDevice());
    g_pt_pass = std::make_unique<optix_wrap::Pass<SBTTypes, OptixLaunchParams>>(g_optix_device->context, g_optix_device->cuda_stream);

    ConfigOptix();
    gui_window->Resize(g_scene->sensor.film.w, g_scene->sensor.film.h, true);

    backend->SetScreenResource(g_optix_device->GetSharedFrameResource());

    do {
        // TODO: handle minimize event
        if (g_render_flag) {
            g_params.camera.SetData(g_camera->GetCudaMemory());
            g_params.frame_buffer = reinterpret_cast<float4 *>(backend->GetCurrentFrameResource().src->cuda_buffer_ptr);
            g_pt_pass->Run(g_params, g_params.config.frame.width, g_params.config.frame.height);

            g_params.sample_cnt += g_params.config.accumulated_flag;
            ++g_params.frame_cnt;
        }

        auto msg = gui_window->Show();
        if (msg == gui::GlobalMessage::Quit)
            break;
        else if (msg == gui::GlobalMessage::Minimized)
            g_render_flag = false;

    } while (true);
    CUDA_FREE(g_emitters_cuda_memory);

    g_camera.reset();
    g_pt_module.reset();
    g_sphere_module.reset();
    g_optix_device.reset();
    gui_window->Destroy();
    return 0;
}

void ConfigPipeline(device::Optix *device) {
    g_sphere_module = std::make_unique<optix_wrap::Module>(device->context, OPTIX_PRIMITIVE_TYPE_SPHERE);
    g_pt_module = std::make_unique<optix_wrap::Module>(device->context, "path_tracer/main.ptx");
    optix_wrap::PipelineDesc pipeline_desc;
    {
        optix_wrap::ProgramDesc desc{
            .module = g_pt_module.get(),
            .ray_gen_entry = "__raygen__main",
            .hit_miss = "__miss__default",
            .shadow_miss = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__default" },
            .shadow_grop = { .ch_entry = "__closesthit__shadow" }
        };
        pipeline_desc.programs.push_back(desc);
    }

    {
        optix_wrap::ProgramDesc desc{
            .module = g_pt_module.get(),
            .hit_group = { .ch_entry = "__closesthit__default", .intersect_module = g_sphere_module.get() },
            .shadow_grop = { .ch_entry = "__closesthit__shadow", .intersect_module = g_sphere_module.get() }
        };
        pipeline_desc.programs.push_back(desc);
    }
    g_pt_pass->InitPipeline(pipeline_desc);
}

void ConfigScene(device::Optix *device) {
    g_scene = std::make_unique<scene::Scene>();
    std::string scene_name = "mis.xml";
    std::ifstream scene_config_file(std::string{ ROOT_DIR } + "/pt_config.ini", std::ios::in);
    if (scene_config_file.is_open()) {
        std::getline(scene_config_file, scene_name);
        scene_config_file.close();
    }
    g_scene->LoadFromXML(scene_name, DATA_DIR);
    device->InitScene(g_scene.get());

    g_emitters = optix_util::GenerateEmitters(g_scene.get());
    g_emitters_cuda_memory = cuda::CudaMemcpy(g_emitters.data(), g_emitters.size() * sizeof(optix_util::Emitter));
    g_params.emitters.SetData(g_emitters_cuda_memory, g_emitters.size());

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
}

void ConfigSBT(device::Optix *device) {
    optix_wrap::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program_name = "__raygen__main",
        .data = SBTTypes::RayGenDataType{}
    };
    {
        int emitter_index_offset = 0;
        using HitGroupDataRecord = decltype(desc)::Pair<SBTTypes::HitGroupDataType>;
        for (auto &&shape : g_scene->shapes) {
            HitGroupDataRecord hit_default_data{};
            hit_default_data.program_name = "__closesthit__default";
            hit_default_data.data.mat.LoadMaterial(shape.mat);
            hit_default_data.data.geo.LoadGeometry(shape);
            if (shape.is_emitter) {
                hit_default_data.data.emitter_index_offset = emitter_index_offset;
                emitter_index_offset += shape.sub_emitters_num;
            }

            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program_name = "__closesthit__shadow";
            hit_shadow_data.data.mat.type = shape.mat.type;
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_data = {
            .program_name = "__miss__default",
            .data = SBTTypes::MissDataType{}
        };
        desc.miss_datas.push_back(miss_data);
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_shadow_data = {
            .program_name = "__miss__shadow",
            .data = SBTTypes::MissDataType{}
        };
        desc.miss_datas.push_back(miss_shadow_data);
    }
    g_pt_pass->InitSBT(desc);
}

void InitLaunchParams(device::Optix *device) {
    g_params.config.frame.width = g_scene->sensor.film.w;
    g_params.config.frame.height = g_scene->sensor.film.h;
    g_params.config.max_depth = g_scene->integrator.max_depth;
    g_params.config.accumulated_flag = true;
    g_params.config.use_tone_mapping = true;

    g_params.frame_cnt = 0;
    g_params.sample_cnt = 0;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&g_params.accum_buffer),
        g_params.config.frame.height * g_params.config.frame.width * sizeof(float4)));

    g_params.frame_buffer = nullptr;
    g_params.handle = device->ias_handle;
}

void ConfigOptix() {
    ConfigPipeline(g_optix_device.get());
    ConfigScene(g_optix_device.get());
    ConfigSBT(g_optix_device.get());
    InitLaunchParams(g_optix_device.get());
}

void InitGuiAndEventCallback() {
    auto gui_window = util::Singleton<gui::Window>::instance();

    gui_window->AppendGuiConsoleOperations(
        "Path Tracer Option",
        []() {
            ImGui::Text("sample count: %d", g_params.sample_cnt + 1);
            ImGui::SameLine();
            if (ImGui::Button(g_render_flag ? "Stop" : "Continue")) {
                g_render_flag ^= 1;
                if (g_render_flag == false) {
                    util::Singleton<gui::Backend>::instance()->SynchronizeFrameResource();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                g_params.sample_cnt = 0;
            }
            if (ImGui::Button("Reset Camera")) {
                g_camera->Reset(g_camera_init_desc);
                g_params.frame_cnt = 0;
                g_params.sample_cnt = 0;
            }

            int depth = g_params.config.max_depth;
            ImGui::InputInt("trace depth", &depth);
            depth = clamp(depth, 1, 128);
            if (g_params.config.max_depth != depth) {
                g_params.config.max_depth = (unsigned int)depth;
                g_params.frame_cnt = 0;
                g_params.sample_cnt = 0;
            }

            if (ImGui::Checkbox("accumulate radiance", &g_params.config.accumulated_flag)) {
                g_params.sample_cnt = 0;
            }
            ImGui::Checkbox("ACES tone mapping", &g_params.config.use_tone_mapping);
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::Resize,
        [&]() {
            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;

            unsigned int &w = g_params.config.frame.width;
            unsigned int &h = g_params.config.frame.height;
            gui_window->GetWindowSize(w, h);

            CUDA_FREE(g_params.accum_buffer);

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_params.accum_buffer), w * h * sizeof(float4)));

            g_optix_device->ClearSharedFrameResource();
            gui_window->GetBackend()->SetScreenResource(g_optix_device->GetSharedFrameResource());

            float aspect = static_cast<float>(w) / h;
            g_camera->SetAspectRatio(aspect);
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::MouseLeftButtonMove,
        [&]() {
            if (!g_render_flag) return;
            float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
            float dx = gui_window->GetMouseLastDeltaX() * scale;
            float dy = gui_window->GetMouseLastDeltaY() * scale;

            g_camera->Rotate(dx, dy);

            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::MouseWheel,
        [&]() {
            if (!g_render_flag) return;
            float fov_delta = 1.f / 120.f * gui_window->GetMouseWheelDelta();
            g_camera->SetFovDelta(fov_delta);

            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::KeyboardMove,
        [&]() {
            if (!g_render_flag) return;

            auto right = util::Camera::X;
            auto up = util::Camera::Y;
            auto forward = util::Camera::Z;

            util::Float3 translation{ 0.f };
            if (gui_window->IsKeyPressed('W') || gui_window->IsKeyPressed(VK_UP)) {
                translation += forward;
            }
            if (gui_window->IsKeyPressed('S') || gui_window->IsKeyPressed(VK_DOWN)) {
                translation -= forward;
            }

            if (gui_window->IsKeyPressed('A') || gui_window->IsKeyPressed(VK_LEFT)) {
                translation += right;
            }
            if (gui_window->IsKeyPressed('D') || gui_window->IsKeyPressed(VK_RIGHT)) {
                translation -= right;
            }

            if (gui_window->IsKeyPressed('Q')) {
                translation += up;
            }
            if (gui_window->IsKeyPressed('E')) {
                translation -= up;
            }

            g_camera->Move(translation * util::Camera::sensitivity * util::Camera::sensitivity_scale);

            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;
        });
}
