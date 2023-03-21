#pragma once

#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/cuda_texture.h"
#include "device/optix_wrap/module.h"
#include "device/optix_wrap/pipeline.h"
#include "device/optix_wrap/pass.h"

#include "scene/scene.h"
#include "gui/dx12_backend.h"

#include "ddgi/pt/type.h"

#include <memory>

struct PTPassSBTTypes {
    using RayGenDataType = RayGenData;
    using MissDataType = MissData;
    using HitGroupDataType = HitGroupData;
};

struct PTPass {
    device::Optix *optix_device = nullptr;
    std::unique_ptr<optix_wrap::Pass<PTPassSBTTypes, PTPassOptixLaunchParams>> pass;
    PTPassOptixLaunchParams params;

    std::unique_ptr<optix_wrap::Module> sphere_module;
    std::unique_ptr<optix_wrap::Module> pt_module;

    optix_util::CameraHelper *camera;
    optix_util::EmitterHelper *emitters;

    PTPass(device::Optix *device) noexcept {
        optix_device = device;
        pass = std::make_unique<optix_wrap::Pass<PTPassSBTTypes, PTPassOptixLaunchParams>>(optix_device->context, optix_device->cuda_stream);
        ConfigPipeline();
    }

    void Run() noexcept {
        params.camera.SetData(camera->GetCudaMemory());
        auto backend = util::Singleton<gui::Window>::instance()->GetBackend();
        params.frame_buffer = reinterpret_cast<float4 *>(backend->GetCurrentFrameResource().src->cuda_buffer_ptr);
        pass->Run(params, params.config.frame.width, params.config.frame.height);

        params.sample_cnt += params.config.accumulated_flag;
        ++params.frame_cnt;
    }

    void ConfigPipeline() noexcept {
        sphere_module = std::make_unique<optix_wrap::Module>(optix_device->context, OPTIX_PRIMITIVE_TYPE_SPHERE);
        pt_module = std::make_unique<optix_wrap::Module>(optix_device->context, "ddgi/pt/pt_main.ptx");
        optix_wrap::PipelineDesc pipeline_desc;
        {
            optix_wrap::ProgramDesc desc{
                .module = pt_module.get(),
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
                .module = pt_module.get(),
                .hit_group = { .ch_entry = "__closesthit__default", .intersect_module = sphere_module.get() },
                .shadow_grop = { .ch_entry = "__closesthit__shadow", .intersect_module = sphere_module.get() }
            };
            pipeline_desc.programs.push_back(desc);
        }
        pass->InitPipeline(pipeline_desc);
    }

    void ConfigSBT(scene::Scene *scene) noexcept {
        optix_wrap::SBTDesc<PTPassSBTTypes> desc{};
        desc.ray_gen_data = {
            .program_name = "__raygen__main",
            .data = PTPassSBTTypes::RayGenDataType{}
        };
        {
            int emitter_index_offset = 0;
            using HitGroupDataRecord = decltype(desc)::Pair<PTPassSBTTypes::HitGroupDataType>;
            for (auto &&shape : scene->shapes) {
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
            decltype(desc)::Pair<PTPassSBTTypes::MissDataType> miss_data = {
                .program_name = "__miss__default",
                .data = PTPassSBTTypes::MissDataType{}
            };
            desc.miss_datas.push_back(miss_data);
            decltype(desc)::Pair<PTPassSBTTypes::MissDataType> miss_shadow_data = {
                .program_name = "__miss__shadow",
                .data = PTPassSBTTypes::MissDataType{}
            };
            desc.miss_datas.push_back(miss_shadow_data);
        }
        pass->InitSBT(desc);
    }

    void InitLaunchParams(scene::Scene *scene) {
        params.config.frame.width = scene->sensor.film.w;
        params.config.frame.height = scene->sensor.film.h;
        params.config.max_depth = scene->integrator.max_depth;
        params.config.accumulated_flag = true;
        params.config.use_tone_mapping = false;

        params.frame_cnt = 0;
        params.sample_cnt = 0;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&params.accum_buffer),
            params.config.frame.height * params.config.frame.width * sizeof(float4)));

        params.frame_buffer = nullptr;
        params.handle = optix_device->ias_handle;
    }

    void SetScene(scene::Scene *scene, optix_util::CameraHelper *camera,
                  optix_util::EmitterHelper *emitters) noexcept {
        this->camera = camera;
        this->emitters = emitters;
        params.emitters = emitters->GetEmitterGroup();

        ConfigSBT(scene);
        InitLaunchParams(scene);
    }

    void SetGui() noexcept {
        auto gui_window = util::Singleton<gui::Window>::instance();

        // gui_window->AppendGuiConsoleOperations(
        //     "Path Tracer Option",
        //     []() {
        //         ImGui::SeparatorText("scene");
        //         {
        //             ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.5f);
        //             ImGui::InputText("scene file", s_scene_name, 256);
        //             ImGui::SameLine();
        //             if (ImGui::Button("Load")) {
        //                 LoadScene(s_scene_name, "");
        //                 params.frame_cnt = 0;
        //                 params.sample_cnt = 0;
        //                 g_render_flag = true;
        //             }
        //             ImGui::PopItemWidth();

        //             if (ImGui::Button("Reset Camera")) {
        //                 g_camera->Reset(g_camera_init_desc);
        //                 params.frame_cnt = 0;
        //                 params.sample_cnt = 0;
        //             }
        //         }

        //         ImGui::SeparatorText("render options");
        //         ImGui::Text("sample count: %d", params.sample_cnt + 1);
        //         ImGui::SameLine();
        //         if (ImGui::Button(g_render_flag ? "Stop" : "Continue")) {
        //             g_render_flag ^= 1;
        //             if (g_render_flag == false) {
        //                 util::Singleton<gui::Backend>::instance()->SynchronizeFrameResource();
        //             }
        //         }
        //         ImGui::SameLine();
        //         if (ImGui::Button("Reset")) {
        //             params.sample_cnt = 0;
        //         }

        //         int depth = params.config.max_depth;
        //         ImGui::InputInt("trace depth", &depth);
        //         if (depth < 1)
        //             depth = 1;
        //         else if (depth > 128)
        //             depth = 128;
        //         if (params.config.max_depth != depth) {
        //             params.config.max_depth = (unsigned int)depth;
        //             params.frame_cnt = 0;
        //             params.sample_cnt = 0;
        //         }

        //         if (ImGui::Checkbox("accumulate radiance", &params.config.accumulated_flag)) {
        //             params.sample_cnt = 0;
        //         }
        //         ImGui::Checkbox("ACES tone mapping", &params.config.use_tone_mapping);
        //     });

        // gui_window->SetWindowMessageCallback(
        //     gui::GlobalMessage::Resize,
        //     [&]() {
        //         params.frame_cnt = 0;
        //         params.sample_cnt = 0;

        //         unsigned int &w = params.config.frame.width;
        //         unsigned int &h = params.config.frame.height;
        //         gui_window->GetWindowSize(w, h);

        //         CUDA_FREE(params.accum_buffer);

        //         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&params.accum_buffer), w * h * sizeof(float4)));

        //         g_optix_device->ClearSharedFrameResource();
        //         gui_window->GetBackend()->SetScreenResource(g_optix_device->GetSharedFrameResource());

        //         float aspect = static_cast<float>(w) / h;
        //         g_camera->SetAspectRatio(aspect);
        //     });

        // gui_window->SetWindowMessageCallback(
        //     gui::GlobalMessage::MouseLeftButtonMove,
        //     [&]() {
        //         if (!g_render_flag) return;
        //         float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
        //         float dx = gui_window->GetMouseLastDeltaX() * scale;
        //         float dy = gui_window->GetMouseLastDeltaY() * scale;

        //         g_camera->Rotate(dx, dy);

        //         params.frame_cnt = 0;
        //         params.sample_cnt = 0;
        //     });

        // gui_window->SetWindowMessageCallback(
        //     gui::GlobalMessage::MouseWheel,
        //     [&]() {
        //         if (!g_render_flag) return;
        //         float fov_delta = 1.f / 120.f * gui_window->GetMouseWheelDelta();
        //         g_camera->SetFovDelta(fov_delta);

        //         params.frame_cnt = 0;
        //         params.sample_cnt = 0;
        //     });

        // gui_window->SetWindowMessageCallback(
        //     gui::GlobalMessage::KeyboardMove,
        //     [&]() {
        //         if (!g_render_flag) return;

        //         auto right = util::Camera::X;
        //         auto up = util::Camera::Y;
        //         auto forward = util::Camera::Z;

        //         util::Float3 translation{ 0.f };
        //         if (gui_window->IsKeyPressed('W') || gui_window->IsKeyPressed(VK_UP)) {
        //             translation += forward;
        //         }
        //         if (gui_window->IsKeyPressed('S') || gui_window->IsKeyPressed(VK_DOWN)) {
        //             translation -= forward;
        //         }

        //         if (gui_window->IsKeyPressed('A') || gui_window->IsKeyPressed(VK_LEFT)) {
        //             translation += right;
        //         }
        //         if (gui_window->IsKeyPressed('D') || gui_window->IsKeyPressed(VK_RIGHT)) {
        //             translation -= right;
        //         }

        //         if (gui_window->IsKeyPressed('Q')) {
        //             translation += up;
        //         }
        //         if (gui_window->IsKeyPressed('E')) {
        //             translation -= up;
        //         }

        //         g_camera->Move(translation * util::Camera::sensitivity * util::Camera::sensitivity_scale);

        //         params.frame_cnt = 0;
        //         params.sample_cnt = 0;
        //     });
    }
};