#include "static.h"

#include "path_tracer/type.h"

#include "gui/dx12_backend.h"
#include "gui/window.h"

#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/optix_wrap/module.h"
#include "device/optix_wrap/pipeline.h"

#include "scene/scene.h"
#include "material/optix_material.h"

#include <memory>
#include <iostream>

std::unique_ptr<optix_wrap::Module> g_sphere_module;
std::unique_ptr<optix_wrap::Module> g_ReSTIR_module;

std::unique_ptr<scene::Scene> g_scene;

OptixLaunchParams g_params;

struct SBTTypes {
    using RayGenDataType = RayGenData;
    using MissDataType = MissData;
    using HitGroupDataType = HitGroupData;
};

void ConfigOptix(device::Optix *device);

int main() {
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Init();

    //
    {
        auto backend = gui_window->GetBackend();
        std::unique_ptr<device::Optix> optix_device =
            std::make_unique<device::Optix>(backend->GetDevice());

        ConfigOptix(optix_device.get());
        gui_window->Resize(g_scene->sensor.film.w, g_scene->sensor.film.h, true);

        backend->SetScreenResource(optix_device->GetSharedFrameResource());

        do {
            optix_device->Run(&g_params, sizeof(g_params), reinterpret_cast<void**>(&g_params.frame_buffer));
            auto windows_message = gui_window->Show();

            if (windows_message == gui::GlobalMessage::Quit)
                break;
            if (windows_message == gui::GlobalMessage::Resize) {
                gui_window->GetWindowSize(g_params.config.frame.width, g_params.config.frame.height);

                CUDA_FREE(g_params.accum_buffer);

                CUDA_CHECK(cudaMalloc(
                    reinterpret_cast<void **>(&g_params.accum_buffer),
                    g_params.config.frame.height * g_params.config.frame.width * sizeof(float4)));

                optix_device->ClearSharedFrameResource();
                backend->SetScreenResource(optix_device->GetSharedFrameResource());
            }
        } while (true);
    }
    g_ReSTIR_module.reset();
    g_sphere_module.reset();
    gui_window->Destroy();
    return 0;
}

void ConfigPipeline(device::Optix *device) {
    g_sphere_module = std::make_unique<optix_wrap::Module>(device, OPTIX_PRIMITIVE_TYPE_SPHERE);
    g_ReSTIR_module = std::make_unique<optix_wrap::Module>(device, "path_tracer/main.ptx");
    optix_wrap::PipelineDesc pipeline_desc;
    {
        optix_wrap::ProgramDesc desc{
            .module = g_ReSTIR_module.get(),
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
            .module = g_ReSTIR_module.get(),
            .hit_group = { .ch_entry = "__closesthit__default_sphere", .intersect_module = g_sphere_module.get() },
            .shadow_grop = { .ch_entry = "__closesthit__shadow_sphere", .intersect_module = g_sphere_module.get() }
        };
        pipeline_desc.programs.push_back(desc);
    }
    device->InitPipeline(pipeline_desc);
}

void ConfigScene(device::Optix *device) {
    g_scene = std::make_unique<scene::Scene>();
    g_scene->LoadFromXML("D:/work/ReSTIR/OptixReSTIR/data/veach-ajar/test.xml");
    device->InitScene(g_scene.get());
}

void ConfigSBT(device::Optix *device) {
    optix_wrap::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program_name = "__raygen__main",
        .data = SBTTypes::RayGenDataType{}
    };
    {
        using HitGroupDataRecord = decltype(desc)::Pair<SBTTypes::HitGroupDataType>;
        for (auto &&shape : g_scene->shapes) {
            HitGroupDataRecord hit_default_data{};
            hit_default_data.program_name = "__closesthit__default";
            hit_default_data.data.mat.LoadMaterial(shape.mat);
            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program_name = "__closesthit__shadow";
            hit_shadow_data.data.mat.type = shape.mat.type;
            desc.hit_datas.push_back(hit_shadow_data);

            HitGroupDataRecord hit_default_sphere_data{};
            hit_default_sphere_data.program_name = "__closesthit__default_sphere";
            hit_default_sphere_data.data.mat.LoadMaterial(shape.mat);
            desc.hit_datas.push_back(hit_default_sphere_data);

            HitGroupDataRecord hit_shadow_sphere_data{};
            hit_shadow_sphere_data.program_name = "__closesthit__shadow_sphere",
            hit_shadow_sphere_data.data.mat.type = shape.mat.type;
            desc.hit_datas.push_back(hit_shadow_sphere_data);
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
    device->InitSBT(desc);
}

void InitLaunchParams(device::Optix *device) {
    g_params.config.frame.width = g_scene->sensor.film.w;
    g_params.config.frame.height = g_scene->sensor.film.h;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&g_params.accum_buffer),
        g_params.config.frame.height * g_params.config.frame.width * sizeof(float4)));

    g_params.frame_buffer = nullptr;
    g_params.handle = device->ias_handle;
}

void ConfigOptix(device::Optix *device) {
    ConfigPipeline(device);
    ConfigScene(device);
    ConfigSBT(device);
    InitLaunchParams(device);
}