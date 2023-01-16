#include "static.h"

#include "path_tracer/type.h"

#include "gui/dx12_backend.h"
#include "gui/window.h"

#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/optix_wrap/module.h"
#include "device/optix_wrap/pipeline.h"

#include "scene/scene.h"

#include <memory>
#include <iostream>

std::unique_ptr<optix_wrap::Module> g_sphere_module;
std::unique_ptr<optix_wrap::Module> g_ReSTIR_module;

struct SBTTypes {
    using RayGenDataType = RayGenData;
    using MissDataType = MissData;
    using HitGroupDataType = HitGroupData;
};

void ConfigOptix(device::Optix *device);

int main() {
    //scene::Scene scene;
    //scene.LoadFromXML("D:/work/ReSTIR/OptixReSTIR/data/veach-ajar/test.xml");
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Init();

    //
    {
        auto backend = gui_window->GetBackend();
        std::unique_ptr<device::Optix> optix_device =
            std::make_unique<device::Optix>(backend->GetDevice());

        ConfigOptix(optix_device.get());

        backend->SetScreenResource(optix_device->GetSharedFrameResource());

        do {
            optix_device->Run();
        } while (gui_window->Show());
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
            .shadow_grop = { .ch_entry = "__closesthit__shadow_sphere", .intersect_module = g_sphere_module.get() },
        };
        pipeline_desc.programs.push_back(desc);
    }
    device->InitPipeline(pipeline_desc);
}

void ConfigScene() {
    scene::Scene scene;
    scene.LoadFromXML("D:/work/ReSTIR/OptixReSTIR/data/veach-ajar/test.xml");
}

void ConfigSBT(device::Optix *device) {
    optix_wrap::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program_name = "__raygen__main",
        .data = SBTTypes::RayGenDataType{}
    };
    {
        decltype(desc)::Pair<SBTTypes::HitGroupDataType> hit_default_data = {
            .program_name = "__closesthit__default",
            .data = SBTTypes::HitGroupDataType{}
        };
        desc.hit_datas.push_back(hit_default_data);
        decltype(desc)::Pair<SBTTypes::HitGroupDataType> hit_shadow_data = {
            .program_name = "__closesthit__shadow",
            .data = SBTTypes::HitGroupDataType{}
        };
        desc.hit_datas.push_back(hit_shadow_data);
        decltype(desc)::Pair<SBTTypes::HitGroupDataType> hit_default_sphere_data = {
            .program_name = "__closesthit__default_sphere",
            .data = SBTTypes::HitGroupDataType{}
        };
        desc.hit_datas.push_back(hit_default_sphere_data);
        decltype(desc)::Pair<SBTTypes::HitGroupDataType> hit_shadow_sphere_data = {
            .program_name = "__closesthit__shadow_sphere",
            .data = SBTTypes::HitGroupDataType{}
        };
        desc.hit_datas.push_back(hit_shadow_sphere_data);
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

void ConfigOptix(device::Optix *device) {
    ConfigPipeline(device);
    ConfigScene();
    ConfigSBT(device);
}