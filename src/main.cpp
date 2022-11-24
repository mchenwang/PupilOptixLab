#include "static.h"

#include "gui/window.h"
#include "gui/dx12_backend.h"
#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/optix_wrap/module.h"

#include <memory>
#include <iostream>

std::unique_ptr<optix_wrap::Module> g_sphere_module;
std::unique_ptr<optix_wrap::Module> g_ReSTIR_module;
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

        auto shared_frame_resource = optix_device->CreateSharedFrameResource();
        backend->SetScreenResource(shared_frame_resource.get());

        do {
            optix_device->Run();
        } while (gui_window->Show());
    }
    g_ReSTIR_module.reset();
    g_sphere_module.reset();
    gui_window->Destroy();
    return 0;
}

void ConfigOptix(device::Optix* device) {
    g_sphere_module = std::make_unique<optix_wrap::Module>(device, OPTIX_PRIMITIVE_TYPE_SPHERE);
    g_ReSTIR_module = std::make_unique<optix_wrap::Module>(device, "path_tracer/main.cu");
    optix_wrap::PipelineDesc pipeline_desc;
    {
        optix_wrap::ProgramDesc desc{
            .module = g_ReSTIR_module.get(),
            .ray_gen_entry = "__raygen__main",
            .hit_miss = "__miss__ray",
            .shadow_miss = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__ray" },
            .shadow_grop = { .ch_entry = "__closesthit__shadow" }
        };
        pipeline_desc.programs.push_back(desc);
    }

    {
        optix_wrap::ProgramDesc desc{
            .module = g_ReSTIR_module.get(),
            .hit_group = { .ch_entry = "__closesthit__ray_sphere", .intersect_module = g_sphere_module.get() },
            .shadow_grop = { .ch_entry = "__closesthit__shadow", .intersect_module = g_sphere_module.get() },
        };
        pipeline_desc.programs.push_back(desc);
    }
    device->InitPipeline(pipeline_desc);
}