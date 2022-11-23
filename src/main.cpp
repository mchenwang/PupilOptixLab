#include "static.h"

#include "gui/window.h"
#include "gui/dx12_backend.h"
#include "device/optix_device.h"
#include "device/dx12_device.h"

#include <memory>
#include <iostream>

int main() {
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Init();

    //
    {
        auto backend = gui_window->GetBackend();
        std::unique_ptr<device::Optix> optix_device =
            std::make_unique<device::Optix>(backend->GetDevice());

        auto shared_frame_resource = optix_device->CreateSharedFrameResource();
        backend->SetScreenResource(shared_frame_resource.get());

        do {
            optix_device->Run();
        } while (gui_window->Show());
    }

    gui_window->Destroy();
    return 0;
}