#include "system/system.h"
#include "pass.h"
#include "system/gui.h"
#include "util/event.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
    {
        auto pass = std::make_unique<CudaPass>();
        system->AddPass(pass.get());

        struct {
            uint32_t w, h;
        } size{ 512, 512 };
        Pupil::EventDispatcher<Pupil::ECanvasEvent::Resize>(size);

        system->Run();
    }

    system->Destroy();

    return 0;
}