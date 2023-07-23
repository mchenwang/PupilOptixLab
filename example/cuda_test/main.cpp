#include "system/system.h"
#include "pass.h"
#include "system/gui/gui.h"
#include "util/event.h"

using namespace Pupil;

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
    {
        auto pass = std::make_unique<CudaPass>();
        system->AddPass(pass.get());
        EventDispatcher<ESystemEvent::StartRendering>();
        system->Run();
    }

    system->Destroy();

    return 0;
}