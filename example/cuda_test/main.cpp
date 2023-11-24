#include "system/system.h"
#include "system/event.h"
#include "pass.h"

using namespace Pupil;

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
    {
        system->AddPass(new CudaPass());

        util::Singleton<Pupil::Event::Center>::instance()
            ->Send(Pupil::Event::LimitRenderRate, {60});
        Pupil::util::Singleton<Pupil::Event::Center>::instance()
            ->Send(Pupil::Event::RenderRestart);
        system->Run();
    }

    system->Destroy();

    return 0;
}