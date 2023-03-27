#include "system/system.h"
#include "pt_pass.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        auto pt_pass = std::make_unique<Pupil::pt::PTPass>("Path Tracing");
        system->AddPass(pt_pass.get());

        system->Run();
    }

    system->Destroy();

    return 0;
}