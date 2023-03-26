#include "system/system.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
    system->Run();
    system->Destroy();

    return 0;
}