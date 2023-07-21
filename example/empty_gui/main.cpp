#include "system/system.h"
#include "dx12/context.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
    system->Run();
    system->Destroy();
    atexit(&Pupil::DirectX::Context::ReportLiveObjects);

    return 0;
}