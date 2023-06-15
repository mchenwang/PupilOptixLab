#include "system/system.h"
#include "pt_pass.h"
#include "static.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        auto pt_pass = std::make_unique<Pupil::pt::PTPass>("Path Tracing");
        system->AddPass(pt_pass.get());
        std::filesystem::path scene_file_path{ Pupil::DATA_DIR };
        scene_file_path /= "static/default.xml";
        system->SetScene(scene_file_path);

        system->Run();
    }

    system->Destroy();

    return 0;
}