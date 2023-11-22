#include "system/system.h"
#include "system/event.h"
#include "pt_pass.h"
#include "static.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        system->AddPass(new Pupil::pt::PTPass());

        std::filesystem::path scene_file_path{Pupil::DATA_DIR};
        scene_file_path /= "static/default.xml";
        // scene_file_path = "D:/work/PupilRay/data/static/default.xml";
        Pupil::util::Singleton<Pupil::Event::Center>::instance()
            ->Send(Pupil::Event::RequestSceneLoad, {scene_file_path.string()});

        system->Run();
    }

    system->Destroy();

    return 0;
}