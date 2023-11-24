#include "system/system.h"
#include "system/event.h"
#include "system/denoise_pass.h"
#include "pt_pass.h"
#include "static.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        system->AddPass(new Pupil::pt::PTPass());

        Pupil::DenoisePass::Config denoise_config{
            .default_enable = true,
            .noise_name     = "pt result",
            .use_albedo     = true,
            .albedo_name    = "albedo",
            .use_normal     = true,
            .normal_name    = "normal"};
        system->AddPass(new Pupil::DenoisePass(denoise_config));

        std::filesystem::path scene_file_path{Pupil::DATA_DIR};
        scene_file_path /= "static/default.xml";

        Pupil::util::Singleton<Pupil::Event::Center>::instance()
            ->Send(Pupil::Event::RequestSceneLoad, {scene_file_path.string()});

        system->Run();
    }

    system->Destroy();

    return 0;
}