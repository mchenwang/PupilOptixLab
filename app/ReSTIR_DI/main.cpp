#include "system/system.h"
#include "static.h"

#include "GBufferPass/pass.h"
#include "TemporalReusePass/pass.h"
#include "SpatialReusePass/pass.h"
#include "ShadowRayPass/pass.h"
#include "ShadingPass/pass.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        auto gbuffer_pass = std::make_unique<GBufferPass>();
        auto temp_pass = std::make_unique<TemporalReusePass>();
        auto spat_pass = std::make_unique<SpatialReusePass>();
        auto shadow_pass = std::make_unique<ShadowRayPass>();
        auto shading_pass = std::make_unique<ShadingPass>();

        system->AddPass(gbuffer_pass.get());
        system->AddPass(temp_pass.get());
        system->AddPass(spat_pass.get());
        system->AddPass(shadow_pass.get());
        system->AddPass(shading_pass.get());

        std::filesystem::path scene_file_path{ Pupil::DATA_DIR };
        scene_file_path /= "restir_test.xml";
        system->SetScene(scene_file_path);

        system->Run();
    }

    system->Destroy();

    return 0;
}