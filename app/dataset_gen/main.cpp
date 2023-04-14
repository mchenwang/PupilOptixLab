#include "system/system.h"
#include "util/event.h"
#include "util/texture.h"
#include "pt_pass.h"
#include "static.h"
int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(false);

    {
        auto pt_pass = std::make_unique<Pupil::pt::PTPass>("Path Tracing");
        system->AddPass(pt_pass.get());
        std::filesystem::path scene_file_path{ Pupil::DATA_DIR };

        std::vector<std::string> scenes_name{
            "bathroom0",
            "bedroom0"
        };
        int i = 0, j = 0, k = 0;
        auto scene_root = scene_file_path / "scene";
        scene_file_path = scene_root / scenes_name[i] / (scenes_name[i] + "_" + std::to_string(j) + "_" + std::to_string(k) + ".xml");
        system->SetScene(scene_file_path);

        Pupil::EventBinder<Pupil::ESystemEvent::FrameFinished>([&](void *) {
            auto result = Pupil::util::Singleton<Pupil::BufferManager>::instance()->GetBuffer("result");

            std::filesystem::path path = scene_root / scenes_name[i] / "result" / (scenes_name[i] + "_" + std::to_string(j) + "_" + std::to_string(k) + ".exr");
            size_t size = 1920 * 1080 * 4;
            auto image = new float[size];
            memset(image, 0, size);
            Pupil::cuda::CudaMemcpyToHost(image, result->cuda_res.ptr, size * sizeof(float));
            Pupil::util::BitmapTexture::Save(image, 1920, 1080, path.string().c_str(), Pupil::util::BitmapTexture::FileFormat::EXR);
            delete[] image;

            ++k;
            if (k == 5) {
                k = 0;
                ++j;
                if (j == 2) {
                    j = 0;
                    ++i;
                    if (i >= scenes_name.size()) {
                        Pupil::EventDispatcher<Pupil::ESystemEvent::Quit>();
                        return;
                    }
                }
            }
            scene_file_path = scene_root / scenes_name[i] / (scenes_name[i] + "_" + std::to_string(j) + "_" + std::to_string(k) + ".xml");

            system->SetScene(scene_file_path);
        });

        system->Run();
    }

    system->Destroy();

    return 0;
}