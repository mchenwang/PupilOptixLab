#include "world.h"
#include "scene/scene.h"
#include "scene/loader/mixml.h"

#include "system/system.h"
#include "system/gui/gui.h"

#include "util/event.h"
#include "util/log.h"
#include "util/timer.h"

namespace Pupil {
    struct World::Impl {
        std::unique_ptr<Scene> scene;

        void LogInformation() noexcept;
    };

    void World::Init() noexcept {
        if (m_impl) return;
        m_impl        = new Impl();
        m_impl->scene = std::make_unique<Scene>();

        EventBinder<ECanvasEvent::MouseDragging>([this](void* p) {
            // if (!util::Singleton<System>::instance()->render_flag || !camera) return;

            // const struct {
            //     float x, y;
            // } delta = *(decltype(delta) *)p;
            // float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
            // camera->Rotate(delta.x * scale, delta.y * scale);
            // EventDispatcher<EWorldEvent::CameraViewChange>();
            // EventDispatcher<EWorldEvent::CameraChange>();
        });

        EventBinder<ECanvasEvent::MouseWheel>([this](void* p) {
            // if (!util::Singleton<System>::instance()->render_flag || !camera) return;

            // float delta = *(float *)p;
            // camera->SetFovDelta(delta);
            // EventDispatcher<EWorldEvent::CameraFovChange>();
            // EventDispatcher<EWorldEvent::CameraChange>();
        });

        EventBinder<ECanvasEvent::CameraMove>([this](void* p) {
            // if (!util::Singleton<System>::instance()->render_flag || !camera) return;

            // Float3 delta = *(Float3 *)p;
            // camera->Move(delta * util::Camera::sensitivity * util::Camera::sensitivity_scale);
            // EventDispatcher<EWorldEvent::CameraMove>();
            // EventDispatcher<EWorldEvent::CameraChange>();
        });

        EventBinder<EWorldEvent::RenderInstanceTransform>([this](void* p) {
        });

        EventBinder<EWorldEvent::RenderInstanceUpdate>([this](void* p) {
        });
    }

    void World::Destroy() noexcept {
        m_impl->scene.reset();
        util::Singleton<resource::ShapeManager>::instance()->Clear();
        util::Singleton<resource::MaterialManager>::instance()->Clear();
        util::Singleton<resource::TextureManager>::instance()->Clear();

        delete m_impl;
        m_impl = nullptr;
    }

    Scene* World::GetScene() noexcept {
        return m_impl->scene.get();
    }

    bool World::LoadScene(std::filesystem::path scene_file_path) noexcept {
        if (!std::filesystem::exists(scene_file_path)) {
            Pupil::Log::Error("scene file [{}] does not exist.", scene_file_path.string());
            return false;
        }

        std::unique_ptr<SceneLoader> loader;
        if (scene_file_path.extension() == ".xml") {
            loader = std::make_unique<resource::mixml::MixmlSceneLoader>();
        } else {
            Pupil::Log::Error("unknown scene file format {}.", scene_file_path.extension().string());
            return false;
        }

        Pupil::Log::Info("start loading scene [{}].", scene_file_path.string());

        auto scene = std::make_unique<Scene>();

        Pupil::Timer timer;
        timer.Start();
        if (!loader->Load(scene_file_path, scene.get())) {
            timer.Stop();
            Pupil::Log::Error("scene load failed.");
            return false;
        }
        timer.Stop();
        Pupil::Log::Info("Time consumed for scene loading: {:.3f}s", timer.ElapsedSeconds());

        m_impl->scene.reset(scene.release());

        m_impl->LogInformation();
        util::Singleton<resource::ShapeManager>::instance()->Clear();
        util::Singleton<resource::MaterialManager>::instance()->Clear();
        util::Singleton<resource::TextureManager>::instance()->Clear();

        m_impl->scene->UploadToCuda();
        return true;
    }

    void World::Impl::LogInformation() noexcept {
        uint32_t tri_num = 0, sphere_num = 0;
        uint32_t curve_vertex_num = 0, curve_strand_num = 0;
        for (auto& ins : scene->GetInstances()) {
            if (auto tri = dynamic_cast<resource::TriangleMesh*>(ins.shape.Get());
                tri != nullptr)
                tri_num += tri->GetFaceNum();
            else if (auto sphere = dynamic_cast<resource::Sphere*>(ins.shape.Get());
                     sphere != nullptr)
                sphere_num += 1;
            else if (auto curve = dynamic_cast<resource::Curve*>(ins.shape.Get());
                     curve != nullptr)
                curve_strand_num += curve->GetStrandNum(), curve_vertex_num += curve->GetCtrlVertexNum();
        }

        uint32_t mesh_light_num = 0, sphere_light_num = 0, env_num = 0, const_env_num = 0;
        for (auto& emitter : scene->GetEmitters()) {
            if (dynamic_cast<SphereEmitter*>(emitter.get())) {
                ++sphere_light_num;
            } else if (dynamic_cast<TriMeshEmitter*>(emitter.get())) {
                ++mesh_light_num;
            } else if (dynamic_cast<EnvmapEmitter*>(emitter.get())) {
                ++env_num;
            } else if (dynamic_cast<ConstEmitter*>(emitter.get())) {
                ++const_env_num;
            }
        }

        Pupil::Log::Info("scene structure: ");
        Pupil::Log::Info("  objects: ");
        if (tri_num > 0) Pupil::Log::Info("    triangles: {}", tri_num);
        if (sphere_num > 0) Pupil::Log::Info("    spheres: {}", sphere_num);
        if (curve_strand_num > 0) Pupil::Log::Info("    curve strands: {}", curve_strand_num);
        if (curve_vertex_num > 0) Pupil::Log::Info("    curve control vertices: {}", curve_vertex_num);
        Pupil::Log::Info("  lights: ");
        if (mesh_light_num > 0) Pupil::Log::Info("    mesh lights: {}", mesh_light_num);
        if (sphere_light_num > 0) Pupil::Log::Info("    sphere lights: {}", sphere_light_num);
        if (env_num || const_env_num) Pupil::Log::Info("    environment light: {}", env_num ? "envmap" : "const");
        if (env_num + const_env_num > 1) Pupil::Log::Warn("multiple environment lights");
    }

}// namespace Pupil