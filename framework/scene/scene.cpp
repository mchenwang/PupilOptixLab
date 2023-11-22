#include "scene.h"
#include "ias.h"
#include "gas.h"

#include "cuda/stream.h"
#include "cuda/check.h"

#include <unordered_map>

namespace Pupil {
    struct Scene::Impl {
        uint32_t object_id = 0;

        bool        emitter_dirty = false;
        int         env_index     = -1;
        CUdeviceptr emitter_group = 0;

        std::vector<optix::Emitter>                optix_emitters;
        std::unordered_map<const Emitter*, size_t> emitter_index;

        Camera camera;

        bool       ias_dirty = true;
        IASManager ias_mngr;
    };

    Scene::Scene() noexcept {
        m_impl = new Impl();
    }

    Scene::~Scene() noexcept {
        delete m_impl;
        m_instances.clear();
    }

    void Scene::Reset() noexcept {
        m_impl->ias_mngr.Clear();
        m_impl->object_id     = 0;
        m_impl->emitter_dirty = true;
        m_impl->env_index     = -1;
        m_impl->optix_emitters.clear();
        m_impl->emitter_index.clear();
    }

    void Scene::SetCamera(const CameraDesc& desc) noexcept {
        m_impl->camera.SetProjectionFactor(desc.fov_y, desc.aspect_ratio, desc.near_clip, desc.far_clip);
        m_impl->camera.SetWorldTransform(desc.to_world);
    }

    Camera& Scene::GetCamera() const noexcept {
        return m_impl->camera;
    }

    void Scene::AddInstance(std::string_view                              name,
                            const util::CountableRef<resource::Shape>&    shape,
                            const Transform&                              transform,
                            const util::CountableRef<resource::Material>& material,
                            const resource::TextureInstance&              emit_radiance) noexcept {
        Instance instance;
        instance.name            = name.empty() ? "object " + std::to_string(m_impl->object_id++) : name;
        instance.visibility_mask = 1;
        instance.shape           = shape;
        // instance.gas = util::Singleton<GASManager>::instance()->GetGAS(shape);
        instance.material  = material;
        instance.transform = transform;
        instance.aabb      = util::AABB{};
        auto aabb_ls       = shape->GetAABB();
        instance.aabb.Merge(transform * aabb_ls.min);
        instance.aabb.Merge(transform * aabb_ls.max);

        instance.emitter = nullptr;
        if (emit_radiance) {
            if (dynamic_cast<resource::Sphere*>(shape.Get())) {
                util::CountableRef<resource::Sphere> sphere = shape;

                auto emitter     = std::make_unique<SphereEmitter>(sphere, transform, emit_radiance);
                instance.emitter = emitter.get();
                AddEmitter(std::move(emitter));
            } else if (dynamic_cast<resource::TriangleMesh*>(shape.Get())) {
                util::CountableRef<resource::TriangleMesh> mesh = shape;

                auto emitter     = std::make_unique<TriMeshEmitter>(mesh, transform, emit_radiance);
                instance.emitter = emitter.get();
                AddEmitter(std::move(emitter));
            }
        }

        m_instances.emplace_back(instance);
        m_impl->ias_dirty = true;
    }

    void Scene::AddEmitter(std::unique_ptr<Emitter>&& emitter) noexcept {
        if (emitter->IsEnvironmentEmitter())
            m_impl->env_index = static_cast<int>(m_emitters.size());
        // m_impl->optix_emitters.emplace_back(emitter->GetOptixEmitter());

        m_impl->emitter_index[emitter.get()] = m_emitters.size();
        m_emitters.emplace_back(std::move(emitter));
        m_impl->emitter_dirty = true;
    }

    int Scene::GetEmitterIndex(const Emitter* emitter) const noexcept {
        if (auto it = m_impl->emitter_index.find(emitter);
            it != m_impl->emitter_index.end())
            return static_cast<int>(it->second);
        return -1;
    }

    optix::EmitterGroup Scene::GetOptixEmitters() noexcept {
        optix::EmitterGroup emit_gp;
        emit_gp.emitters.SetData(m_impl->emitter_group, m_impl->optix_emitters.size());
        emit_gp.env_index = m_impl->env_index;

        return emit_gp;
    }

    void Scene::UploadToCuda() noexcept {
        for (auto& ins : m_instances) {
            ins.shape->UploadToCuda();
        }

        for (auto& ins : m_instances) {
            ins.material->UploadToCuda();
        }

        auto gas_mngr = util::Singleton<GASManager>::instance();
        for (auto& ins : m_instances) {
            ins.gas = gas_mngr->GetGAS(ins.shape);
        }

        if (m_impl->emitter_dirty) {
            m_impl->optix_emitters.clear();
            for (auto& emitter : m_emitters) {
                emitter->UploadToCuda();
                optix::Emitter optix_emitter     = emitter->GetOptixEmitter();
                optix_emitter.select_probability = 1.f / m_emitters.size();
                m_impl->optix_emitters.push_back(optix_emitter);
            }

            auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::EmitterUploading);
            CUDA_FREE_ASYNC(m_impl->emitter_group, *stream);
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->emitter_group),
                                       sizeof(optix::Emitter) * m_impl->optix_emitters.size(),
                                       *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_impl->emitter_group),
                                       m_impl->optix_emitters.data(),
                                       sizeof(optix::Emitter) * m_impl->optix_emitters.size(),
                                       cudaMemcpyHostToDevice,
                                       *stream));
            m_impl->emitter_dirty = false;
        }

        if (m_impl->ias_dirty) {
            m_impl->ias_mngr.SetInstance(m_instances);
            m_impl->ias_dirty = false;
        }
    }

    OptixTraversableHandle Scene::GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept {
        util::Singleton<cuda::StreamManager>::instance()->Synchronize(cuda::EStreamTaskType::GASCreation);
        return m_impl->ias_mngr.GetIASHandle(gas_offset, allow_update);
    }
}// namespace Pupil
