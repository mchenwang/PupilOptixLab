#pragma once

#include "resource/shape.h"
#include "resource/texture.h"
#include "resource/material.h"
#include "emitter.h"
#include "camera.h"
#include "gas.h"

#include <vector>
#include <filesystem>

namespace Pupil {
    struct Instance {
        std::string  name;
        unsigned int visibility_mask;

        util::CountableRef<resource::Shape>    shape;
        util::CountableRef<GAS>                gas;
        util::CountableRef<resource::Material> material;
        Emitter*                               emitter;

        Transform  transform;
        util::AABB aabb;
    };

    class Scene {
    public:
        unsigned int film_w;
        unsigned int film_h;
        unsigned int max_depth;

        Scene() noexcept;
        ~Scene() noexcept;

        Scene(const Scene&)            = delete;
        Scene& operator=(const Scene&) = delete;

        void Reset() noexcept;

        void    SetCamera(const CameraDesc&) noexcept;
        Camera& GetCamera() const noexcept;

        void AddInstance(std::string_view                              name,
                         const util::CountableRef<resource::Shape>&    shape,
                         const Transform&                              transform,
                         const util::CountableRef<resource::Material>& material,
                         const resource::TextureInstance&              emit_radiance) noexcept;
        void AddEmitter(std::unique_ptr<Emitter>&& emitter) noexcept;

        void UploadToCuda() noexcept;

        auto& GetInstances() const noexcept { return m_instances; }
        auto& GetEmitters() const noexcept { return m_emitters; }
        int   GetEmitterIndex(const Emitter*) const noexcept;

        optix::EmitterGroup GetOptixEmitters() noexcept;

        OptixTraversableHandle GetIASHandle(unsigned int gas_offset = 2, bool allow_update = false) noexcept;

    protected:
        std::vector<Instance>                 m_instances;
        std::vector<std::unique_ptr<Emitter>> m_emitters;

        struct Impl;
        Impl* m_impl = nullptr;
    };

    class SceneLoader {
    public:
        virtual bool Load(std::filesystem::path path, Scene*) noexcept = 0;

    protected:
        virtual bool Visit(void* obj, Scene*) noexcept = 0;
    };
}// namespace Pupil