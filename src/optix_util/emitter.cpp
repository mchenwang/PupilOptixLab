#include "emitter.h"
#include "scene/scene.h"

namespace {
void SplitMesh(std::vector<optix_util::Emitter> &emitters,
               uint32_t vertex_num, uint32_t face_num, uint32_t *indices,
               const float *positions, const float *normals, const float *texcoords,
               const util::Transform &transform) noexcept {
    for (auto i = 0u; i < face_num; ++i) {
        optix_util::Emitter emitter;
        emitter.type = optix_util::EEmitterType::Triangle;

        auto idx0 = indices[i * 3 + 0];
        auto idx1 = indices[i * 3 + 1];
        auto idx2 = indices[i * 3 + 2];
    }
}
}// namespace

namespace optix_util {
std::vector<Emitter> GenerateEmitters(const scene::Scene *scene) noexcept {
    std::vector<Emitter> emitters;
    for (auto &&shape : scene->shapes) {
        if (!shape.is_emitter) continue;

        switch (shape.type) {
            case scene::EShapeType::_cube: {
            }
            case scene::EShapeType::_obj:
            case scene::EShapeType::_rectangle:
            case scene::EShapeType::_sphere:
                break;
        }
    }
}
}// namespace optix_util