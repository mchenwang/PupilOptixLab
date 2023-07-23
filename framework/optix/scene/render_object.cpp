#include "render_object.h"
#include "ias_manager.h"
#include "cuda/util.h"

#include "util/event.h"
#include "system/world.h"

namespace Pupil::optix {
RenderObject::RenderObject(EMeshEntityType type, void *mesh, util::Transform transform, std::string_view id, unsigned int v_mask) noexcept
    : id(id), gas_handle(0), visibility_mask(v_mask), transform(transform) {
    auto mesh_mngr = util::Singleton<MeshManager>::instance();
    gas_handle = mesh_mngr->GetGASHandle(type, mesh);
}

void RenderObject::UpdateTransform(const util::Transform &new_transform) noexcept {
    transform = new_transform;
    EventDispatcher<EWorldEvent::RenderObjectTransform>(this);
}

void RenderObject::ApplyTransform(const util::Transform &new_transform) noexcept {
    transform.matrix = new_transform.matrix * transform.matrix;
    EventDispatcher<EWorldEvent::RenderObjectTransform>(this);
}
}// namespace Pupil::optix