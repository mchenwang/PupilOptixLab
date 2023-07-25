#include "render_object.h"
#include "ias_manager.h"
#include "cuda/util.h"

#include "util/event.h"
#include "world.h"

namespace Pupil::world {

RenderObject::RenderObject(const resource::Shape *shape, const util::Transform &trans, std::string_view id, unsigned int v_mask) noexcept
    : id(id), transform(trans), visibility_mask(v_mask) {
    gas = util::Singleton<GASManager>::instance()->GetGASHandle(shape);
}

RenderObject::RenderObject(std::string_view shape_id, const util::Transform &trans, std::string_view id, unsigned int v_mask) noexcept
    : id(id), transform(trans), visibility_mask(v_mask) {
    gas = util::Singleton<GASManager>::instance()->GetGASHandle(shape_id);
}

void RenderObject::UpdateTransform(const util::Transform &new_transform) noexcept {
    transform = new_transform;
    EventDispatcher<EWorldEvent::RenderInstanceUpdate>(this);
}

void RenderObject::ApplyTransform(const util::Transform &new_transform) noexcept {
    transform.matrix = new_transform.matrix * transform.matrix;
    EventDispatcher<EWorldEvent::RenderInstanceUpdate>(this);
}
}// namespace Pupil::world