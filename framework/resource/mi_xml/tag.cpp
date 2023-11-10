#include "tag.h"

namespace Pupil::resource::mixml {
    const std::unordered_map<std::string, ETag> S_TAG_MAP{
        {"", ETag::Unknown},
        {"scene", ETag::Scene},
        {"default", ETag::Default},
        {"bsdf", ETag::Bsdf},
        {"emitter", ETag::Emitter},
        {"film", ETag::Film},
        {"integrator", ETag::Integrator},
        {"sensor", ETag::Sensor},
        {"shape", ETag::Shape},
        {"texture", ETag::Texture},
        {"lookat", ETag::Lookat},
        {"transform", ETag::Transform},
        {"integer", ETag::Integer},
        {"string", ETag::String},
        {"float", ETag::Float},
        {"vector", ETag::Vector},
        {"rgb", ETag::RGB},
        {"point", ETag::Point},
        {"matrix", ETag::Matrix},
        {"scale", ETag::Scale},
        {"rotate", ETag::Rotate},
        {"translate", ETag::Translate},
        {"boolean", ETag::Boolean},
        {"ref", ETag::Ref},
    };
}