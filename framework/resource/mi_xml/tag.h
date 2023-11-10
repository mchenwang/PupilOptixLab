#pragma once

#include <string>
#include <unordered_map>

namespace Pupil::resource::mixml {
    enum class ETag : unsigned int {
        Unknown = 0,
        // xml objects
        Scene,
        Default,
        Bsdf,
        Emitter,
        Film,
        Integrator,
        Sensor,
        Shape,
        Texture,
        Lookat,
        Transform,
        // properties
        Integer,
        String,
        Float,
        Vector,
        RGB,
        Point,
        Matrix,
        Scale,
        Rotate,
        Translate,
        Boolean,
        // reference
        Ref,
        NumCount
    };

    extern const std::unordered_map<std::string, ETag> S_TAG_MAP;
}// namespace Pupil::resource::mixml