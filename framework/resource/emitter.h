#pragma once

#include "util/texture.h"
#include "util/type.h"

namespace Pupil::resource {
enum class EEmitterType {
    Unknown,
    Area,
    Point,
    ConstEnv,
    EnvMap,
    DistDir
};

struct AreaEmitter {
    util::Texture radiance;
};

struct PointEmitter {
    util::Float3 pos;
    util::Float3 intensity;
};

struct ConstEnv {
    util::Float3 radiance;
};

struct EnvMap {
    util::Texture radiance;
    float scale;

    util::Transform transform{};
};

struct DirectionalEmitter {
    util::Float3 irradiance;
    util::Float3 dir;
};

struct Emitter {
    EEmitterType type = EEmitterType::Unknown;
    union {
        AreaEmitter area;
        PointEmitter point;
        ConstEnv const_env;
        EnvMap env_map;
        DirectionalEmitter dir;
    };

    Emitter() noexcept {}
};
}// namespace Pupil::resource