#pragma once

#include "util/texture.h"
#include "util/type.h"

namespace Pupil::scene {
enum class EEmitterType {
    Unknown,
    Area,
    ConstEnv,
    EnvMap
};

struct AreaEmitter {
    util::Texture radiance;
};

struct ConstEnv {
    util::Float3 radiance;
};

struct EnvMap {
    util::Texture radiance;
};

struct Emitter {
    EEmitterType type = EEmitterType::Unknown;
    union {
        AreaEmitter area;
        ConstEnv const_env;
        EnvMap env_map;
    };

    Emitter() noexcept {}
};
}// namespace Pupil::scene