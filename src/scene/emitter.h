#pragma once

#include "common/texture.h"

namespace scene {
enum class EEmitterType {
    Area,
    Constant
};

struct AreaEmitter {
    util::Texture radiance;
};

struct Constant {
    util::Texture radiance;
};

struct Emitter {
    EEmitterType type = EEmitterType::Constant;
    union {
        AreaEmitter area;
        Constant constant;
    };

    Emitter() noexcept {}
};
}// namespace scene