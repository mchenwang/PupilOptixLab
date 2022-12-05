#pragma once

namespace scene {
struct Integrator {
    int max_depth = 0;
};

struct Transform {
    float matrix[12]{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
};

struct Film {
    int w = 1;
    int h = 1;
};

struct Sensor {
    float fov = 90.f;
    Transform transform{};
    Film film{};
};

struct Scene {
    Integrator integrator;
    Sensor sensor;
};
}// namespace scene