#pragma once

#include <optix.h>
#include <vector>

namespace device {
class Optix;
}

namespace optix_wrap {
struct Module;

struct ProgramDesc {
    Module *module = nullptr;
    const char *ray_gen_entry = nullptr;
    const char *hit_miss = nullptr;
    const char *shadow_miss = nullptr;
    struct {
        const char *ch_entry = nullptr;
        const char *ah_entry = nullptr;
        Module *intersect_module = nullptr;// use for builtin type
        const char *is_entry = nullptr;
    } hit_group, shadow_grop;
};
struct PipelineDesc {
    std::vector<ProgramDesc> programs;
};

struct Pipeline {
    static OptixPipelineCompileOptions pipeline_compile_options;

    OptixPipeline pipeline;
    std::vector<OptixProgramGroup> programs;

    Pipeline(device::Optix *, const PipelineDesc &desc) noexcept;
    ~Pipeline() noexcept;
};
}// namespace optix_wrap