#pragma once

#include <optix.h>
#include <vector>
#include <unordered_map>
#include <string>

namespace Pupil::optix {
struct Module;

// The implementation of pipeline should be constrained in the same module,
// while builtin intersection module can be separated
struct RayTraceProgramDesc {
    Module *module_ptr = nullptr;
    const char *ray_gen_entry = nullptr;
    const char *miss_entry = nullptr;
    struct {
        const char *ch_entry = nullptr;
        const char *ah_entry = nullptr;
        Module *intersect_module = nullptr;
        const char *is_entry = nullptr;
    } hit_group;
};
struct CallableProgramDesc {
    Module *module_ptr = nullptr;
    const char *cc_entry = nullptr;
    const char *dc_entry = nullptr;
};
// TODO: user's exception programs
struct PipelineDesc {
    std::vector<RayTraceProgramDesc> ray_trace_programs;
    std::vector<CallableProgramDesc> callable_programs;

    unsigned int max_trace_depth = 2;// forward ray and shadow ray
    unsigned int max_cc_depth = 1;   // for material
    unsigned int max_dc_depth = 1;   // for ...TODO
};

// A pipeline object only contains one ray_gen_entry
struct Pipeline {
private:
    OptixPipeline m_pipeline = nullptr;
    std::vector<OptixProgramGroup> m_programs;
    std::unordered_map<std::string, OptixProgramGroup> m_program_map;

    std::string m_ray_gen_program_name;
    OptixProgramGroup m_ray_gen_program = nullptr;

public:
    static OptixPipelineCompileOptions pipeline_compile_options;

    operator OptixPipeline() const noexcept { return m_pipeline; }

    Pipeline(const PipelineDesc &desc) noexcept;
    ~Pipeline() noexcept;

    std::string_view GetRayGenProgramName() const noexcept { return m_ray_gen_program_name; }
    OptixProgramGroup GetRayGenProgram() const noexcept { return m_ray_gen_program; }
    OptixProgramGroup FindProgram(std::string) const noexcept;
};
}// namespace Pupil::optix