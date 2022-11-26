#include "pipeline.h"
#include "module.h"
#include "../optix_device.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <assert.h>

using namespace optix_wrap;

OptixPipelineCompileOptions Pipeline::pipeline_compile_options = {
    .usesMotionBlur = false,
    .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
    .numPayloadValues = 2,
    .numAttributeValues = 2,
    .exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW,
    .pipelineLaunchParamsVariableName = "optix_launch_params",
    .usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE) |
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE)
};

Pipeline::Pipeline(device::Optix *device, const PipelineDesc &desc) noexcept {
    // create program group
    {
        OptixProgramGroupOptions program_group_options{};

        auto CreateProgramGroup = [&](OptixProgramGroupKind kind, Module *module, const char *entry) {
            if (entry) {
                OptixProgramGroupDesc desc{};
                desc.kind = kind;
                if (kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN) {
                    desc.raygen.module = module->module;
                    desc.raygen.entryFunctionName = entry;
                } else if (kind == OPTIX_PROGRAM_GROUP_KIND_MISS) {
                    desc.miss.module = module->module;
                    desc.miss.entryFunctionName = entry;
                }

                OptixProgramGroup p = nullptr;
                OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    device->context, &desc, 1, &program_group_options,
                    LOG, &LOG_SIZE, &p));

                programs.push_back(p);
                m_program_map[entry] = p;
            }
        };

        auto CreateHitGroup = [&](Module *module, decltype(ProgramDesc::hit_group) hit_group) {
            OptixProgramGroupDesc desc{};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            bool create_flag = false;

            if (hit_group.ah_entry) {
                desc.hitgroup.entryFunctionNameAH = hit_group.ah_entry;
                desc.hitgroup.moduleAH = module->module;
                create_flag = true;
            }
            if (hit_group.ch_entry) {
                desc.hitgroup.entryFunctionNameCH = hit_group.ch_entry;
                desc.hitgroup.moduleCH = module->module;
                create_flag = true;
            } else {
                assert("empty entry cannot be used as the program id" && false);
            }
            if (hit_group.is_entry) {
                desc.hitgroup.entryFunctionNameIS = hit_group.is_entry;
                create_flag = true;
                if (hit_group.intersect_module)
                    desc.hitgroup.moduleIS = hit_group.intersect_module->module;
                else
                    desc.hitgroup.moduleIS = module->module;
            }

            if (create_flag) {
                OptixProgramGroup p = nullptr;
                OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    device->context, &desc, 1, &program_group_options,
                    LOG, &LOG_SIZE, &p));
                programs.push_back(p);
                m_program_map[hit_group.ch_entry] = p;
            }
        };

        for (auto &program_desc : desc.programs) {
            assert(program_desc.module);
            CreateProgramGroup(OPTIX_PROGRAM_GROUP_KIND_RAYGEN, program_desc.module, program_desc.ray_gen_entry);
            CreateProgramGroup(OPTIX_PROGRAM_GROUP_KIND_MISS, program_desc.module, program_desc.hit_miss);
            CreateProgramGroup(OPTIX_PROGRAM_GROUP_KIND_MISS, program_desc.module, program_desc.shadow_miss);

            CreateHitGroup(program_desc.module, program_desc.hit_group);
            CreateHitGroup(program_desc.module, program_desc.shadow_grop);
        }
    }

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 20;// TODO
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        device->context,
        &Pipeline::pipeline_compile_options,
        &pipeline_link_options,
        programs.data(),
        (uint32_t)programs.size(),
        LOG,
        &LOG_SIZE,
        &pipeline));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    for (auto i = 0u; i < programs.size(); ++i) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(programs[i], &stack_sizes));
    }

    uint32_t max_trace_depth = 20;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size));

    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth));
}

Pipeline::~Pipeline() noexcept {
    if (pipeline) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    for (auto &p : programs) {
        if (p) OPTIX_CHECK(optixProgramGroupDestroy(p));
    }
}

OptixProgramGroup optix_wrap::Pipeline::FindProgram(std::string name) noexcept {
    if (m_program_map.find(name) == m_program_map.end())
        return nullptr;
    else
        return m_program_map[name];
}
