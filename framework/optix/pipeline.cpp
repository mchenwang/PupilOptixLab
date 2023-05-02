#include "pipeline.h"
#include "module.h"
#include "context.h"
#include "check.h"

#include "util/log.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <assert.h>

namespace Pupil::optix {

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

Pipeline::Pipeline(const PipelineDesc &desc) noexcept {
    auto ctx = Pupil::util::Singleton<optix::Context>::instance();

    // create program group
    {
        OptixProgramGroupOptions program_group_options{};
        for (auto &program : desc.ray_trace_programs) {
            if (program.module_ptr == nullptr) {
                Pupil::Log::Error("ray trace pipeline can not be created by a null module.");
                continue;
            }

            if (program.ray_gen_entry) {
                if (m_ray_gen_program) {
                    Pupil::Log::Warn("A pipeline is only allowed to contain one ray_gen_entry.\
                                      \n\t[{}] will be overwritten as [{}]",
                                     m_ray_gen_program_name, program.ray_gen_entry);
                }

                OptixProgramGroupDesc prog_desc{};
                prog_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                prog_desc.raygen.entryFunctionName = program.ray_gen_entry;
                prog_desc.raygen.module = *program.module_ptr;

                OptixProgramGroup prog = nullptr;
                OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    *ctx, &prog_desc, 1, &program_group_options,
                    LOG, &LOG_SIZE, &prog));

                m_programs.push_back(prog);
                m_program_map[program.ray_gen_entry] = prog;

                m_ray_gen_program = prog;
                m_ray_gen_program_name = program.ray_gen_entry;
            }

            if (program.miss_entry) {
                OptixProgramGroupDesc prog_desc{};
                prog_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                prog_desc.miss.entryFunctionName = program.miss_entry;
                prog_desc.miss.module = *program.module_ptr;

                OptixProgramGroup prog = nullptr;
                OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    *ctx, &prog_desc, 1, &program_group_options,
                    LOG, &LOG_SIZE, &prog));

                m_programs.push_back(prog);
                m_program_map[program.miss_entry] = prog;
            }

            // hit group
            {
                OptixProgramGroupDesc prog_desc{};
                prog_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                bool create_flag = false;

                if (program.hit_group.ah_entry) {
                    prog_desc.hitgroup.entryFunctionNameAH = program.hit_group.ah_entry;
                    prog_desc.hitgroup.moduleAH = *program.module_ptr;
                    create_flag = true;
                }
                if (program.hit_group.ch_entry) {
                    prog_desc.hitgroup.entryFunctionNameCH = program.hit_group.ch_entry;
                    prog_desc.hitgroup.moduleCH = *program.module_ptr;
                    create_flag = true;
                }

                if (program.hit_group.is_entry) {
                    prog_desc.hitgroup.entryFunctionNameIS = program.hit_group.is_entry;
                    prog_desc.hitgroup.moduleIS = *program.module_ptr;
                    create_flag = true;
                }

                if (program.hit_group.intersect_module) {
                    prog_desc.hitgroup.moduleIS = *program.hit_group.intersect_module;
                    create_flag = true;
                }

                if (create_flag) {
                    OptixProgramGroup prog = nullptr;
                    OPTIX_CHECK_LOG(optixProgramGroupCreate(
                        *ctx, &prog_desc, 1, &program_group_options,
                        LOG, &LOG_SIZE, &prog));

                    m_programs.push_back(prog);
                    if (program.hit_group.ah_entry)
                        m_program_map[program.hit_group.ah_entry] = prog;
                    if (program.hit_group.ch_entry)
                        m_program_map[program.hit_group.ch_entry] = prog;
                    if (program.hit_group.is_entry)
                        m_program_map[program.hit_group.is_entry] = prog;
                }
            }
        }

        for (auto &program : desc.callable_programs) {
            if (program.module_ptr == nullptr) {
                Pupil::Log::Error("ray trace pipeline can not be created by a null module.");
                continue;
            }

            OptixProgramGroupDesc prog_desc{};
            prog_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            if (program.cc_entry) {
                prog_desc.callables.entryFunctionNameCC = program.cc_entry;
                prog_desc.callables.moduleCC = *program.module_ptr;
            }
            if (program.dc_entry) {
                prog_desc.callables.entryFunctionNameDC = program.dc_entry;
                prog_desc.callables.moduleDC = *program.module_ptr;
            }

            OptixProgramGroup prog = nullptr;
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                *ctx, &prog_desc, 1, &program_group_options,
                LOG, &LOG_SIZE, &prog));

            m_programs.push_back(prog);
            if (program.cc_entry)
                m_program_map[program.cc_entry] = prog;
            if (program.dc_entry)
                m_program_map[program.dc_entry] = prog;
        }
    }

    // create pipeline
    {
        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = desc.max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OPTIX_CHECK_LOG(optixPipelineCreate(
            *ctx,
            &Pipeline::pipeline_compile_options,
            &pipeline_link_options,
            m_programs.data(),
            static_cast<unsigned int>(m_programs.size()),
            LOG,
            &LOG_SIZE,
            &m_pipeline));

        // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
        // parameters to optixPipelineSetStackSize.
        OptixStackSizes stack_sizes = {};
        for (auto i = 0u; i < m_programs.size(); ++i) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(m_programs[i], &stack_sizes));
        }

        unsigned int max_trace_depth = desc.max_trace_depth;
        unsigned int max_cc_depth = desc.max_cc_depth;
        unsigned int max_dc_depth = desc.max_dc_depth;
        unsigned int direct_callable_stack_size_from_traversal;
        unsigned int direct_callable_stack_size_from_state;
        unsigned int continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(
            &stack_sizes,
            max_trace_depth,
            max_cc_depth,
            max_dc_depth,
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state,
            &continuation_stack_size));

        const unsigned int max_traversal_depth = 2;
        OPTIX_CHECK(optixPipelineSetStackSize(
            m_pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size,
            max_traversal_depth));
    }
}

Pipeline::~Pipeline() noexcept {
    if (m_pipeline) OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    for (auto &p : m_programs) {
        if (p) OPTIX_CHECK(optixProgramGroupDestroy(p));
    }
}

OptixProgramGroup Pupil::optix::Pipeline::FindProgram(std::string name) const noexcept {
    if (m_program_map.find(name) == m_program_map.end())
        return nullptr;
    else
        return m_program_map.at(name);
}
}// namespace Pupil::optix