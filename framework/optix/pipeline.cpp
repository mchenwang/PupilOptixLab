#include "pipeline.h"
#include "context.h"
#include "check.h"

#include "util/hash.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <vector>
#include <unordered_map>

namespace Pupil::optix {
    struct Pipeline::Impl {
        OptixPipeline               pipeline = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options;
        unsigned int                max_trace_depth = 2;
        unsigned int                max_dc_depth    = 0;
        unsigned int                max_cc_depth    = 0;

        std::unordered_map<EModuleType, std::unique_ptr<Module>> builtin_modules;
        std::unordered_map<const char*, std::unique_ptr<Module>> embedded_modules;

        std::vector<std::unique_ptr<Program>> programs;

        std::unordered_map<std::string, Program*, util::StringHash, std::equal_to<>> program_map;
    };

    Pipeline::Pipeline() noexcept {
        m_impl                                                            = new Impl();
        m_impl->pipeline_compile_options.usesMotionBlur                   = false;
        m_impl->pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        m_impl->pipeline_compile_options.numPayloadValues                 = 2;
        m_impl->pipeline_compile_options.numAttributeValues               = 0;
        m_impl->pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        m_impl->pipeline_compile_options.pipelineLaunchParamsVariableName = "optix_launch_params";
        m_impl->pipeline_compile_options.usesPrimitiveTypeFlags           = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    }

    Pipeline::~Pipeline() noexcept {
        if (m_impl->pipeline) OPTIX_CHECK(optixPipelineDestroy(m_impl->pipeline));
        m_impl->programs.clear();
        delete m_impl;
        m_impl = nullptr;
    }

    Pipeline::operator OptixPipeline() const noexcept {
        return m_impl->pipeline;
    }

    void Pipeline::Finish() noexcept {
        auto ctx = Pupil::util::Singleton<optix::Context>::instance();

        std::vector<OptixProgramGroup> progs(m_impl->programs.size());
        // create program groups
        {
            OptixProgramGroupOptions           program_group_options{};
            std::vector<OptixProgramGroupDesc> progs_desc(m_impl->programs.size());
            for (int i = 0; i < m_impl->programs.size(); i++)
                progs_desc[i] = m_impl->programs[i]->GetOptixProgramGroupDesc();

            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                *ctx, progs_desc.data(),
                static_cast<unsigned int>(m_impl->programs.size()),
                &program_group_options,
                LOG, &LOG_SIZE, progs.data()));

            for (int i = 0; i < m_impl->programs.size(); i++)
                m_impl->programs[i]->Set(progs[i]);
        }

        // create pipeline
        {
            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth            = m_impl->max_trace_depth;

            switch (ctx->GetDebugLevel()) {
                case optix::EDebugLevel::None:
                    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
                    break;
                case optix::EDebugLevel::Minimal:
                    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
                    break;
                case optix::EDebugLevel::Full:
                    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
                    break;
            }

            OPTIX_CHECK_LOG(optixPipelineCreate(
                *ctx,
                &m_impl->pipeline_compile_options,
                &pipeline_link_options,
                progs.data(),
                static_cast<unsigned int>(progs.size()),
                LOG,
                &LOG_SIZE,
                &m_impl->pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto i = 0u; i < progs.size(); ++i) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(progs[i], &stack_sizes));
            }

            unsigned int max_trace_depth = m_impl->max_trace_depth;
            unsigned int max_cc_depth    = m_impl->max_cc_depth;
            unsigned int max_dc_depth    = m_impl->max_dc_depth;
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
                m_impl->pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth));
        }
    }

    Pipeline& Pipeline::SetMaxTraceDepth(unsigned int depth) noexcept {
        m_impl->max_trace_depth = depth;
        return *this;
    }

    Pipeline& Pipeline::SetMaxCCDepth(unsigned int depth) noexcept {
        m_impl->max_cc_depth = depth;
        return *this;
    }

    Pipeline& Pipeline::SetMaxDCDepth(unsigned int depth) noexcept {
        m_impl->max_dc_depth = depth;
        return *this;
    }

    Pipeline& Pipeline::EnableMotionBlur(bool enable) noexcept {
        m_impl->pipeline_compile_options.usesMotionBlur = enable;
        return *this;
    }

    Pipeline& Pipeline::SetTraversableGraphFlags(OptixTraversableGraphFlags flags) noexcept {
        m_impl->pipeline_compile_options.traversableGraphFlags = flags;
        return *this;
    }

    Pipeline& Pipeline::SetNumPayloadValues(unsigned int num) noexcept {
        m_impl->pipeline_compile_options.numPayloadValues = num;
        return *this;
    }

    Pipeline& Pipeline::SetNumAttributeValues(unsigned int num) noexcept {
        m_impl->pipeline_compile_options.numAttributeValues = num;
        return *this;
    }

    Pipeline& Pipeline::SetExceptionFlags(OptixExceptionFlags flags) noexcept {
        m_impl->pipeline_compile_options.exceptionFlags = flags;
        return *this;
    }

    Pipeline& Pipeline::SetPipelineLaunchParamsVariableName(const char* name) noexcept {
        m_impl->pipeline_compile_options.pipelineLaunchParamsVariableName = name;
        return *this;
    }

    Pipeline& Pipeline::EnalbePrimitiveType(EPrimitiveType type) noexcept {
        m_impl->pipeline_compile_options.usesPrimitiveTypeFlags |= static_cast<unsigned int>(type);
        return *this;
    }

    OptixPipelineCompileOptions Pipeline::GetCompileOptions() const noexcept {
        return m_impl->pipeline_compile_options;
    }

    Module::Module(Pipeline* pipeline, OptixPrimitiveType type) noexcept {
        auto ctx = Pupil::util::Singleton<optix::Context>::instance();

        OptixModuleCompileOptions module_compile_options{};
        switch (ctx->GetDebugLevel()) {
            case optix::EDebugLevel::None:
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
                break;
            case optix::EDebugLevel::Minimal:
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
                break;
            case optix::EDebugLevel::Full:
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
                break;
        }

        OptixBuiltinISOptions options{.builtinISModuleType = type};
        auto                  pipline_compile_options = pipeline->GetCompileOptions();

        OPTIX_CHECK_LOG(optixBuiltinISModuleGet(
            *ctx,
            &module_compile_options,
            &pipline_compile_options,
            &options,
            &m_module));
    }

    Module::Module(Pipeline* pipeline, std::string_view embedded_ptx_code) noexcept {
        auto ctx = Pupil::util::Singleton<optix::Context>::instance();

        OptixModuleCompileOptions module_compile_options{};
        switch (ctx->GetDebugLevel()) {
            case optix::EDebugLevel::None:
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
                break;
            case optix::EDebugLevel::Minimal:
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
                break;
            case optix::EDebugLevel::Full:
                module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
                module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
                break;
        }

        auto pipline_compile_options = pipeline->GetCompileOptions();

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            *ctx,
            &module_compile_options,
            &pipline_compile_options,
            embedded_ptx_code.data(),
            embedded_ptx_code.size(),
            LOG,
            &LOG_SIZE,
            &m_module));
    }

    Module::~Module() noexcept {
        if (m_module) OPTIX_CHECK(optixModuleDestroy(m_module));
    }

    Module* Pipeline::CreateModule(EModuleType type, std::string_view embedded_ptx_code) noexcept {
        switch (type) {
            case EModuleType::UserDefined:
                m_impl->embedded_modules.emplace(embedded_ptx_code.data(), std::make_unique<Module>(this, embedded_ptx_code));
                return m_impl->embedded_modules.at(embedded_ptx_code.data()).get();
            case EModuleType::BuiltinSphereIS:
                m_impl->builtin_modules.emplace(type, std::make_unique<Module>(this, OPTIX_PRIMITIVE_TYPE_SPHERE));
                return m_impl->builtin_modules.at(type).get();
            case EModuleType::BuiltinCurveQuadraticIS:
                m_impl->builtin_modules.emplace(type, std::make_unique<Module>(this, OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE));
                return m_impl->builtin_modules.at(type).get();
            case EModuleType::BuiltinCurveCubicIS:
                m_impl->builtin_modules.emplace(type, std::make_unique<Module>(this, OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE));
                return m_impl->builtin_modules.at(type).get();
            case EModuleType::BuiltinCurveLinearIS:
                m_impl->builtin_modules.emplace(type, std::make_unique<Module>(this, OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR));
                return m_impl->builtin_modules.at(type).get();
            case EModuleType::BuiltinCurveCatmullromIS:
                m_impl->builtin_modules.emplace(type, std::make_unique<Module>(this, OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM));
                return m_impl->builtin_modules.at(type).get();
            case EModuleType::BuiltinCustomIS:
                m_impl->builtin_modules.emplace(type, std::make_unique<Module>(this, OPTIX_PRIMITIVE_TYPE_CUSTOM));
                return m_impl->builtin_modules.at(type).get();
        }
        return nullptr;
    }

    RayGenProgram& Pipeline::CreateRayGen(std::string_view name) noexcept {
        m_impl->programs.emplace_back(std::make_unique<RayGenProgram>());
        if (!name.empty()) m_impl->program_map.emplace(name, m_impl->programs.back().get());
        return *static_cast<RayGenProgram*>(m_impl->programs.back().get());
    }

    MissProgram& Pipeline::CreateMiss(std::string_view name) noexcept {
        m_impl->programs.emplace_back(std::make_unique<MissProgram>());
        if (!name.empty()) m_impl->program_map.emplace(name, m_impl->programs.back().get());
        return *static_cast<MissProgram*>(m_impl->programs.back().get());
    }

    HitgroupProgram& Pipeline::CreateHitgroup(std::string_view name) noexcept {
        m_impl->programs.emplace_back(std::make_unique<HitgroupProgram>());
        if (!name.empty()) m_impl->program_map.emplace(name, m_impl->programs.back().get());
        return *static_cast<HitgroupProgram*>(m_impl->programs.back().get());
    }

    CallableProgram& Pipeline::CreateCallable(std::string_view name) noexcept {
        m_impl->programs.emplace_back(std::make_unique<CallableProgram>());
        if (!name.empty()) m_impl->program_map.emplace(name, m_impl->programs.back().get());
        return *static_cast<CallableProgram*>(m_impl->programs.back().get());
    }

    ExceptionProgram& Pipeline::CreateException(std::string_view name) noexcept {
        m_impl->programs.emplace_back(std::make_unique<ExceptionProgram>());
        if (!name.empty()) m_impl->program_map.emplace(name, m_impl->programs.back().get());
        return *static_cast<ExceptionProgram*>(m_impl->programs.back().get());
    }

    Program* Pipeline::FindProgram(std::string_view name) const noexcept {
        if (auto it = m_impl->program_map.find(name);
            it != m_impl->program_map.end())
            return it->second;
        return nullptr;
    }

    Program::~Program() noexcept {
        if (m_program) OPTIX_CHECK(optixProgramGroupDestroy(m_program));
        m_program = nullptr;
    }

    RayGenProgram::RayGenProgram() noexcept {
        this->module            = nullptr;
        this->entryFunctionName = nullptr;
        this->m_program         = nullptr;
    }

    OptixProgramGroupDesc RayGenProgram::GetOptixProgramGroupDesc() const noexcept {
        OptixProgramGroupDesc desc;
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.flags                    = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        desc.raygen.module            = module;
        desc.raygen.entryFunctionName = entryFunctionName;
        return desc;
    }

    RayGenProgram& RayGenProgram::SetModule(Module* module) noexcept {
        this->module = *module;
        return *this;
    }

    RayGenProgram& RayGenProgram::SetEntry(std::string_view entry) noexcept {
        this->entryFunctionName = entry.data();
        return *this;
    }

    MissProgram::MissProgram() noexcept {
        this->module            = nullptr;
        this->entryFunctionName = nullptr;
        this->m_program         = nullptr;
    }

    OptixProgramGroupDesc MissProgram::GetOptixProgramGroupDesc() const noexcept {
        OptixProgramGroupDesc desc;
        desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.flags                  = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        desc.miss.module            = module;
        desc.miss.entryFunctionName = entryFunctionName;
        return desc;
    }

    MissProgram& MissProgram::SetModule(Module* module) noexcept {
        this->module = *module;
        return *this;
    }

    MissProgram& MissProgram::SetEntry(std::string_view entry) noexcept {
        this->entryFunctionName = entry.data();
        return *this;
    }

    HitgroupProgram::HitgroupProgram() noexcept {
        this->moduleCH            = nullptr;
        this->entryFunctionNameCH = nullptr;
        this->moduleAH            = nullptr;
        this->entryFunctionNameAH = nullptr;
        this->moduleIS            = nullptr;
        this->entryFunctionNameIS = nullptr;
        this->m_program           = nullptr;
    }

    OptixProgramGroupDesc HitgroupProgram::GetOptixProgramGroupDesc() const noexcept {
        OptixProgramGroupDesc desc;
        desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.flags                        = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        desc.hitgroup.moduleCH            = moduleCH;
        desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        desc.hitgroup.moduleAH            = moduleAH;
        desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        desc.hitgroup.moduleIS            = moduleIS;
        desc.hitgroup.entryFunctionNameIS = entryFunctionNameIS;
        return desc;
    }

    HitgroupProgram& HitgroupProgram::SetCHModule(Module* module) noexcept {
        this->moduleCH = *module;
        return *this;
    }

    HitgroupProgram& HitgroupProgram::SetCHEntry(std::string_view entry) noexcept {
        this->entryFunctionNameCH = entry.data();
        return *this;
    }

    HitgroupProgram& HitgroupProgram::SetAHModule(Module* module) noexcept {
        this->moduleAH = *module;
        return *this;
    }

    HitgroupProgram& HitgroupProgram::SetAHEntry(std::string_view entry) noexcept {
        this->entryFunctionNameAH = entry.data();
        return *this;
    }

    HitgroupProgram& HitgroupProgram::SetISModule(Module* module) noexcept {
        this->moduleIS = *module;
        return *this;
    }

    HitgroupProgram& HitgroupProgram::SetISEntry(std::string_view entry) noexcept {
        this->entryFunctionNameIS = entry.data();
        return *this;
    }

    CallableProgram::CallableProgram() noexcept {
        this->moduleDC            = nullptr;
        this->entryFunctionNameDC = nullptr;
        this->moduleCC            = nullptr;
        this->entryFunctionNameCC = nullptr;
        this->m_program           = nullptr;
    }

    OptixProgramGroupDesc CallableProgram::GetOptixProgramGroupDesc() const noexcept {
        OptixProgramGroupDesc desc;
        desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        desc.flags                         = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        desc.callables.moduleDC            = moduleDC;
        desc.callables.entryFunctionNameDC = entryFunctionNameDC;
        desc.callables.moduleCC            = moduleCC;
        desc.callables.entryFunctionNameCC = entryFunctionNameCC;
        return desc;
    }

    CallableProgram& CallableProgram::SetDCModule(Module* module) noexcept {
        this->moduleDC = *module;
        return *this;
    }

    CallableProgram& CallableProgram::SetDCEntry(std::string_view entry) noexcept {
        this->entryFunctionNameDC = entry.data();
        return *this;
    }

    CallableProgram& CallableProgram::SetCCModule(Module* module) noexcept {
        this->moduleCC = *module;
        return *this;
    }

    CallableProgram& CallableProgram::SetCCEntry(std::string_view entry) noexcept {
        this->entryFunctionNameCC = entry.data();
        return *this;
    }

    ExceptionProgram::ExceptionProgram() noexcept {
        this->module            = nullptr;
        this->entryFunctionName = nullptr;
        this->m_program         = nullptr;
    }

    OptixProgramGroupDesc ExceptionProgram::GetOptixProgramGroupDesc() const noexcept {
        OptixProgramGroupDesc desc;
        desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        desc.flags                       = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        desc.exception.module            = module;
        desc.exception.entryFunctionName = entryFunctionName;
        return desc;
    }

    ExceptionProgram& ExceptionProgram::SetModule(Module* module) noexcept {
        this->module = *module;
        return *this;
    }

    ExceptionProgram& ExceptionProgram::SetEntry(std::string_view entry) noexcept {
        this->entryFunctionName = entry.data();
        return *this;
    }

}// namespace Pupil::optix