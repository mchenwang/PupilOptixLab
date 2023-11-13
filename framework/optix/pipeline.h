#pragma once

#include <optix.h>
#include <string>

namespace Pupil::optix {
    enum class EModuleType {
        UserDefined,
        BuiltinSphereIS,
        BuiltinCurveQuadraticIS,
        BuiltinCurveCubicIS,
        BuiltinCurveLinearIS,
        BuiltinCurveCatmullromIS,
        BuiltinCustomIS
    };

    class Pipeline;
    struct Module {
    public:
        Module(Pipeline* pipeline, OptixPrimitiveType) noexcept;
        Module(Pipeline* pipeline, std::string_view) noexcept;
        ~Module() noexcept;

        operator OptixModule() const noexcept { return m_module; }

    private:
        OptixModule m_module = nullptr;
    };

    struct Program {
    public:
        virtual ~Program() noexcept;
        virtual OptixProgramGroupDesc GetOptixProgramGroupDesc() const noexcept = 0;

        operator OptixProgramGroup() const noexcept { return m_program; }

        void Set(OptixProgramGroup program) noexcept { m_program = program; }

    protected:
        OptixProgramGroup m_program = nullptr;
    };

    struct RayGenProgram : public Program, public OptixProgramGroupSingleModule {
    public:
        RayGenProgram() noexcept;
        virtual OptixProgramGroupDesc GetOptixProgramGroupDesc() const noexcept override;

        RayGenProgram& SetModule(Module* module) noexcept;
        RayGenProgram& SetEntry(std::string_view entry) noexcept;
    };

    struct MissProgram : public Program, public OptixProgramGroupSingleModule {
    public:
        MissProgram() noexcept;
        virtual OptixProgramGroupDesc GetOptixProgramGroupDesc() const noexcept override;

        MissProgram& SetModule(Module* module) noexcept;
        MissProgram& SetEntry(std::string_view entry) noexcept;
    };

    struct HitgroupProgram : public Program, public OptixProgramGroupHitgroup {
    public:
        HitgroupProgram() noexcept;
        virtual OptixProgramGroupDesc GetOptixProgramGroupDesc() const noexcept override;

        HitgroupProgram& SetCHModule(Module* module) noexcept;
        HitgroupProgram& SetCHEntry(std::string_view entry) noexcept;
        HitgroupProgram& SetAHModule(Module* module) noexcept;
        HitgroupProgram& SetAHEntry(std::string_view entry) noexcept;
        HitgroupProgram& SetISModule(Module* module) noexcept;
        HitgroupProgram& SetISEntry(std::string_view entry) noexcept;
    };

    struct CallableProgram : public Program, public OptixProgramGroupCallables {
    public:
        CallableProgram() noexcept;
        virtual OptixProgramGroupDesc GetOptixProgramGroupDesc() const noexcept override;

        CallableProgram& SetDCModule(Module* module) noexcept;
        CallableProgram& SetDCEntry(std::string_view entry) noexcept;
        CallableProgram& SetCCModule(Module* module) noexcept;
        CallableProgram& SetCCEntry(std::string_view entry) noexcept;
    };

    struct ExceptionProgram : public Program, public OptixProgramGroupSingleModule {
    public:
        ExceptionProgram() noexcept;
        virtual OptixProgramGroupDesc GetOptixProgramGroupDesc() const noexcept override;

        ExceptionProgram& SetModule(Module* module) noexcept;
        ExceptionProgram& SetEntry(std::string_view entry) noexcept;
    };

    class Pipeline {
    public:
        enum class EPrimitiveType {
            Default,
            Sphere,
            CurveQuadratic,
            CurveCubic,
            CurveLinear,
            CurveCatrom,
            Curve = CurveCubic
        };

        Pipeline() noexcept;
        ~Pipeline() noexcept;
        operator OptixPipeline() const noexcept;

        // Set pipeline compile options
        Pipeline& SetMaxTraceDepth(unsigned int depth) noexcept;
        Pipeline& SetMaxCCDepth(unsigned int depth) noexcept;
        Pipeline& SetMaxDCDepth(unsigned int depth) noexcept;
        Pipeline& EnableMotionBlur(bool enable = true) noexcept;
        Pipeline& SetTraversableGraphFlags(OptixTraversableGraphFlags flags) noexcept;
        Pipeline& SetNumPayloadValues(unsigned int num) noexcept;
        Pipeline& SetNumAttributeValues(unsigned int num) noexcept;
        Pipeline& SetExceptionFlags(OptixExceptionFlags flags) noexcept;
        Pipeline& SetPipelineLaunchParamsVariableName(const char* name) noexcept;
        Pipeline& EnalbePrimitiveType(EPrimitiveType type) noexcept;

        OptixPipelineCompileOptions GetCompileOptions() const noexcept;

        // Set pipeline modules
        Module* CreateModule(EModuleType type, std::string_view embedded_ptx_code = "") noexcept;

        // Set pipeline programs
        RayGenProgram&    CreateRayGen(std::string_view name = "") noexcept;
        MissProgram&      CreateMiss(std::string_view name = "") noexcept;
        HitgroupProgram&  CreateHitgroup(std::string_view name = "") noexcept;
        CallableProgram&  CreateCallable(std::string_view name = "") noexcept;
        ExceptionProgram& CreateException(std::string_view name = "") noexcept;

        void Finish() noexcept;

        Program* FindProgram(std::string_view name) const noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil::optix