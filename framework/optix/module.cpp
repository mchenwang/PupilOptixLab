#include "module.h"
#include "pipeline.h"
#include "check.h"
#include "context.h"
#include "static.h"

#include <optix_stubs.h>

#include <fstream>
#include <filesystem>
#include <unordered_map>

// typedef enum OptixPrimitiveType
// {
//     /// Custom primitive.
//     OPTIX_PRIMITIVE_TYPE_CUSTOM                        = 0x2500,
//     /// B-spline curve of degree 2 with circular cross-section.
//     OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE       = 0x2501,
//     /// B-spline curve of degree 3 with circular cross-section.
//     OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE           = 0x2502,
//     /// Piecewise linear curve with circular cross-section.
//     OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR                  = 0x2503,
//     /// CatmullRom curve with circular cross-section.
//     OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM              = 0x2504,
//     OPTIX_PRIMITIVE_TYPE_SPHERE                        = 0x2506,
//     /// Triangle.
//     OPTIX_PRIMITIVE_TYPE_TRIANGLE                      = 0x2531,
// } OptixPrimitiveType;

namespace {
//built-in modules
std::unordered_map<Pupil::optix::EModuleBuiltinType,
                   std::unique_ptr<Pupil::optix::Module>>
    s_builtin_modules;
}// namespace

//pupil built-in modules
extern "C" char g_pupil_material_embedded_ptx_code[];

namespace Pupil::optix {
[[nodiscard]] Module *ModuleManager::GetModule(EModuleBuiltinType builtin_type) noexcept {
    if (s_builtin_modules.find(builtin_type) != s_builtin_modules.end())
        return s_builtin_modules[builtin_type].get();

    auto context = util::Singleton<Context>::instance();
    switch (builtin_type) {
        case Pupil::optix::EModuleBuiltinType::CustomPrimitive:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, OPTIX_PRIMITIVE_TYPE_CUSTOM);
            break;
        case Pupil::optix::EModuleBuiltinType::RoundQuadraticBsplinePrimitive:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE);
            break;
        case Pupil::optix::EModuleBuiltinType::RoundCubicBsplinePrimitive:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
            break;
        case Pupil::optix::EModuleBuiltinType::RoundLinearPrimitive:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR);
            break;
        case Pupil::optix::EModuleBuiltinType::RoundCatmullromPrimitive:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM);
            break;
        case Pupil::optix::EModuleBuiltinType::SpherePrimitive:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, OPTIX_PRIMITIVE_TYPE_SPHERE);
            break;
        case Pupil::optix::EModuleBuiltinType::TrianglePrimitive:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, OPTIX_PRIMITIVE_TYPE_TRIANGLE);
            break;
        case Pupil::optix::EModuleBuiltinType::Material:
            s_builtin_modules[builtin_type] = std::make_unique<Module>(*context, g_pupil_material_embedded_ptx_code);
            break;
    }
    return s_builtin_modules[builtin_type].get();
}

[[nodiscard]] Module *ModuleManager::GetModule(std::string_view embedded_ptx_code) noexcept {
    const char *id = embedded_ptx_code.data();

    auto it = m_modules.find(id);
    if (it == m_modules.end()) {
        auto context = util::Singleton<Context>::instance();
        m_modules.emplace(id, std::make_unique<Module>(*context, embedded_ptx_code));
    }
    return m_modules.find(id)->second.get();
}

void ModuleManager::Clear() noexcept {
    m_modules.clear();
    s_builtin_modules.clear();
}

Module::Module(OptixDeviceContext context, OptixPrimitiveType builtin_type) noexcept {
    OptixModuleCompileOptions module_compile_options{
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
    };

    OptixBuiltinISOptions options{ .builtinISModuleType = builtin_type };
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(
        context,
        &module_compile_options,
        &Pipeline::pipeline_compile_options,
        &options,
        &optix_module));
}

Module::Module(OptixDeviceContext context, std::string_view embedded_ptx_code) noexcept {
    OptixModuleCompileOptions module_compile_options{
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
    };

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &Pipeline::pipeline_compile_options,
        embedded_ptx_code.data(),
        embedded_ptx_code.size(),
        LOG,
        &LOG_SIZE,
        &optix_module));
}

Module::~Module() noexcept {
    if (optix_module) OPTIX_CHECK(optixModuleDestroy(optix_module));
}
}// namespace Pupil::optix
