#include "module.h"
#include "pipeline.h"
#include "check.h"
#include "context.h"
#include "static.h"

#include <optix_stubs.h>

#include <fstream>
#include <filesystem>
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
//built-in module
std::unique_ptr<Pupil::optix::Module> s_custom;
std::unique_ptr<Pupil::optix::Module> s_round_quadratic_bspline;
std::unique_ptr<Pupil::optix::Module> s_round_cubic_bspline;
std::unique_ptr<Pupil::optix::Module> s_round_linear;
std::unique_ptr<Pupil::optix::Module> s_round_catmullrom;
std::unique_ptr<Pupil::optix::Module> s_sphere;
std::unique_ptr<Pupil::optix::Module> s_triangle;
}// namespace

namespace Pupil::optix {
[[nodiscard]] Module *ModuleManager::GetModule(OptixPrimitiveType builtin_type) noexcept {
    auto *temp = &s_custom;
    switch (builtin_type) {
        case OPTIX_PRIMITIVE_TYPE_CUSTOM:
            if (s_custom != nullptr) return s_custom.get();
            break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
            if (s_round_quadratic_bspline != nullptr) return s_round_quadratic_bspline.get();
            temp = &s_round_quadratic_bspline;
            break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
            if (s_round_cubic_bspline != nullptr) return s_round_cubic_bspline.get();
            temp = &s_round_cubic_bspline;
            break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
            if (s_round_linear != nullptr) return s_round_linear.get();
            temp = &s_round_linear;
            break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:
            if (s_round_catmullrom != nullptr) return s_round_catmullrom.get();
            temp = &s_round_catmullrom;
            break;
        case OPTIX_PRIMITIVE_TYPE_SPHERE:
            if (s_sphere != nullptr) return s_sphere.get();
            temp = &s_sphere;
            break;
        case OPTIX_PRIMITIVE_TYPE_TRIANGLE:
            if (s_triangle != nullptr) return s_triangle.get();
            temp = &s_triangle;
            break;
    }

    auto context = util::Singleton<Context>::instance();
    *temp = std::make_unique<Module>(*context, builtin_type);
    return temp->get();
}
[[nodiscard]] Module *ModuleManager::GetModule(std::string_view file_relative_path) noexcept {
    std::filesystem::path path = std::filesystem::path{ ROOT_DIR } / file_relative_path;
    path.make_preferred();

    std::string id = path.string();

    auto it = m_modules.find(id);
    if (it == m_modules.end()) {
        auto context = util::Singleton<Context>::instance();
        m_modules.emplace(id, std::make_unique<Module>(*context, id));
    }
    return m_modules.find(id)->second.get();
}

void ModuleManager::Clear() noexcept {
    m_modules.clear();
    s_custom.reset();
    s_round_quadratic_bspline.reset();
    s_round_cubic_bspline.reset();
    s_round_linear.reset();
    s_round_catmullrom.reset();
    s_sphere.reset();
    s_triangle.reset();
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

Module::Module(OptixDeviceContext context, std::string_view file_id) noexcept {
    OptixModuleCompileOptions module_compile_options{
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
    };

    std::filesystem::path file_path{ file_id };
    std::ifstream file(file_path.string(), std::ios::binary);
    std::string ptx_source;
    if (!file.good()) assert(false);

    std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
    ptx_source.assign(buffer.begin(), buffer.end());

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &Pipeline::pipeline_compile_options,
        ptx_source.c_str(),
        ptx_source.size(),
        LOG,
        &LOG_SIZE,
        &optix_module));
}

Module::~Module() noexcept {
    if (optix_module) OPTIX_CHECK(optixModuleDestroy(optix_module));
}
}// namespace Pupil::optix
