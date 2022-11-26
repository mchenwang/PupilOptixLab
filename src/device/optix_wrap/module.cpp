#include "module.h"
#include "pipeline.h"
#include "../optix_device.h"
#include "static.h"

#include <optix_stubs.h>

#include <fstream>
#include <filesystem>

optix_wrap::Module::Module(device::Optix *device, OptixPrimitiveType builtin_type) noexcept {
    OptixModuleCompileOptions module_compile_options{
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
    };

    OptixBuiltinISOptions options{.builtinISModuleType = builtin_type};
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(
        device->context,
        &module_compile_options,
        &Pipeline::pipeline_compile_options,
        &options,
        &module));
}

optix_wrap::Module::Module(device::Optix *device, std::string_view file_name) noexcept {
    OptixModuleCompileOptions module_compile_options{
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL};

    std::filesystem::path file_path = std::filesystem::path{ CODE_DIR } / file_name;
    std::ifstream file(file_path.string(), std::ios::binary);
    std::string ptx_source;
    if (!file.good()) assert(false);

    std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
    ptx_source.assign(buffer.begin(), buffer.end());
    
    //std::ifstream ptx_fin(file_path.string());
    //const std::string ptx_source((std::istreambuf_iterator<char>(ptx_fin)), std::istreambuf_iterator<char>());
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        device->context,
        &module_compile_options,
        &Pipeline::pipeline_compile_options,
        ptx_source.c_str(),
        ptx_source.size(),
        LOG,
        &LOG_SIZE,
        &module));
}

optix_wrap::Module::~Module() noexcept {
    if (module) OPTIX_CHECK(optixModuleDestroy(module));
}

