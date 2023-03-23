#include "context.h"
#include "check.h"

#include "cuda/context.h"
#include "cuda/util.h"

#include <assert.h>
#include <format>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

namespace {
void ContextLogCB(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << std::format("[{:2}][{:12}]: {}\n", level, tag, message);
}
}// namespace

namespace Pupil::optix {
void Context::Init() noexcept {
    auto cuda_ctx = util::Singleton<Pupil::cuda::Context>::instance();
    assert(cuda_ctx->IsInitialized() && "cuda should be initialized before optix.");

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = &ContextLogCB;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(*cuda_ctx, &options, &context));
}

void Context::Destroy() noexcept {
    CUDA_SYNC_CHECK();
}
}// namespace Pupil::optix