#include "context.h"
#include "check.h"

#include "cuda/context.h"
#include "cuda/util.h"

#include "util/log.h"

#include <assert.h>
#include <format>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

namespace {
void ContextLogCB(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    switch (level) {
        case 1:// fatal
            Pupil::Log::Error("OPTIX [{}][{}]: {}", level, tag, message);
            break;
        case 2:// error
            Pupil::Log::Error("OPTIX [{}][{}]: {}", level, tag, message);
            break;
        case 3:// warning
            Pupil::Log::Warn("OPTIX [{}][{}]: {}", level, tag, message);
            break;
        case 4:// print
            Pupil::Log::Info("OPTIX [{}][{}]: {}", level, tag, message);
            break;
        default:
            Pupil::Log::Error("OPTIX [{}][{}]: {}", level, tag, message);
            break;
    }
}
}// namespace

namespace Pupil::optix {
void Context::Init() noexcept {
    if (IsInitialized()) return;
    auto cuda_ctx = util::Singleton<Pupil::cuda::Context>::instance();
    assert(cuda_ctx->IsInitialized() && "cuda should be initialized before optix.");

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = &ContextLogCB;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(*cuda_ctx, &options, &context));
    m_init_flag = true;
    Pupil::Log::Info("OPTIX is initialized.");
}

void Context::Destroy() noexcept {
    if (IsInitialized()) {
        CUDA_SYNC_CHECK();
        m_init_flag = false;
    }
}
}// namespace Pupil::optix