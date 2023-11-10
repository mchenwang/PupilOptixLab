#pragma once

#include "util/log.h"

#include <optix.h>
#include <sstream>
#include <assert.h>
#include <iostream>

namespace Pupil::optix {
    inline void OptixCheck(OptixResult res, const char* call, const char* file, unsigned int line) {
        if (res != OPTIX_SUCCESS) {
            Pupil::Log::Error("Optix call failed: {}", call);
            Pupil::Log::Error("location:{} : {}.", file, line);
            assert(false);
        }
    }

    inline void OptixCheckLog(
        OptixResult  res,
        const char*  log,
        size_t       sizeof_log,
        size_t       sizeof_log_returned,
        const char*  call,
        const char*  file,
        unsigned int line) {
        if (res != OPTIX_SUCCESS) {
            Pupil::Log::Error("Optix call failed: {}", call);
            Pupil::Log::Error("with log: {} {}", log, (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : ""));
            Pupil::Log::Error("location:{} : {}.", file, line);
            assert(false);
        }
    }
}// namespace Pupil::optix

#define OPTIX_CHECK(call) Pupil::optix::OptixCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call)                                                \
    do {                                                                     \
        char   LOG[400];                                                     \
        size_t LOG_SIZE = sizeof(LOG);                                       \
        Pupil::optix::OptixCheckLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, \
                                    __FILE__, __LINE__);                     \
    } while (false)
