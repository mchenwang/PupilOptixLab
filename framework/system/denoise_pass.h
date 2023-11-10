#pragma once

#include "system/pass.h"

namespace Pupil {
    class DenoisePass : public Pass {
    public:
        static bool s_enabled_flag;

        struct Config {
            bool        default_enable;
            std::string noise_name;
            bool        use_albedo;
            std::string albedo_name;
            bool        use_normal;
            std::string normal_name;
        };

        DenoisePass(const Config& config, std::string_view name = "Denoise") noexcept;
        virtual void OnRun() noexcept override;
        virtual void Inspector() noexcept override;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil