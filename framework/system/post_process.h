#pragma once

#include "util/util.h"
#include "pass.h"
#include "resource.h"

namespace Pupil {
namespace scene {
class Scene;
}
namespace cuda {
class Stream;
}

enum class EToneMappingType : uint32_t {
    None,
    ACES
};

// transfer custom output buffer to gui(dx12 resource(shared or native))
// with[out] tone mapping, and gamma correction, etc
class PostProcessPass : public Pass, public util::Singleton<PostProcessPass> {
public:
    cudaEvent_t finished_event;
    PostProcessPass() noexcept : Pass("Post Process") {}

    virtual void BeforeRunning() noexcept override;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;
    virtual void SetScene(scene::Scene *) noexcept override;

    void Init() noexcept;
    void Destroy() noexcept;

    void SetToneMappingType(EToneMappingType) noexcept;
    void SetGammaCorrection(bool, float gamma = 2.2f) noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }

    // empty override
    virtual void AfterRunning() noexcept override {}

private:
    uint32_t m_image_w;
    uint32_t m_image_h;
    size_t m_image_size;
    Buffer *output = nullptr;
    Buffer *input = nullptr;

    bool m_init_flag = false;
    std::unique_ptr<cuda::Stream> m_stream;
};
}// namespace Pupil