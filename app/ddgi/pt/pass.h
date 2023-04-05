#pragma once

#include "type.h"

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "scene/scene.h"
#include "optix/pass.h"
#include "optix/scene/scene.h"
#include "optix/scene/camera.h"

#include "cuda/stream.h"

#include "util/timer.h"

#include <memory>
#include <mutex>

namespace Pupil::ddgi::pt {
// Shading binding table，用于绑定管线在不同阶段可以访问的数据
// 封装的版本要求必须包含__raygen__xxx、__miss__xxx、__closesthit__xxx三个阶段对应的数据类型
// 这里使用了concept，只要将等号右边的类型改成自定义的类型即可
struct SBTTypes {
    using RayGenDataType = Pupil::ddgi::pt::RayGenData;
    using MissDataType = Pupil::ddgi::pt::MissData;
    using HitGroupDataType = Pupil::ddgi::pt::HitGroupData;
};

// 自定义Pass需要重载四个方法：
//   1. Run() Pass执行的逻辑
//   2. BeforeRunning() 在所有Pass的Run()执行之前执行
//   3. AfterRunning() 在所有Pass的Run()执行之后执行
//   4. Inspector() 用于自定义UI上的显示与操作
// 默认的Pass是会每帧都执行，如果需要预处理Pass，则添加Pre Tag：
// Pass(std::string_view name, EPassTag tag = EPassTag::Pre)
class PTPass : public Pass {
public:
    PTPass(std::string_view name = "DDGI PT Pass") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(World *world) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;
    void SetSBT(scene::Scene *) noexcept;

    OptixLaunchParams m_optix_launch_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;

    std::atomic_bool m_dirty = true;
    optix::CameraHelper *m_world_camera = nullptr;

    Timer m_timer;
};
}// namespace Pupil::ddgi::pt