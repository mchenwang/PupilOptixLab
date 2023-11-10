#include "cuda/context.h"
#include "cuda/check.h"

#include "dx12/context.h"

#include "util/log.h"

namespace Pupil::cuda {

    void Context::Synchronize() noexcept {
        CUDA_SYNC_CHECK();
    }

    void Context::Init() noexcept {
        if (IsInitialized()) {
            Pupil::Log::Warn("CUDA is initialized repeatedly.");
            return;
        }

        if (auto dx_context = util::Singleton<DirectX::Context>::instance();
            dx_context->IsInitialized()) {
            int num_cuda_devices = 0;
            CUDA_CHECK(cudaGetDeviceCount(&num_cuda_devices));
            assert(num_cuda_devices);

            DXGI_ADAPTER_DESC1 dxgi_adapter_desc{};
            dx_context->adapter->GetDesc1(&dxgi_adapter_desc);

            for (int dev_id = 0; dev_id < num_cuda_devices; dev_id++) {
                cudaDeviceProp dev_prop{};
                CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev_id));
                const auto cmp1 =
                    memcmp(&dxgi_adapter_desc.AdapterLuid.LowPart,
                           dev_prop.luid,
                           sizeof(dxgi_adapter_desc.AdapterLuid.LowPart)) == 0;
                const auto cmp2 =
                    memcmp(&dxgi_adapter_desc.AdapterLuid.HighPart,
                           dev_prop.luid + sizeof(dxgi_adapter_desc.AdapterLuid.LowPart),
                           sizeof(dxgi_adapter_desc.AdapterLuid.HighPart)) == 0;

                if (cmp1 && cmp2) {
                    CUDA_CHECK(cudaSetDevice(dev_id));
                    cuda_device_id = (uint32_t)dev_id;
                    cuda_node_mask = dev_prop.luidDeviceNodeMask;
                    Pupil::Log::Info("CUDA Device with DirectX12 Used [{}] {}.", dev_id, dev_prop.name);
                    break;
                }
            }
        }

        CUDA_CHECK(cudaFree(0));
        CUDA_CHECK(cuCtxGetCurrent(&context));
        m_init_flag = true;
    }

    void Context::Destroy() noexcept {
        if (IsInitialized()) {
            m_init_flag = false;
        }
    }
}// namespace Pupil::cuda