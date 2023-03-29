#include "cuda/context.h"
#include "cuda/util.h"

#include "dx12/context.h"

namespace Pupil::cuda {
void Context::Init() noexcept {
    if (IsInitialized()) {
        std::wcerr << "CUDA is initialized repeatedly.\n";
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
                std::cout << std::format("CUDA Device with DirectX12 Used [{}] {}\n", dev_id, dev_prop.name);
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
        for (auto &&[_, stream] : m_streams) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        m_streams.clear();
        m_init_flag = false;
    }
}

cudaStream_t Context::GetStream(std::string_view stream_id) noexcept {
    auto it = m_streams.find(stream_id);
    if (it == m_streams.end()) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        m_streams.emplace(stream_id, stream);
        return stream;
    }

    return it->second;
}
}// namespace Pupil::cuda