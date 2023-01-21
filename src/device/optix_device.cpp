#include "optix_device.h"
#include "wsa.h"
#include "d3dx12.h"
#include "optix_wrap/pipeline.h"
#include "optix_wrap/module.h"
#include "optix_wrap/sbt.h"
#include "optix_wrap/mesh.h"

#include "common/util.h"
#include "cuda_util/util.h"

#include "scene/scene.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <iostream>
#include <format>

using namespace device;
using Microsoft::WRL::ComPtr;

namespace {
void ContextLogCB(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << std::format("[{:2}][{:12}]: {}\n", level, tag, message);
}
}// namespace

Optix::Optix(DX12 *dx12_backend) noexcept {
    m_dx12_backend = dx12_backend;

    InitCuda();
    CUcontext cu_ctx = 0;
    CUDA_CHECK(cuCtxGetCurrent(&cu_ctx));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options{};
    options.logCallbackFunction = &ContextLogCB;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
}

Optix::~Optix() noexcept {
    std::vector<std::unique_ptr<optix_wrap::RenderObject>>().swap(m_ros);
    CUDA_CHECK(cudaDestroyExternalSemaphore(cuda_semaphore));
    if (m_frame_resource) {
        for (auto i = 0u; i < DX12::NUM_OF_FRAMES; i++) {
            if (m_frame_resource->frame[i]) {
                CUDA_CHECK(cudaDestroyExternalMemory(m_frame_resource->frame[i]->cuda_ext_memory));
                m_frame_resource->frame[i]->dx12_resource.Reset();
            }
        }
    }
    CUDA_FREE(ias_buffer);
    pipeline.reset();
}

void Optix::InitCuda() noexcept {
    int num_cuda_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_cuda_devices));
    assert(num_cuda_devices);

    DXGI_ADAPTER_DESC1 dxgi_adapter_desc{};
    m_dx12_backend->m_adapter->GetDesc1(&dxgi_adapter_desc);

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
            CUDA_CHECK(cudaStreamCreate(&cuda_stream));
            std::cout << std::format("CUDA Device Used [{}] {}\n", dev_id, dev_prop.name);
            break;
        }
    }

    CUDA_CHECK(cudaFree(0));

    // semaphore
    cudaExternalSemaphoreHandleDesc cuda_semaphore_desc{};
    WindowsSecurityAttributes semaphore_sec_attr{};
    LPCWSTR name = L"Cuda semaphore";
    HANDLE semaphore_shared_handle{};
    cuda_semaphore_desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    m_dx12_backend->device->CreateSharedHandle(m_dx12_backend->m_fence.Get(), &semaphore_sec_attr, GENERIC_ALL, name, &semaphore_shared_handle);
    cuda_semaphore_desc.handle.win32.handle = semaphore_shared_handle;
    cuda_semaphore_desc.flags = 0;
    CUDA_CHECK(cudaImportExternalSemaphore(&cuda_semaphore, &cuda_semaphore_desc));
    CloseHandle(semaphore_shared_handle);
}

[[nodiscard]] std::unique_ptr<CudaDx12SharedTexture> Optix::CreateSharedResourceWithDX12() noexcept {
    auto target = std::make_unique<CudaDx12SharedTexture>();

    auto buffer_size = m_dx12_backend->m_frame_h * m_dx12_backend->m_frame_w * sizeof(float4);
    auto texture_desc = CD3DX12_RESOURCE_DESC::Buffer(
        buffer_size,
        D3D12_RESOURCE_FLAG_NONE);
    //auto texture_desc = CD3DX12_RESOURCE_DESC::Tex2D(
    //    DXGI_FORMAT_R32G32B32A32_FLOAT,
    //    m_dx12_backend->m_frame_w,
    //    m_dx12_backend->m_frame_h,
    //    1, 0, 1, 0,
    //    D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
    auto properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    StopIfFailed(m_dx12_backend->device->CreateCommittedResource(
        &properties, D3D12_HEAP_FLAG_SHARED, &texture_desc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        nullptr, IID_PPV_ARGS(&target->dx12_resource)));
    target->dx12_resource->SetName(L"cuda shared texture");

    HANDLE shared_handle{};
    WindowsSecurityAttributes sec_attr{};

    StopIfFailed(m_dx12_backend->device->CreateSharedHandle(target->dx12_resource.Get(), &sec_attr, GENERIC_ALL, 0, &shared_handle));
    const auto texAllocInfo = m_dx12_backend->device->GetResourceAllocationInfo(cuda_node_mask, 1, &texture_desc);

    cudaExternalMemoryHandleDesc cuda_ext_handle_desc{};
    cuda_ext_handle_desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    cuda_ext_handle_desc.handle.win32.handle = shared_handle;
    cuda_ext_handle_desc.size = texAllocInfo.SizeInBytes;
    cuda_ext_handle_desc.flags = cudaExternalMemoryDedicated;
    CUDA_CHECK(cudaImportExternalMemory(&target->cuda_ext_memory, &cuda_ext_handle_desc));
    CloseHandle(shared_handle);

    cudaExternalMemoryBufferDesc cuda_ext_buffer_desc{};
    cuda_ext_buffer_desc.offset = 0;
    cuda_ext_buffer_desc.size = buffer_size;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&target->cuda_buffer_ptr, target->cuda_ext_memory, &cuda_ext_buffer_desc));

    //cudaExternalMemoryMipmappedArrayDesc cuda_ext_mip_desc{};
    //cuda_ext_mip_desc.extent = make_cudaExtent(texture_desc.Width, texture_desc.Height, 0);
    //cuda_ext_mip_desc.formatDesc = cudaCreateChannelDesc<float4>();
    //cuda_ext_mip_desc.numLevels = 1;
    //cuda_ext_mip_desc.flags = cudaArraySurfaceLoadStore;

    //cudaMipmappedArray_t cuda_mip_array{};
    //CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&cuda_mip_array, target->cuda_ext_memory, &cuda_ext_mip_desc));

    //cudaArray_t cuda_array{};
    //CUDA_CHECK(cudaGetMipmappedArrayLevel(&cuda_array, cuda_mip_array, 0));

    //cudaResourceDesc cuda_res_desc{};
    //cuda_res_desc.resType = cudaResourceTypeArray;
    //cuda_res_desc.res.array.array = cuda_array;
    //CUDA_CHECK(cudaCreateSurfaceObject(&target->cuda_surf_obj, &cuda_res_desc));

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    return std::move(target);
}

SharedFrameResource *Optix::GetSharedFrameResource() noexcept {
    if (m_frame_resource == nullptr) {
        m_frame_resource = std::make_unique<SharedFrameResource>();
        for (auto i = 0u; i < DX12::NUM_OF_FRAMES; i++) {
            m_frame_resource->frame[i] = CreateSharedResourceWithDX12();
        }
    }

    return m_frame_resource.get();
}

void Optix::Run(void *params, size_t params_size, void **frame_buffer) noexcept {
    uint32_t frame_index = m_dx12_backend->GetCurrentFrameIndex();
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.params.fence.value = m_frame_resource->frame[frame_index]->fence_value;
    cudaWaitExternalSemaphoresAsync(&cuda_semaphore, &wait_params, 1, cuda_stream);
    
    *frame_buffer = m_frame_resource->frame[frame_index]->cuda_buffer_ptr;

    if (m_cuda_params != nullptr && m_cuda_params_size != params_size) {
        CUDA_FREE(m_cuda_params);
        m_cuda_params = nullptr;
    }

    if (m_cuda_params == nullptr) {
        CUDA_CHECK(cudaMalloc(&m_cuda_params, params_size));
        m_cuda_params_size = params_size;
    }

    // TODO: do ray tracing
    CUDA_CHECK(cudaMemcpyAsync(
        m_cuda_params,
        params, params_size,
        cudaMemcpyHostToDevice, cuda_stream));

    OPTIX_CHECK(optixLaunch(
        pipeline->pipeline,
        cuda_stream,
        reinterpret_cast<CUdeviceptr>(m_cuda_params),
        params_size,
        m_sbt_ptr,
        m_dx12_backend->m_frame_w,// launch width
        m_dx12_backend->m_frame_h,// launch height
        1                         // launch depth
        ));
    CUDA_SYNC_CHECK();

    cudaExternalSemaphoreSignalParams signal_params{};
    signal_params.params.fence.value = m_frame_resource->frame[frame_index]->fence_value + 1;
    cudaSignalExternalSemaphoresAsync(&cuda_semaphore, &signal_params, 1, cuda_stream);
}

void Optix::InitPipeline(const optix_wrap::PipelineDesc &desc) noexcept {
    pipeline = std::make_unique<optix_wrap::Pipeline>(this, desc);
}

void Optix::InitScene(scene::Scene *scene) noexcept {
    m_ros.clear();
    m_ros.reserve(scene->shapes.size());

    optix_wrap::Mesh temp_mesh{};
    optix_wrap::Sphere temp_sphere{};

    float rect_position[] = {
        1.f, 0.f, 1.f,  // v0
        1.f, 0.f, -1.f, // v1
        -1.f, 0.f, -1.f,// v2
        -1.f, 0.f, 1.f  // v3
    };
    uint32_t rect_indices[] = { 0, 1, 2, 0, 2, 3 };

    float cube_position[] = {
        0.f, 0.f, 0.f,
        0.f, 0.f, 1.f,
        0.f, 1.f, 0.f,
        0.f, 1.f, 1.f,
        1.f, 0.f, 0.f,
        1.f, 0.f, 1.f,
        1.f, 1.f, 0.f,
        1.f, 1.f, 1.f
    };
    uint32_t cube_indices[] = {
        0, 1, 2, 1, 3, 2,
        4, 6, 7, 4, 7, 5,
        0, 2, 6, 0, 6, 4,
        1, 5, 7, 1, 7, 3,
        2, 3, 7, 2, 7, 6,
        0, 4, 5, 0, 5, 1
    };

    for (int i = 0; auto &&shape : scene->shapes) {
        switch (shape.type) {
            case scene::EShapeType::_obj:
                temp_mesh.vertex_num = shape.obj.vertex_num;
                temp_mesh.vertices = shape.obj.positions;
                temp_mesh.index_triplets_num = shape.obj.face_num;
                temp_mesh.indices = shape.obj.indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<optix_wrap::RenderObject>(this, optix_wrap::EMeshType::Custom, &temp_mesh));
                break;
            case scene::EShapeType::_rectangle:
                temp_mesh.vertex_num = 4;
                temp_mesh.vertices = rect_position;
                temp_mesh.index_triplets_num = 2;
                temp_mesh.indices = rect_indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<optix_wrap::RenderObject>(this, optix_wrap::EMeshType::Custom, &temp_mesh));
                break;
            case scene::EShapeType::_cube:
                temp_mesh.vertex_num = 8;
                temp_mesh.vertices = cube_position;
                temp_mesh.index_triplets_num = 12;
                temp_mesh.indices = cube_indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<optix_wrap::RenderObject>(this, optix_wrap::EMeshType::Custom, &temp_mesh));
                break;
            case scene::EShapeType::_sphere:
                temp_sphere.center = make_float3(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
                temp_sphere.radius = shape.sphere.radius;
                std::memcpy(temp_sphere.transform, shape.transform.matrix, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<optix_wrap::RenderObject>(this, optix_wrap::EMeshType::BuiltinSphere, &temp_sphere));
            default:
                break;
        }
    }

    CreateTopLevelAccel(m_ros);
}

void Optix::CreateTopLevelAccel(std::vector<std::unique_ptr<optix_wrap::RenderObject>> &ros) noexcept {
    float transform[12] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f
    };

    const auto num_instances = ros.size();
    std::vector<OptixInstance> instances(num_instances);

    for (auto i = 0u; i < num_instances; i++) {
        memcpy(instances[i].transform, transform, sizeof(float) * 12);
        instances[i].instanceId = i;
        instances[i].sbtOffset = i * 2;
        instances[i].visibilityMask = ros[i]->visibility_mask;
        instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[i].traversableHandle = ros[i]->gas_handle;
    }

    const auto instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
    CUdeviceptr d_instances = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances), instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void **>(d_instances),
        instances.data(),
        instances_size_in_bytes,
        cudaMemcpyHostToDevice));

    OptixBuildInput instance_input{};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &instance_input,
        1,// num build inputs
        &ias_buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), ias_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_ias_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_ias_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,
        &accel_options,
        &instance_input,
        1,
        d_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_ias_and_compacted_size,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle,
        &emitProperty,
        1));

    CUDA_FREE(d_temp_buffer);
    CUDA_FREE(d_instances);

    size_t compacted_ias_size;
    CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ias_buffer), compacted_ias_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(context, 0, ias_handle, ias_buffer, compacted_ias_size, &ias_handle));

        CUDA_FREE(d_buffer_temp_output_ias_and_compacted_size);
    } else {
        ias_buffer = d_buffer_temp_output_ias_and_compacted_size;
    }
}
