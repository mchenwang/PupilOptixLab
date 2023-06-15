#include <optix.h>
#include "type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"
#include "material/bsdf/bsdf.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ pt::OptixLaunchParams optix_launch_params;
//pt::OptixLaunchParams optix_launch_params;
}

struct HitInfo {
    optix::LocalGeometry geo;
    optix::material::Material::LocalBsdf bsdf;
    int emitter_index;
};

struct PathPayloadRecord {
    float3 radiance;
    float3 env_radiance;
    float env_pdf;
    cuda::Random random;

    float3 throughput;

    HitInfo hit;

    unsigned int depth;
    bool done;
};

__device__ int2 ToPreviousScreen(const float3 & world_position, int w, int h) {
    float4 p = optix_launch_params.pre_proj_view_mat * make_float4(world_position, 1);
    float x = 0.5 * (p.x / p.w) + 0.5;
    float y = 0.5 * (p.y / p.w) + 0.5;
    return make_int2(__float2int_rn(x * w), __float2int_rn(y * h));
}

__device__ float vMFPdf(const pt::vMF &vmf, const float3 & w) {
    const float3 &mu = vmf.mu;
    const float kappa = vmf.kappa;

    float eMin2Kappa = exp(-2.f * kappa);
    float de = 2 * M_PIf * (1.f - eMin2Kappa);
    float pdfFactor = kappa / de;
    float t = dot(mu, w) - 1.f;
    float e = exp(kappa * t);
    return pdfFactor * e;
}

__device__ float3 vMFSample(const pt::vMF &vmf, const float2 & rn, float * pdf) {
    const float3 &mu = vmf.mu;
    const float kappa = vmf.kappa;

    float sinPhi, cosPhi;
    sincos(2 * M_PIf * rn.y, &sinPhi, &cosPhi);

    float eMin2Kappa = exp(-2.f * kappa);
    float value = rn.x + (1.f - rn.x) * eMin2Kappa;
    float cosTheta = clamp(1.f + std::log(value) / kappa, -1.f, 1.f);
    float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

    float3 w = optix::ToWorld(make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta), mu);

    float de = 2 * M_PIf * (1.f - eMin2Kappa);
    float e = exp(kappa * (dot(mu, w) - 1.f));
    *pdf = kappa / de * e;

    return w;
}

__device__ const float bsdf_sampling_fraction = 0.1;

__device__ void guidedBsdfSample(optix::BsdfSamplingRecord &record, const HitInfo &local_hit, const pt::vMF &pre_model) {
    float pdf_vmf = 0;
    if (record.sampler->Next() < bsdf_sampling_fraction) {
        local_hit.bsdf.Sample(record);
        pdf_vmf = vMFPdf(pre_model, optix::ToWorld(record.wi, local_hit.geo.normal));
    } else {
        float3 wi = vMFSample(pre_model, record.sampler->Next2(), &pdf_vmf);
        record.wi = optix::ToLocal(wi, local_hit.geo.normal);
        local_hit.bsdf.Eval(record);
    }
    record.pdf = bsdf_sampling_fraction * record.pdf + (1 - bsdf_sampling_fraction) * pdf_vmf;
}

__device__ void guidedBsdfEval(optix::BsdfSamplingRecord &record, const HitInfo &local_hit, const pt::vMF &pre_model) {
    local_hit.bsdf.Eval(record);
    float pdf_vmf = vMFPdf(pre_model, optix::ToWorld(record.wi, local_hit.geo.normal));
    record.pdf = bsdf_sampling_fraction * record.pdf + (1 - bsdf_sampling_fraction) * pdf_vmf;
}

__device__ void vMFUpdate(int2 index, int w, int h) {
    int pixel_index = index.y * w + index.x;
    float3 position = optix_launch_params.position_buffer[pixel_index];
    pt::vMF &new_model = optix_launch_params.new_model_buffer[pixel_index];

    // compute batch sufficient statistics
    int radius = 3;
    float bat_weight_sum = 0;
    float3 bat_weight_dir_sum = make_float3(0);
    for (int dx = -radius; dx <= radius; ++dx) {
        int x = index.x + dx;
        if (x >= 0 && x < w) {
            for (int dy = -radius; dy <= radius; ++dy) {
                int y = index.y + dy;
                if (y >= 0 && y < h) {
                    int target_index = y * w + x;
                    float radiance = optix_launch_params.radiance_buffer[target_index];
                    float pdf = optix_launch_params.pdf_buffer[target_index];

                    if (radiance > 0 && pdf > 0) {
                        float weight = radiance / pdf;
                        float3 target_position = optix_launch_params.target_buffer[target_index];
                        float3 dir = normalize(target_position - position);
                        bat_weight_sum += weight;
                        bat_weight_dir_sum += weight * dir;
                    }
                }
            }
        }
    }

    // no training data
    if (bat_weight_sum <= 0) {
        return;
    }

    // compute previous sufficient statistics
    float pre_weight_sum = 0;
    float3 pre_weight_dir_sum = make_float3(0);
    if (new_model.iteration_cnt > 0) {
        // this pixel has a valid history, reuse the model
        pre_weight_sum = new_model.weight_sum;
        pre_weight_dir_sum = pre_weight_sum * new_model.mean_cosine * new_model.mu;
    }

    // step-wise EM
    new_model.iteration_cnt += 1;
    float moving_weight = 1.f / new_model.iteration_cnt;
    /*float weight_sum = (1.f - moving_weight) * pre_weight_sum + moving_weight * bat_weight_sum;
    float3 weight_dir_sum = (1.f- moving_weight) * pre_weight_dir_sum + moving_weight * bat_weight_dir_sum;*/
    float weight_sum = bat_weight_sum;
    float3 weight_dir_sum = bat_weight_dir_sum;

    float r_length = length(weight_dir_sum);

    // in case of singularity
    if (weight_sum <= 0 || r_length <= 0) {
        return;
    }

    // update the model
    float mc = r_length / weight_sum;
    new_model.mu = weight_dir_sum / r_length;
    new_model.kappa = clamp(mc * (3 - mc * mc) / (1 - mc * mc), 1e-2, 1e3);
    new_model.mean_cosine = mc;
    new_model.weight_sum = weight_sum;
}

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    // check whether this is an update pass
    if (optix_launch_params.update_pass) {
        vMFUpdate(make_int2(index.x, index.y), w, h);
        return;
    }

    auto &camera = *optix_launch_params.camera.GetDataPtr();

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);

    record.done = false;
    record.depth = 0u;
    record.throughput = make_float3(1.f);
    record.radiance = make_float3(0.f);
    record.env_radiance = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);

    const float2 subpixel_jitter = make_float2(record.random.Next(), record.random.Next());

    const float2 subpixel = make_float2(
            (static_cast<float>(index.x + 0.99999f) - subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(index.y + 0.99999f) - subpixel_jitter.y) / static_cast<float>(h)
    );
    const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

    float4 d = camera.sample_to_camera * point_on_film;

    d /= d.w;
    d.w = 0.f;
    d = normalize(d);

    float3 ray_direction = normalize(make_float3(camera.camera_to_world * d));

    float3 ray_origin = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 2, 0,
               u0, u1);

    int depth = 0;
    auto local_hit = record.hit;

    if (!record.done && record.hit.emitter_index >= 0) {
        auto &emitter = optix_launch_params.emitters.areas[local_hit.emitter_index];
        auto emission = emitter.GetRadiance(local_hit.geo.texcoord);
        record.radiance += emission;
    }

    bool use_path_guiding = optix_launch_params.config.use_path_guiding;

    // for gathering sample data
    float first_bounce_pdf = 0;
    float3 first_bounce_position = local_hit.geo.position;
    float3 first_bounce_target_position = make_float3(0.f);
    float3 first_bounce_radiance = make_float3(0.f);
    float3 first_bounce_throughput = make_float3(1.f);

    // project current position to previous screen
    int2 pre_index = ToPreviousScreen(local_hit.geo.position, w, h);
    int pre_pixel_index = pre_index.y * w + pre_index.x;

    // TODO handle occlusion (more validation)
    // check validity of temporal reuse
    bool pre_model_valid = false;
    if (pre_index.x >= 0 && pre_index.x < w && pre_index.y >= 0 && pre_index.y < h &&
        optix_launch_params.pre_model_buffer[pre_pixel_index].iteration_cnt > 0)
    {
        pre_model_valid = true;
    }

    while (!record.done) {
        ++depth;
        if (depth >= optix_launch_params.config.max_depth)
            break;

        // direct light sampling
        {
            auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(record.random.Next());
            optix::EmitterSampleRecord emitter_sample_record;
            emitter.SampleDirect(emitter_sample_record, local_hit.geo, record.random.Next2());

            bool occluded =
                optix::Emitter::TraceShadowRay(
                    optix_launch_params.handle,
                    local_hit.geo.position, emitter_sample_record.wi,
                    0.001f, emitter_sample_record.distance - 0.001f);
            if (!occluded) {
                optix::BsdfSamplingRecord eval_record;
                eval_record.wi = optix::ToLocal(emitter_sample_record.wi, local_hit.geo.normal);
                eval_record.wo = optix::ToLocal(-ray_direction, local_hit.geo.normal);
                eval_record.sampler = &record.random;

                if (use_path_guiding && depth == 1 && pre_model_valid) {
                    const pt::vMF &pre_model = optix_launch_params.pre_model_buffer[pre_pixel_index];
                    guidedBsdfEval(eval_record, local_hit, pre_model);
                } else {
                    record.hit.bsdf.Eval(eval_record);
                }

                float3 f = eval_record.f;
                float pdf = eval_record.pdf;
                if (!optix::IsZero(f * emitter_sample_record.pdf)) {
                    float NoL = abs(dot(local_hit.geo.normal, emitter_sample_record.wi));
                    float mis = emitter_sample_record.is_delta ? 1.f : optix::MISWeight(emitter_sample_record.pdf, pdf);
                    emitter_sample_record.pdf *= emitter.select_probability;
                    float3 value = emitter_sample_record.radiance * f * NoL * mis / emitter_sample_record.pdf;
                    record.radiance += record.throughput * value;

                    if (depth > 1) {
                        first_bounce_radiance += first_bounce_throughput * value;
                    }
                }
            }
        }
        // bsdf sampling
        {
            optix::BsdfSamplingRecord bsdf_sample_record;
            bsdf_sample_record.wo = optix::ToLocal(-ray_direction, local_hit.geo.normal);
            bsdf_sample_record.sampler = &record.random;

            // retrain if an invalid history
            pt::vMF &new_model = optix_launch_params.new_model_buffer[pixel_index];
            new_model.iteration_cnt = 0;

            if (use_path_guiding && depth == 1 && pre_model_valid) {
                const pt::vMF &pre_model = optix_launch_params.pre_model_buffer[pre_pixel_index];
                guidedBsdfSample(bsdf_sample_record, local_hit, pre_model);

                // project previous model to current screen position
                new_model = pre_model;
            } else {
                local_hit.bsdf.Sample(bsdf_sample_record);
            }

            float3 bsdfWeight = bsdf_sample_record.f * abs(bsdf_sample_record.wi.z);
            if (optix::IsZero(bsdfWeight) || optix::IsZero(bsdf_sample_record.pdf)) {
                break;
            }

            record.throughput *= bsdfWeight / bsdf_sample_record.pdf;
            if (depth > 1) {
                first_bounce_throughput *= bsdfWeight / bsdf_sample_record.pdf;
            }

            float rr = depth > 2 ? 0.95 : 1.0;
            if (record.random.Next() > rr)
                break;
            record.throughput /= rr;
            if (depth > 1) {
                first_bounce_throughput /= rr;
            }

            ray_origin = record.hit.geo.position;
            ray_direction = optix::ToWorld(bsdf_sample_record.wi, local_hit.geo.normal);

            optixTrace(optix_launch_params.handle,
                       ray_origin, ray_direction,
                       0.001f, 1e16f, 0.f,
                       255, OPTIX_RAY_FLAG_NONE,
                       0, 2, 0,
                       u0, u1);

            // TODO looks strange
            if (record.done) {
                float mis = optix::MISWeight(bsdf_sample_record.pdf, record.env_pdf);
                record.env_radiance *= record.throughput * mis;
                first_bounce_radiance += first_bounce_throughput * mis;
                break;
            }

            if (depth == 1) {
                first_bounce_pdf = bsdf_sample_record.pdf;
                first_bounce_target_position = record.hit.geo.position;
            }

            local_hit = record.hit;
            if (record.hit.emitter_index >= 0) {
                auto &emitter = optix_launch_params.emitters.areas[record.hit.emitter_index];
                optix::EmitEvalRecord emit_record;
                emitter.Eval(emit_record, record.hit.geo, ray_origin);

                if (!optix::IsZero(emit_record.pdf)) {
                    float mis = bsdf_sample_record.sampled_type & optix::EBsdfLobeType::Delta ?
                                    1.f :
                                    optix::MISWeight(bsdf_sample_record.pdf, emit_record.pdf * emitter.select_probability);
                    record.radiance += record.throughput * emit_record.radiance * mis;
                    first_bounce_radiance += first_bounce_throughput * emit_record.radiance * mis;
                }
            }
        }
    }
    record.radiance += record.env_radiance;

    if (optix_launch_params.config.accumulated_flag && optix_launch_params.sample_cnt > 0) {
        const float t = 1.f / (optix_launch_params.sample_cnt + 1.f);
        const float3 pre = make_float3(optix_launch_params.accum_buffer[pixel_index]);
        record.radiance = lerp(pre, record.radiance, t);
    }
    optix_launch_params.accum_buffer[pixel_index] = make_float4(record.radiance, 1.f);
    optix_launch_params.frame_buffer[pixel_index] = make_float4(record.radiance, 1.f);

    // save training data
    if (use_path_guiding) {
        optix_launch_params.position_buffer[pixel_index] = first_bounce_position;
        optix_launch_params.target_buffer[pixel_index] = first_bounce_target_position;
        optix_launch_params.pdf_buffer[pixel_index] = first_bounce_pdf;
        optix_launch_params.radiance_buffer[pixel_index] = optix::GetLuminance(first_bounce_radiance);

        if (!optix::IsZero(first_bounce_position)) {
            // optix_launch_params.frame_buffer[pixel_index] = make_float4(make_float3(0), 1);
            // optix_launch_params.frame_buffer[pixel_index] = make_float4(normalize(first_bounce_target_position), 1);
            // optix_launch_params.frame_buffer[pixel_index] = make_float4(normalize(optix_launch_params.new_model_buffer[pixel_index].mu), 1);
        }
    }
}

extern "C" __global__ void __miss__default() {
    auto record = optix::GetPRD<PathPayloadRecord>();
    if (optix_launch_params.emitters.env) {
        optix::LocalGeometry temp;
        temp.position = optixGetWorldRayDirection();
        float3 scatter_pos = make_float3(0.f);
        optix::EmitEvalRecord env_emit_record;
        optix_launch_params.emitters.env->Eval(env_emit_record, temp, scatter_pos);
        record->env_radiance = env_emit_record.radiance;
        record->env_pdf = env_emit_record.pdf;
    }
    record->done = true;
}
extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}
extern "C" __global__ void __closesthit__default() {
    const pt::HitGroupData *sbt_data = (pt::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();
    const auto ray_o = optixGetWorldRayOrigin();

    sbt_data->geo.GetHitLocalGeometry(record->hit.geo, ray_dir, sbt_data->mat.twosided);
    if (sbt_data->emitter_index_offset >= 0) {
        record->hit.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
    } else {
        record->hit.emitter_index = -1;
    }
    record->hit.bsdf = sbt_data->mat.GetLocalBsdf(record->hit.geo.texcoord);
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}
