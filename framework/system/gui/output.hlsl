struct VSInput {
    float3 position : POSITION;
    float2 texcoord : TEXCOORD;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

struct CanvasDesc
{
    uint w;
    uint h;
    uint tone_mapping;
    uint gamma_correct;
};
ConstantBuffer<CanvasDesc> desc : register(b0);

// Texture2D tex : register(t0);
Buffer<float4> render_result;

PSInput VSMain(VSInput input) {
    PSInput result;
    result.position = float4(input.position, 1.0f);
    result.texcoord = input.position.xy;
    return result;
}

float3 ACESToneMapping(float3 color, float adapted_lum) {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;

    color *= adapted_lum;
    return (color * (A * color + B)) / (color * (C * color + D) + E);
}

float3 GammaCorrection(float3 color, float gamma) {
    float3 ret;
    ret.x = pow(color.x, 1.f / gamma);
    ret.y = pow(color.y, 1.f / gamma);
    ret.z = pow(color.z, 1.f / gamma);
    return ret;
}

float4 GammaCorrection(float4 color, float gamma) {
    float4 ret;
    ret.x = pow(color.x, 1.f / gamma);
    ret.y = pow(color.y, 1.f / gamma);
    ret.z = pow(color.z, 1.f / gamma);
    ret.w = color.w;
    return ret;
}

float4 PSMain(PSInput input) : SV_TARGET {
    // texcoord:
    // [-1, 1]-----------------[1, 1]
    //    |                      |
    //    |                      |
    //    |                      |
    // [-1,-1]-----------------[1,-1]
    int tex_x = (int)((input.texcoord.x + 1.f) / 2.f * desc.w);
    int tex_y = (int)((input.texcoord.y + 1.f) / 2.f * desc.h);

    float3 color = render_result.Load(tex_x + tex_y * (int)desc.w).xyz;
    if (desc.tone_mapping == 1) color = ACESToneMapping(color, 1.f);
    if (desc.gamma_correct == 1) color = GammaCorrection(color, 2.2f);
    //  return GammaCorrection(render_result.Load(tex_x + tex_y * (int)desc.w), 2.2f);
    return float4(color, 1.f);
}
