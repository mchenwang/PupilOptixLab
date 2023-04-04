struct VSInput {
    float3 position : POSITION;
    float2 texcoord : TEXCOORD;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

struct FrameInfo
{
    uint w;
    uint h;
    // uint tone_mapping_type;
};
ConstantBuffer<FrameInfo> frame : register(b0);

// Texture2D tex : register(t0);
Buffer<float4> render_result;

PSInput VSMain(VSInput input) {
    PSInput result;
    result.position = float4(input.position, 1.0f);
    result.texcoord = input.position.xy;
    return result;
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
    int tex_x = (int)((input.texcoord.x + 1.f) / 2.f * frame.w);
    int tex_y = (int)((input.texcoord.y + 1.f) / 2.f * frame.h);
    // return float4(1.f, 1.f, 1.f, 1.f);
    return GammaCorrection(render_result.Load(tex_x + tex_y * (int)frame.w), 2.2f);
}
