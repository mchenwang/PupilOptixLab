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
    float h;
    float w;
};
ConstantBuffer<FrameInfo> frame : register(b0);

// Texture2D tex : register(t0);
Buffer<float4> render_result : register(t0);

PSInput VSMain(VSInput input) {
    PSInput result;
    result.position = float4(input.position, 1.0f);
    result.texcoord = input.position.xy;
    return result;
}

float4 PSMain(PSInput input) : SV_TARGET {
    // float3 color = tex.Sample(linearWrapSampler, input.texcoord).rgb;
    // color += input.position.xyz * 0.5 + 0.5;
    // float3 color = float3(input.texcoord.xy * 0.5 + 0.5, 1.f);
    // return float4(color, 1.f);
    uint tex_x = input.texcoord.x * frame.w;
    uint tex_y = input.texcoord.y * frame.h;

    return render_result.Load(tex_x + tex_y * frame.w);
}
