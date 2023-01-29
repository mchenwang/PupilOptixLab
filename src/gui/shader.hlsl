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
    // texcoord:
    // [-1, 1]-----------------[1, 1]
    //    |                      |
    //    |                      |
    //    |                      |
    // [-1,-1]-----------------[1,-1]
    uint tex_x = (input.texcoord.x + 1.f) / 2.f * frame.w;
    uint tex_y = (input.texcoord.y + 1.f) / 2.f * frame.h;

    return render_result.Load(tex_x + tex_y * frame.w);
}
