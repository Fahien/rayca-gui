// Copyright © 2021-2025
// Author: Antonio Caggiano <info@antoniocaggiano.eu>
// SPDX-License-Identifier: MIT

import vertex;

struct PushConstant
{
    float4x4 pretransform;
    float2 screen_size;
}

[vk::push_constant]
PushConstant constant;

[shader("vertex")]
Vertex main(float2 pos, float2 uv, float4 color)
{
    Vertex out;
    float4 model_pos = float4(2.0 * pos.x / constant.screen_size.x - 1.0,
                              2.0 * pos.y / constant.screen_size.y - 1.0,
                              0.0,
                              1.0);
    out.pos = mul(constant.pretransform, model_pos);
    out.color = color;
    out.uv = uv;
    return out;
}
