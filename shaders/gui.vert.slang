// Copyright © 2021-2024
// Author: Antonio Caggiano <info@antoniocaggiano.eu>
// SPDX-License-Identifier: MIT

import vertex;

[vk::push_constant]
float2 screen_size;

[shader("vertex")]
Vertex main(float2 pos, float2 uv, float4 color)
{
    Vertex out;
    out.pos = float4(2.0 * pos.x / screen_size.x - 1.0,
                     2.0 * pos.y / screen_size.y - 1.0,
                     0.0,
                     1.0);
    out.color = color;
    out.uv = uv;
    return out;
}
