// Copyright © 2021-2024
// Author: Antonio Caggiano <info@antoniocaggiano.eu>
// SPDX-License-Identifier: MIT

import vertex;

Texture2D font_image;

[shader("fragment")]
float4 main(in Vertex vertex) : SV_Target {
    SamplerState s;
    return vertex.color * font_image.Sample(s, vertex.uv);
}
