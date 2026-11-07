#include <SwiftUI/SwiftUI_Metal.h>
using namespace metal;

static float hash21(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

// The image develops out of coarse noise and split colour plates, as diffusion does.
[[ stitchable ]] half4 developReveal(float2 position, SwiftUI::Layer layer, float2 size, float progress) {
    float p = clamp(progress, 0.0, 1.0);
    float remaining = 1.0 - p;
    float cell = mix(1.0, 42.0, remaining * remaining);
    float2 snapped = (floor(position / cell) + 0.5) * cell;
    float split = 14.0 * remaining;
    half4 base = layer.sample(snapped);
    half r = layer.sample(snapped + float2(split, 0)).r;
    half b = layer.sample(snapped - float2(split, 0)).b;
    float grain = hash21(floor(position / max(cell, 2.0))) * remaining;
    half3 mixed = half3(r, base.g, b);
    mixed = mix(mixed, half3(grain), half(remaining * 0.65));
    return half4(mixed * base.a, base.a);
}

// A shockwave across the image when a better one replaces it.
[[ stitchable ]] half4 ripple(float2 position, SwiftUI::Layer layer, float2 origin, float time, float amplitude, float frequency, float decay, float speed) {
    float distance = length(position - origin);
    float delay = distance / speed;
    float t = max(0.0, time - delay);
    float wave = amplitude * sin(frequency * t) * exp(-decay * t);
    float2 direction = normalize(position - origin + float2(0.0001));
    float2 sampled = position + wave * direction;
    half4 color = layer.sample(sampled);
    color.rgb += half(0.25 * (wave / max(amplitude, 0.001)) * 0.5) * color.a;
    return color;
}
