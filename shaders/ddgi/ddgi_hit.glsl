#ifndef DDGI_HIT_GLSL
#define DDGI_HIT_GLSL

struct DDGIHit
{
    bool hit;
    bool backface;
    float t;
    vec3 position;
    vec3 normal;
    vec3 albedo;
};

#endif
