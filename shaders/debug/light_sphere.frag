#version 450

layout(location = 0) in vec3 inColor;
layout(location = 0) out vec4 outFragColor;

void main()
{
    // drawImage is linear HDR and the post chain tonemaps it, so push the line well above 1 to
    // keep the overlay readable against a bright scene.
    outFragColor = vec4(inColor * 4.0, 1.0);
}
