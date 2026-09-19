#version 450

/**
 * Wireframe sphere for one light, generated entirely from gl_VertexIndex -- no vertex buffer.
 *
 * Three great circles (XY, XZ, YZ) drawn as a LINE_LIST: enough to read position and radius at a
 * glance without the clutter of a full wire sphere. Vertex count is 3 * SEG * 2.
 */

layout(push_constant) uniform PushConstants
{
    mat4 viewproj;
    vec4 posRadius; // xyz = world position, w = radius
    vec4 color;
}
pc;

layout(location = 0) out vec3 outColor;

const int SEG = 48;
const float TAU = 6.28318530718;

void main()
{
    int vid = gl_VertexIndex;
    int ring = vid / (SEG * 2);
    int rem = vid % (SEG * 2);

    // Each segment emits two endpoints, so the second shares the next segment's angle.
    float angle = float(rem / 2 + rem % 2) * TAU / float(SEG);
    float c = cos(angle);
    float s = sin(angle);

    vec3 p = ring == 0 ? vec3(c, s, 0.0) : (ring == 1 ? vec3(c, 0.0, s) : vec3(0.0, c, s));

    gl_Position = pc.viewproj * vec4(pc.posRadius.xyz + p * pc.posRadius.w, 1.0);
    outColor = pc.color.rgb;
}
