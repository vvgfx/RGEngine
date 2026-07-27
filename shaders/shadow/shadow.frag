#version 450

// Depth-only pass: no color attachments, nothing to output. Depth is written by the
// rasterizer from gl_Position.
void main()
{
}
