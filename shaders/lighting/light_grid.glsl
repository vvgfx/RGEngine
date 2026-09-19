#ifndef LIGHT_GRID_GLSL
#define LIGHT_GRID_GLSL

const int LIGHT_TILE_SIZE = 16;
const int MAX_LIGHTS_PER_TILE = 64;

layout(set = LIGHT_GRID_SET, binding = 0, std430) readonly buffer LightGridBuffer
{
    uint tileCounts[];
}
lightGrid;

layout(set = LIGHT_GRID_SET, binding = 1, std430) readonly buffer LightIndexBuffer
{
    uint indices[];
}
lightIndexList;

layout(set = LIGHT_GRID_SET, binding = 2) uniform LightGridInfo
{
    uvec4 dims; // xy = tile counts, z = lights culled, w = unused
}
gridInfo;

uint lightTileIndex(vec2 fragCoord)
{
    uvec2 tile = uvec2(fragCoord) / uint(LIGHT_TILE_SIZE);
    tile = min(tile, gridInfo.dims.xy - 1u);
    return tile.y * gridInfo.dims.x + tile.x;
}

uint lightTileCount(uint tileIndex)
{
    return min(lightGrid.tileCounts[tileIndex], uint(MAX_LIGHTS_PER_TILE));
}

uint lightTileEntry(uint tileIndex, uint slot)
{
    return lightIndexList.indices[tileIndex * uint(MAX_LIGHTS_PER_TILE) + slot];
}

#endif
