#include <RGEngine.h>

#include <box3d/box3d.h>
#include <fmt/core.h>

// Mirrors box3d's docs/hello.md: drop a dynamic unit cube onto a static ground
// slab and step the world. Proves box3d links AND simulates. Note: b3BodyDef
// uses b3Pos (double) for position, not b3Vec3 as the docs example shows.
static void box3d_smoke_test()
{
    b3WorldDef worldDef = b3DefaultWorldDef(); // default gravity {0,-10,0}
    b3WorldId world = b3CreateWorld(&worldDef);

    // Static ground slab centered at y = -10 -> top surface at y = 0.
    b3BodyDef groundDef = b3DefaultBodyDef();
    groundDef.position = b3Pos{ 0.0, -10.0, 0.0 };
    b3BodyId ground = b3CreateBody(world, &groundDef);
    b3BoxHull groundBox = b3MakeBoxHull(50.0f, 10.0f, 50.0f);
    b3ShapeDef groundShape = b3DefaultShapeDef();
    b3CreateHullShape(ground, &groundShape, &groundBox.base);

    // Dynamic unit cube starting at y = 4.
    b3BodyDef boxDef = b3DefaultBodyDef();
    boxDef.type = b3_dynamicBody;
    boxDef.position = b3Pos{ 0.0, 4.0, 0.0 };
    b3BodyId box = b3CreateBody(world, &boxDef);
    b3BoxHull cube = b3MakeCubeHull(1.0f);
    b3ShapeDef cubeShape = b3DefaultShapeDef();
    cubeShape.density = 1.0f; // dynamic bodies need a non-zero density
    b3CreateHullShape(box, &cubeShape, &cube.base);

    const float timeStep = 1.0f / 60.0f;
    const int subStepCount = 4;
    b3Pos start = b3Body_GetPosition(box);
    for (int i = 0; i < 90; ++i)
        b3World_Step(world, timeStep, subStepCount);
    b3Pos end = b3Body_GetPosition(box);

    b3DestroyWorld(world);

    fmt::print("cube fell from y={:.2f} to y={:.2f} over 90 steps -> OK\n",
               start.y, end.y);
}
// --- end box3d smoke test ---

int main(int argc, char *argv[])
{
    box3d_smoke_test();

    // switch from vulkan engine to pbr engine for inheritance hierarchy
    RGEngine engine;

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
