#pragma once
#include <SDL_events.h>
#include <vk_types.h>

class Camera
{
  public:
    bool mousePressed = false;
    glm::vec3 velocity;

    // World units per SECOND. zeux's Bistro is in metres and the scene is only ~130 units across,
    // so this is a walking-pace ~10 m/s. (The old 500 assumed centimetres and crossed the whole
    // scene in a quarter second.)
    float speed = 10.f;
    glm::vec3 position;
    // vertical rotation
    float pitch{0.f};
    // horizontal rotation
    float yaw{0.f};

    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();

    glm::vec4 getCameraPos();

    void processSDLEvent(SDL_Event &e);

    /// deltaSeconds keeps movement frame-rate independent; without it speed scales with framerate.
    void update(float deltaSeconds);
};
