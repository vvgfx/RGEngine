#pragma once
#include <SDL_events.h>
#include <vk_types.h>

class Camera
{
  public:
    bool mousePressed = false;
    glm::vec3 velocity;

    // world units per SECOND. Bistro is in centimetres, so this is ~5 m/s.
    float speed = 500.f;
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
