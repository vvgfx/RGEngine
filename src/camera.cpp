#include "SDL_events.h"
#include "fmt/base.h"
#include "glm/ext/vector_float4.hpp"
#include <camera.h>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

void Camera::update(float deltaSeconds)
{
    glm::mat4 cameraRotation = getRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * speed * deltaSeconds, 0.f));
}

void Camera::processSDLEvent(SDL_Event &e)
{
    if (e.type == SDL_KEYDOWN)
    {
        if (e.key.keysym.sym == SDLK_w)
        {
            velocity.z = -1;
        }
        if (e.key.keysym.sym == SDLK_s)
        {
            velocity.z = 1;
        }
        if (e.key.keysym.sym == SDLK_a)
        {
            velocity.x = -1;
        }
        if (e.key.keysym.sym == SDLK_d)
        {
            velocity.x = 1;
        }

        // Dump the view as paste-ready source. Pitch and yaw matter as much as position: without
        // them the camera lands in the right place looking the wrong way.
        if (e.key.keysym.sym == SDLK_p)
        {
            fmt::println("mainCamera.position = glm::vec3({:.3f}f, {:.3f}f, {:.3f}f);", position.x, position.y, position.z);
            fmt::println("mainCamera.pitch = {:.4f}f;", pitch);
            fmt::println("mainCamera.yaw = {:.4f}f;", yaw);
        }
    }

    if (e.type == SDL_KEYUP)
    {
        if (e.key.keysym.sym == SDLK_w)
        {
            velocity.z = 0;
        }
        if (e.key.keysym.sym == SDLK_s)
        {
            velocity.z = 0;
        }
        if (e.key.keysym.sym == SDLK_a)
        {
            velocity.x = 0;
        }
        if (e.key.keysym.sym == SDLK_d)
        {
            velocity.x = 0;
        }
    }

    if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT)
    {
        mousePressed = true;
    }
    else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_RIGHT)
    {
        mousePressed = false;
    }

    if (e.type == SDL_MOUSEMOTION && mousePressed)
    {
        yaw += (float)e.motion.xrel / 200.f;
        pitch -= (float)e.motion.yrel / 200.f;
    }
}

glm::mat4 Camera::getViewMatrix()
{
    // to create a correct model view, we need to move the world in opposite
    // direction to the camera
    //  so we will create the camera model matrix and invert
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix()
{
    // fairly typical FPS style camera. we join the pitch and yaw rotations into
    // the final rotation matrix

    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{1.f, 0.f, 0.f});
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3{0.f, -1.f, 0.f});

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

glm::vec4 Camera::getCameraPos()
{
    return glm::vec4(position, 1.0f);
}