# Add source to this project's executable.
add_executable (engine
  main.cpp
  vk_types.h
  vk_initializers.cpp
  vk_initializers.h
  vk_images.h
  vk_images.cpp
  vk_descriptors.h
  vk_descriptors.cpp
  vk_pipelines.h
  vk_pipelines.cpp
  vk_engine.h
  vk_engine.cpp
  vk_loader.h
  vk_loader.cpp
  camera.cpp
  camera.h
  GPUResourceAllocator.h
  GPUResourceAllocator.cpp
  sgraph/ScenegraphStructs.h
  sgraph/ScenegraphStructs.cpp
  sgraph/ScenegraphImporter.h
  sgraph/IScenegraph.h
  sgraph/Scenegraph.h
  sgraph/Scenegraph.cpp
  RGEngine.h
  RGEngine.cpp
  rgraph/IFeature.h
  rgraph/Rendergraph.h
  rgraph/Rendergraph.cpp
  rgraph/features/PBRShadingFeature.h
  rgraph/features/PBRShadingFeature.cpp
  rgraph/features/ComputeBackgroundFeature.h
  rgraph/features/ComputeBackgroundFeature.cpp
  rgraph/features/DeferredRenderingFeature.h
  rgraph/features/DeferredRenderingFeature.cpp
  rgraph/features/SMAAFeature.h
  rgraph/features/SMAAFeature.cpp
  MaterialSystem.h
  MaterialSystem.cpp
)

set_property(TARGET engine PROPERTY CXX_STANDARD 20)
target_compile_definitions(engine PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_include_directories(engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(engine PUBLIC vma glm Vulkan::Vulkan fmt::fmt stb_image SDL2::SDL2 vkbootstrap imgui fastgltf::fastgltf)

target_precompile_headers(engine PUBLIC <optional> <vector> <memory> <string> <vector> <unordered_map> <glm/mat4x4.hpp>  <glm/vec4.hpp> <vulkan/vulkan.h>)

if(WIN32)
  add_custom_command(TARGET engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> $<TARGET_FILE_DIR:engine>
    COMMAND_EXPAND_LISTS
    )
endif()
