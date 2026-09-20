# CMakeList.txt : CMake project for vulkan_guide, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

project ("vulkan_guide")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_subdirectory(src)

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)




file(GLOB_RECURSE GLSL_SOURCE_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )

foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  file(RELATIVE_PATH REL_PATH "${PROJECT_SOURCE_DIR}/shaders" ${GLSL})
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${REL_PATH}.spv")
  get_filename_component(SPIRV_DIR ${SPIRV} DIRECTORY)
  file(MAKE_DIRECTORY ${SPIRV_DIR})
  message(STATUS ${GLSL})
  # --depfile makes glslang report the .glsl files each shader #includes, and DEPFILE hands that
  # to the generator. Without it only the .frag/.vert/.comp is a dependency, so editing a shared
  # include (sky.glsl, light_grid.glsl, shadow_input.glsl) silently leaves stale SPIR-V behind --
  # the build reports success and the old shader keeps running.
  add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSL_VALIDATOR} -V --target-env vulkan1.3 -gVS ${GLSL} -o ${SPIRV} --depfile ${SPIRV}.d
        DEPENDS ${GLSL}
        DEPFILE ${SPIRV}.d)
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders
    DEPENDS ${SPIRV_BINARY_FILES}
    )

add_dependencies(engine Shaders)
