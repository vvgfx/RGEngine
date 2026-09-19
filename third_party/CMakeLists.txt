find_package(Vulkan REQUIRED)

add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(vma INTERFACE)

add_library(stb_image INTERFACE)

add_subdirectory(fastgltf)
add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(SDL EXCLUDE_FROM_ALL)
add_subdirectory(box3d EXCLUDE_FROM_ALL)

target_sources(vkbootstrap PRIVATE
  vkbootstrap/VkBootstrap.h
  vkbootstrap/VkBootstrap.cpp
  )

target_include_directories(vkbootstrap PUBLIC vkbootstrap)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)
set_property(TARGET vkbootstrap PROPERTY CXX_STANDARD 20)

#both vma and glm and header only libs so we only need the include path
target_include_directories(vma INTERFACE vma)

target_include_directories(glm INTERFACE glm)

#add_library(sdl2 INTERFACE)

#target_include_directories(sdl2 INTERFACE $ENV{VULKAN_SDK}/Include/SDL2 )
#target_link_directories(sdl2 INTERFACE $ENV{VULKAN_SDK}/Lib )

#target_link_libraries(sdl2 INTERFACE SDL2 SDL2main)

add_library(imgui STATIC)

target_include_directories(imgui PUBLIC imgui)

target_sources(imgui PRIVATE
  imgui/imgui.h
  imgui/imgui.cpp

  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_widgets.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_impl_vulkan.cpp
  imgui/imgui_impl_sdl2.cpp
  )

# PUBLIC so every consumer agrees: imgui_internal.h hard-errors if this is not defined before
# imgui.h, and imGuIZMO pulls in imgui_internal.h. Setting it on the target means the order in
# which any .cpp includes the two headers stops mattering.
target_compile_definitions(imgui PUBLIC IMGUI_DEFINE_MATH_OPERATORS)

target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL2::SDL2)

target_include_directories(stb_image INTERFACE stb_image)

# imGuIZMO.quat: 3D direction widget, used for the sun. It draws entirely through ImGui's DrawList,
# so it is backend-agnostic and pulls in nothing from Vulkan.
#
# Only imGuIZMOquat.cpp is compiled: imguizmo_quat.cpp is a v4 alias that #includes it, so building
# both gives duplicate symbols.
add_library(imguizmo STATIC)

target_sources(imguizmo PRIVATE
  imGuIZMO/imGuIZMOquat.h
  imGuIZMO/imGuIZMOquat.cpp
  )

# CMAKE_CURRENT_SOURCE_DIR (third_party/) is on the path so the widget's default <imgui/imgui.h>
# resolves without patching the vendored config header.
target_include_directories(imguizmo PUBLIC imGuIZMO ${CMAKE_CURRENT_SOURCE_DIR})

# VGIZMO_USES_GLM swaps its bundled vgMath for glm, so it speaks the same vec3 the engine does.
#
# Do NOT add VGM_DISABLE_AUTO_NAMESPACE here: on the glm path vGizmo3D.h guards its own
# "using namespace glm;" on that macro alone, and its declarations use unqualified vec3/quat, so
# defining it fails to compile the library itself. The leak is contained by including
# imGuIZMOquat.h last in RGEngine.cpp instead.
target_compile_definitions(imguizmo PUBLIC VGIZMO_USES_GLM)

target_link_libraries(imguizmo PUBLIC imgui glm)
set_property(TARGET imguizmo PROPERTY CXX_STANDARD 20)
