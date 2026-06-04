cmake_minimum_required(VERSION 3.24)
project(render_sr LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT APPLE)
    message(FATAL_ERROR "Render-SR is macOS-only (Metal renderer).")
endif()

# Target Apple Silicon by default. macOS sometimes still defaults to x86_64 when
# invoked from /usr/local/bin tools, which then fails to link against arm64
# Homebrew libraries. Override only if the user has set CMAKE_OSX_ARCHITECTURES.
if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES arm64)
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)

# --- Dependencies -----------------------------------------------------------
find_package(glfw3 3.3 REQUIRED)

find_library(FOUNDATION_LIB Foundation REQUIRED)
find_library(METAL_LIB Metal REQUIRED)
find_library(QUARTZCORE_LIB QuartzCore REQUIRED)
find_library(APPKIT_LIB AppKit REQUIRED)

# --- metal-cpp headers ------------------------------------------------------
set(METAL_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp")
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
    message(FATAL_ERROR
        "metal-cpp headers not found at:\n"
        "    ${METAL_CPP_DIR}\n"
        "Run renderer/scripts/fetch_metal_cpp.sh and follow its instructions, "
        "then re-run cmake.")
endif()

# --- stb_image_write --------------------------------------------------------
set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")
if(NOT EXISTS "${STB_DIR}/stb_image_write.h")
    message(FATAL_ERROR
        "stb_image_write.h not found at:\n"
        "    ${STB_DIR}/stb_image_write.h\n"
        "Run renderer/scripts/fetch_stb.sh, then re-run cmake.")
endif()

# --- Renderer executable ----------------------------------------------------
add_executable(renderer
    src/main.cpp
    src/metal_impl.cpp
    src/app/Renderer.cpp
    src/app/GBufferPass.cpp
    src/app/Camera.cpp
    src/app/Mesh.cpp
    src/app/CocoaBridge.mm
    src/io/NpyWriter.cpp
    src/io/StbWriter.cpp
    src/io/FrameWriter.cpp
)

target_include_directories(renderer PRIVATE
    src
    shaders
    ${METAL_CPP_DIR}
    ${STB_DIR}
)

target_link_libraries(renderer PRIVATE
    glfw
    ${FOUNDATION_LIB}
    ${METAL_LIB}
    ${QUARTZCORE_LIB}
    ${APPKIT_LIB}
)

target_compile_options(renderer PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -Wno-gnu-anonymous-struct 
    -Wno-nested-anon-types
)

set_source_files_properties(src/app/CocoaBridge.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# stb_image_write.h is a vendored third-party single-header library; quiet the
# warnings it produces so they don't drown actual diagnostics.
set_source_files_properties(src/io/StbWriter.cpp PROPERTIES
    COMPILE_FLAGS "-Wno-missing-field-initializers -Wno-deprecated-declarations"
)

# --- Shaders -> default.metallib --------------------------------------------

option(RENDER_SR_BUILD_SHADERS "Compile .metal shaders into default.metallib" ON)

execute_process(
    COMMAND xcrun -sdk macosx metal --version
    RESULT_VARIABLE METAL_TOOL_FOUND
    OUTPUT_QUIET ERROR_QUIET
)
if(NOT METAL_TOOL_FOUND EQUAL 0)
    message(WARNING
        "Metal toolchain not found (`xcrun -sdk macosx -f metal` failed). "
        "Shader compilation will be skipped. Install with:\n"
        "    xcodebuild -downloadComponent MetalToolchain\n"
        "Then re-run cmake.")
    set(RENDER_SR_BUILD_SHADERS OFF)
endif()

if(RENDER_SR_BUILD_SHADERS)
    file(GLOB METAL_SHADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.metal)
    set(METAL_AIRS "")
    foreach(SHADER ${METAL_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air)
        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND xcrun -sdk macosx metal
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                    -c ${SHADER} -o ${AIR_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader ${SHADER_NAME}.metal"
            VERBATIM
        )
        list(APPEND METAL_AIRS ${AIR_FILE})
    endforeach()

    if(METAL_AIRS)
        set(METALLIB_OUTPUT ${CMAKE_BINARY_DIR}/default.metallib)
        add_custom_command(
            OUTPUT ${METALLIB_OUTPUT}
            COMMAND xcrun -sdk macosx metallib ${METAL_AIRS} -o ${METALLIB_OUTPUT}
            DEPENDS ${METAL_AIRS}
            COMMENT "Linking default.metallib"
            VERBATIM
        )
        add_custom_target(shaders ALL DEPENDS ${METALLIB_OUTPUT})
        add_dependencies(renderer shaders)
    endif()
endif()
