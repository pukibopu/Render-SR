// Tiny Cocoa/AppKit bridge. The only Objective-C++ unit in the renderer.
// Exposes a C-style function for the C++ side to obtain a CAMetalLayer
// attached to a GLFW window's NSWindow contentView.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct GLFWwindow;

// Attach a CAMetalLayer to the GLFW window's NSWindow contentView.
// Returns the layer as an opaque pointer. The C++ side casts this to
// CA::MetalLayer* (metal-cpp's wrapper has the same ObjC ABI as CAMetalLayer*).
// The returned layer is owned by the NSView; do not release it.
void* cocoa_bridge_attach_metal_layer(struct GLFWwindow* window);

#ifdef __cplusplus
}
#endif
