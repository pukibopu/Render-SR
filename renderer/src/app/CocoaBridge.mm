#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include "CocoaBridge.h"

void* cocoa_bridge_attach_metal_layer(GLFWwindow* window) {
    NSWindow* nsWin = glfwGetCocoaWindow(window);
    if (!nsWin) {
        return nullptr;
    }

    NSView* view = [nsWin contentView];
    [view setWantsLayer:YES];

    CAMetalLayer* layer = [CAMetalLayer layer];
    [view setLayer:layer];

    // The NSView retains the layer (setLayer:), so it stays alive for the
    // view's lifetime. Return a non-retained opaque pointer to C++.
    return (__bridge void*)layer;
}
