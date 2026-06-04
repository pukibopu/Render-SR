// Single translation unit defining the metal-cpp implementation macros.
// These MUST be defined in exactly one TU, before including the headers.
// Do NOT define MTLFX_PRIVATE_IMPLEMENTATION in v1 (no MetalFX use).

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
