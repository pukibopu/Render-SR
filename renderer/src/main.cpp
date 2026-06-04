#include <cstdio>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "error: no Metal device available on this system.\n");
        return 1;
    }

    std::printf("Metal device: %s\n", device->name()->utf8String());

    device->release();
    return 0;
}
