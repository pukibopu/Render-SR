#include "StbWriter.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_ASSERT(x) ((void)0)
#include "stb_image_write.h"

#include <vector>

namespace rs::io {

bool writePngRGBfromRGBA8(const std::filesystem::path& path,
                          const std::uint8_t* rgba,
                          int width, int height)
{
    const std::size_t px = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint8_t> rgb(px * 3);
    for (std::size_t i = 0; i < px; ++i) {
        rgb[3*i+0] = rgba[4*i+0];
        rgb[3*i+1] = rgba[4*i+1];
        rgb[3*i+2] = rgba[4*i+2];
    }
    const int ok = stbi_write_png(
        path.string().c_str(),
        width, height,
        3,
        rgb.data(),
        width * 3);
    return ok != 0;
}

}
