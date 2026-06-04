#pragma once

#include <cstdint>
#include <filesystem>

namespace rs::io {

// Write an 8-bit PNG by stripping the alpha channel from a tightly packed
// RGBA8 image. `rgba` must point to `width * height * 4` bytes, row-major,
// no row padding. Returns true on success.
bool writePngRGBfromRGBA8(const std::filesystem::path& path,
                          const std::uint8_t* rgba,
                          int width, int height);

}
