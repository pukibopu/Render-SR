#pragma once

#include <cstddef>
#include <filesystem>
#include <initializer_list>

namespace rs::io {

// Write a contiguous row-major float32 array to `path` in NumPy .npy
// (format version 1.0, little-endian). `data` length must equal the product
// of `shape`. Throws std::runtime_error on I/O failure.
void writeNpyF32(const std::filesystem::path& path,
                 const float* data,
                 std::initializer_list<std::size_t> shape);

}
