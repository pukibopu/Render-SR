#include "NpyWriter.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rs::io {

namespace {

std::string shapeTuple(std::initializer_list<std::size_t> shape) {
    std::ostringstream os;
    os << "(";
    bool first = true;
    for (auto d : shape) {
        if (!first) os << ", ";
        os << d;
        first = false;
    }
    if (shape.size() == 1) os << ",";   // (N,) per NumPy convention
    os << ")";
    return os.str();
}

}

void writeNpyF32(const std::filesystem::path& path,
                 const float* data,
                 std::initializer_list<std::size_t> shape)
{
    if (shape.size() == 0) {
        throw std::runtime_error("writeNpyF32: shape must have at least one dim");
    }

    const std::string body =
        "{'descr': '<f4', 'fortran_order': False, 'shape': " +
        shapeTuple(shape) + ", }";

    // Pad header so the data begins on a 64-byte boundary.
    constexpr std::size_t kPreamble = 6 + 2 + 2;   // magic + version + header_len
    const std::size_t naive = kPreamble + body.size() + 1; // +1 for trailing '\n'
    const std::size_t aligned = ((naive + 63) / 64) * 64;
    const std::size_t pad = aligned - naive;
    const std::string header = body + std::string(pad, ' ') + '\n';

    if (header.size() > 0xFFFFu) {
        throw std::runtime_error("writeNpyF32: header too large for v1.0 format");
    }
    const std::uint16_t header_len_le = static_cast<std::uint16_t>(header.size());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("writeNpyF32: failed to open " + path.string());
    }

    // Magic + version 1.0
    out.write("\x93NUMPY", 6);
    const std::uint8_t major = 1, minor = 0;
    out.put(static_cast<char>(major));
    out.put(static_cast<char>(minor));

    // Header length (little-endian uint16). Apple Silicon is little-endian;
    // serialize byte-wise so this is portable anyway.
    out.put(static_cast<char>(header_len_le & 0xFF));
    out.put(static_cast<char>((header_len_le >> 8) & 0xFF));

    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    std::size_t count = 1;
    for (auto d : shape) count *= d;
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(count * sizeof(float)));

    if (!out) {
        throw std::runtime_error("writeNpyF32: write failed for " + path.string());
    }
}

}
