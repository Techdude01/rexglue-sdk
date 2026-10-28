#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/string/buffer.h>

// The translator references the normal GPU debug-dump cvar and sample-position
// tables. Keep this offline harness independent of the full runtime flags/UI.
const std::string& FLAGS_dump_shaders_storage_() {
  static const std::string empty;
  return empty;
}
namespace rex::graphics::draw_util {
extern const int8_t kD3D10StandardSamplePositions2x[2][2] = {{4, 4}, {-4, -4}};
extern const int8_t kD3D10StandardSamplePositions4x[4][2] = {
    {-2, -6}, {6, -2}, {-6, 2}, {2, 6}};
}  // namespace rex::graphics::draw_util

int main(int argc, char** argv) {
  if (argc != 8 && argc != 9 && argc != 12) {
    std::cerr << "usage: spirv_declaration_harness <pixel-ucode> <normal.spv> "
                 "<triples.spv> <census.spv> <output-probe.spv> <normal-again.spv> "
                 "<gradient-probe.spv> [gradient-only-control.spv [no-entry.spv gate-only.spv no-sample-inputs.spv]]\n";
    return 2;
  }
  std::ifstream input(argv[1], std::ios::binary);
  if (!input)
    return 2;
  std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(input), {});
  if (bytes.empty() || bytes.size() % sizeof(uint32_t))
    return 2;
  std::vector<uint32_t> ucode(bytes.size() / sizeof(uint32_t));
  std::memcpy(ucode.data(), bytes.data(), bytes.size());

  rex::graphics::SpirvShader shader(rex::graphics::xenos::ShaderType::kPixel,
                                    UINT64_C(0x79A3448304FA0D22), ucode.data(), ucode.size(),
                                    std::endian::little);
  rex::string::StringBuffer disassembly;
  shader.AnalyzeUcode(disassembly);
  rex::graphics::SpirvShaderTranslator translator(
      rex::graphics::SpirvShaderTranslator::Features(true), false, false, false);

  auto translate = [&](uint32_t diagnostic_revision, const char* path) {
    using Modification = rex::graphics::SpirvShaderTranslator::Modification;
    Modification modification(translator.GetDefaultPixelShaderModification(0));
    modification.pixel.diagnostic_storage_version = diagnostic_revision;
    auto* translation = shader.GetOrCreateTranslation(modification.value);
    if (!translation || !translator.TranslateAnalyzedShader(*translation) ||
        !translation->is_valid() || translation->translated_binary().empty()) {
      std::cerr << "shader translation failed for diagnostic revision " << diagnostic_revision
                << '\n';
      return false;
    }
    std::ofstream output(path, std::ios::binary);
    const auto& binary = translation->translated_binary();
    output.write(reinterpret_cast<const char*>(binary.data()), binary.size());
    return bool(output);
  };

  // Reuse the same shader and translator to exercise Reset between variants.
  return translate(0, argv[2]) && translate(7, argv[3]) && translate(6, argv[4]) &&
                 translate(8, argv[5]) && translate(0, argv[6]) && translate(10, argv[7]) &&
                 (argc == 8 || translate(11, argv[8])) &&
                 (argc != 12 || (translate(12, argv[9]) && translate(13, argv[10]) && translate(14, argv[11])))
             ? 0
             : 1;
}
