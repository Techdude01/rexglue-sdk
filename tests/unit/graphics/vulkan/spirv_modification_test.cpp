#include <catch2/catch_test_macros.hpp>

#include <rex/graphics/pipeline/shader/spirv_translator.h>

TEST_CASE("Diagnostic SPIR-V ABI revision separates shader cache keys", "[graphics][spirv]") {
  using Modification = rex::graphics::SpirvShaderTranslator::Modification;

  const Modification normal;
  Modification diagnostic;
  diagnostic.pixel.diagnostic_storage_version = 7;
  Modification census;
  census.pixel.diagnostic_storage_version = 6;
  Modification output_probe;
  output_probe.pixel.diagnostic_storage_version = 8;

  REQUIRE(normal.pixel.diagnostic_storage_version == 0);
  REQUIRE(normal.value != diagnostic.value);
  REQUIRE(diagnostic.value != census.value);
  REQUIRE(census.value != output_probe.value);

  Modification same_diagnostic(diagnostic.value);
  REQUIRE(same_diagnostic.pixel.diagnostic_storage_version == 7);
  REQUIRE(same_diagnostic.value == diagnostic.value);
  Modification same_census(census.value);
  REQUIRE(same_census.pixel.diagnostic_storage_version == 6);
  REQUIRE(same_census.value == census.value);
  Modification same_output_probe(output_probe.value);
  REQUIRE(same_output_probe.pixel.diagnostic_storage_version == 8);
  REQUIRE(same_output_probe.value == output_probe.value);

  Modification gradient_probe;
  gradient_probe.pixel.diagnostic_storage_version = 10;
  REQUIRE(gradient_probe.value != normal.value);
  REQUIRE(gradient_probe.value != output_probe.value);
  Modification gradient_only;
  gradient_only.pixel.diagnostic_storage_version = 11;
  REQUIRE(gradient_only.value != normal.value);
  REQUIRE(gradient_only.value != gradient_probe.value);
}
