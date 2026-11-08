#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/work_purpose_audit.h>
using namespace rex::graphics::vulkan;

TEST_CASE("Purpose placement waits for a subsequent render in the same submission") {
  WorkPurposeAudit a;
  const WorkPurposeAudit::Key texture{WorkPurpose::kTextureGuest, 8, 4, 1, 0};
  const WorkPurposeAudit::Key upload{WorkPurpose::kSharedUpload, 1, 0, 0, 4096};
  a.BeginSubmission();
  a.Record(upload, true);
  a.Record(texture);
  a.BeginRender();
  a.Record(texture);
  a.Record(texture);
  REQUIRE(a.rows[texture].between == 0);
  a.BeginRender();
  a.Record(texture);
  a.EndSubmission();
  REQUIRE(a.rows[upload].prologue == 1);
  REQUIRE(a.rows[texture].count == 4);
  REQUIRE(a.rows[texture].before_render == 1);
  REQUIRE(a.rows[texture].between == 2);
  REQUIRE(a.rows[texture].tail == 1);
  REQUIRE(a.interrupted_gaps == 1);
  REQUIRE(a.gaps_by_cause[size_t(WorkPurpose::kTextureGuest)] == 1);
  a.BeginSubmission();
  a.BeginRender();
  a.EndSubmission();
  REQUIRE(a.interrupted_gaps == 1);
}

TEST_CASE("Multiple causes share one gap and same dimensions retain distinct purposes") {
  WorkPurposeAudit a;
  WorkPurposeAudit::Key load{WorkPurpose::kTextureGuest, 1, 1, 1, 0};
  WorkPurposeAudit::Key store{WorkPurpose::kHostDepthStore, 1, 1, 1, 0};
  a.BeginSubmission();
  a.BeginRender();
  a.Record(load);
  a.Record(store);
  a.BeginRender();
  a.BeginRender();
  a.EndSubmission();
  REQUIRE(a.interrupted_gaps == 1);
  REQUIRE(a.unattributed_gaps == 1);
  REQUIRE(a.gaps_by_cause[size_t(load.purpose)] == 1);
  REQUIRE(a.gaps_by_cause[size_t(store.purpose)] == 1);
  REQUIRE(a.rows.size() == 2);
  for (auto& [key, row] : a.rows)
    REQUIRE(row.count == row.prologue + row.before_render + row.between + row.tail);
  a.Reset();
  REQUIRE(a.rows.empty());
  REQUIRE(a.render_scopes == 0);
}
