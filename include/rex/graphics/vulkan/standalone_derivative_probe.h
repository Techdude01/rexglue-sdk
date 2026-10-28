// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef REX_GRAPHICS_VULKAN_STANDALONE_DERIVATIVE_PROBE_H_
#define REX_GRAPHICS_VULKAN_STANDALONE_DERIVATIVE_PROBE_H_

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

namespace rex::graphics::vulkan::diagnostics {

// The opt-in child stays in the caller's process group so the existing GPU
// runner can bound and clean it up. No executable or arguments come from the
// gate file or environment.
inline void RunStandaloneDerivativeProbeIfRequested() {
#if defined(__APPLE__)
  static std::atomic_flag invoked = ATOMIC_FLAG_INIT;
  if (invoked.test_and_set()) {
    return;
  }

  constexpr char kGatePath[] =
      "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/quad-partner-marker.txt";
  constexpr char kPython[] = "/usr/bin/python3";
  constexpr char kRunner[] =
      "/Users/Kushal/Projects/cosmicR/out/quad-partner-isolation-20261026/standalone/run_matrix.py";

  bool requested = false;
  std::ifstream gate(kGatePath);
  for (std::string line; std::getline(gate, line);) {
    if (line == "standalone_matrix=1") {
      requested = true;
      break;
    }
  }
  if (!requested) {
    return;
  }

  char* const arguments[] = {const_cast<char*>(kPython), const_cast<char*>(kRunner), nullptr};
  pid_t child = 0;
  const int spawn_error = posix_spawn(&child, kPython, nullptr, nullptr, arguments, environ);
  if (spawn_error != 0) {
    std::fprintf(stderr, "[standalone-derivative-probe] posix_spawn failed: %s\n",
                 std::strerror(spawn_error));
    return;
  }

  int child_status = 0;
  pid_t waited = 0;
  do {
    waited = waitpid(child, &child_status, 0);
  } while (waited < 0 && errno == EINTR);

  if (waited < 0) {
    std::fprintf(stderr, "[standalone-derivative-probe] waitpid failed: %s\n",
                 std::strerror(errno));
  } else if (WIFEXITED(child_status)) {
    std::fprintf(stderr, "[standalone-derivative-probe] matrix child exited: %d\n",
                 WEXITSTATUS(child_status));
  } else if (WIFSIGNALED(child_status)) {
    std::fprintf(stderr, "[standalone-derivative-probe] matrix child killed by signal: %d\n",
                 WTERMSIG(child_status));
  } else {
    std::fprintf(stderr, "[standalone-derivative-probe] matrix child ended unexpectedly\n");
  }
#endif
}

}  // namespace rex::graphics::vulkan::diagnostics

#endif  // REX_GRAPHICS_VULKAN_STANDALONE_DERIVATIVE_PROBE_H_
