"""
Comprehensive 4C Multiphysics knowledge catalogue.

Based on systematic reading of ALL 4C source code.
73 modules, 40 problem types, 120+ materials, 130+ conditions, 20+ cell types.

This is the single source of truth for 4C domain knowledge in the Open FEM Agent.
"""

FOURC_KNOWLEDGE = {
    # ═══════════════════════════════════════════════════════════════════════
    # OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    "overview": {
        "description": "4C is a large-scale parallel C++20 multiphysics FEM code developed at TU Munich",
        "source": "$FOURC_ROOT/src/ (73 modules) — set FOURC_ROOT env var",
        "input_format": "YAML (.4C.yaml) with inline mesh or Exodus mesh references",
        "execution": "mpirun -np N $FOURC_BINARY input.4C.yaml (or just 4C if on PATH)",
        "output": "VTU via IO/RUNTIME VTK OUTPUT sections",
        "build": "CMake (cd build && cmake --build . -j$(nproc))",
        "modules": 73,
        "problem_types": 40,
        "material_models": "120+",
        "condition_types": "130+",
        "entrypoint_dispatch": {
            "source": "apps/global_full/4C_global_full_entrypoint_switch.cpp",
            "function": "entrypoint_switch()",
            "description": (
                "Authoritative Core::ProblemType -> solver-driver mapping. "
                "Every 4C input YAML's PROBLEM TYPE section selects ONE of "
                "these 36 enum values; misspellings hit the default arm "
                "and raise FOUR_C_THROW \"solution of unknown problemtype "
                "<X> requested\"."
            ),
            "problem_types": {
                "structure": "caldyn_drt()",
                "polymernetwork": "caldyn_drt()",
                "fluid": "dyn_fluid_drt(restart)",
                "fluid_redmodels": "dyn_fluid_drt(restart)",
                "lubrication": "lubrication_dyn(restart)",
                "ehl": "ehl_dyn()",
                "scatra": "scatra_dyn(restart)",
                "cardiac_monodomain": "scatra_cardiac_monodomain_dyn(restart)",
                "sti": "sti_dyn(restart)",
                "fluid_xfem": "fluid_xfem_drt(problem)",
                "fluid_ale": "fluid_ale_drt(problem)",
                "fsi": "fsi_ale_drt(problem)",
                "fsi_redmodels": "fsi_ale_drt(problem)",
                "fsi_xfem": "xfsi_drt(problem)",
                "fpsi_xfem": "xfpsi_drt(problem)",
                "gas_fsi": "fs3i_dyn()",
                "biofilm_fsi": "fs3i_dyn()",
                "thermo_fsi": "fs3i_dyn()",
                "fps3i": "fs3i_dyn()",
                "fbi": "fsi_immersed_drt(problem)",
                "ale": "dyn_ale_drt()",
                "thermo": "thermo_dyn_drt()",
                "tsi": "tsi_dyn_drt()",
                "loma": "loma_dyn(restart)",
                "elch": "elch_dyn(restart)",
                "art_net": "dyn_art_net_drt()",
                "red_airways": "dyn_red_airways_drt()",
                "reduced_lung": "ReducedLung::reduced_lung_main()",
                "one_d_pipe_flow": "ReducedLung1dPipeFlow::main()",
                "poroelast": "poroelast_drt()",
                "poroscatra": "poro_scatra_drt()",
                "porofluid_pressure_based": "porofluid_pressure_based_dyn(restart)",
                "porofluid_pressure_based_elast": "porofluid_elast_dyn(restart)",
                "porofluid_pressure_based_elast_scatra": "porofluid_pressure_based_elast_scatra_dyn(restart)",
                "fpsi": "fpsi_drt()",
                "ssi": "ssi_drt()",
                "ssti": "ssti_drt()",
                "particle": "particle_drt()",
                "pasi": "pasi_dyn()",
                "level_set": "levelset_dyn(restart)",
                "np_support": "MultiScale::np_support_drt()",
            },
            "Signal": (
                "[Input] Mis-typed PROBLEM TYPE in input YAML fails the "
                "switch-case in entrypoint_switch() and FOUR_C_THROWs "
                "with literal text 'solution of unknown problemtype "
                "<value> requested'. Common confusions: "
                "'fluid_struct_interaction' or 'fsi3d' instead of 'fsi'; "
                "'thermo_structural_interaction' instead of 'tsi'; "
                "'porous' instead of 'poroelast'. Check this table for "
                "the exact spelling. (File walk apps/global_full/"
                "4C_global_full_entrypoint_switch.cpp 2026-06-02.)"
            ),
        },
        "cli_arguments": {
            "source": "apps/global_full/4C_global_full_io.cpp",
            "description": (
                "Command-line arguments for nested-group parallelism and "
                "I/O configuration. Parsed in setup_global_problem() "
                "and validated by validate_argument_cross_compatibility()."
            ),
            "flags": {
                "--ngroup=N":           "Number of nested-parallelism groups (default 1).",
                "--glayout=N1,N2,...":  "Explicit per-group MPI rank counts. If omitted with --ngroup>1, ranks are split equally and FAIL if num_procs%n_groups!=0.",
                "--nptype=<type>":      "Nested parallelism type (mandatory when --ngroup>1).",
                "<input> <output>":     "Positional pair(s). Multiple pairs allowed when --nptype=separateInputFiles or nestedMultiscale.",
                "--parameters":         "Parameters-dump mode (skips io_pairs validation). When set, the cross-compat validator does NOT require <input> <output> count to match --ngroup.",
                "--diffgroup=N":        "Diff-mode group ID (default -1 = disabled). Used to compare outputs across nested groups.",
                "--interactive":        "Interactive mode (default false).",
                "--restart=N":          "Restart step (default 0 = no restart).",
                "--restartfrom=<id>":   "Output identifier to restart from. With nested parallelism, can be specified per-group via repeated flag.",
            },
            "commandline_arguments_struct_defaults": {
                # Source: apps/global_full/4C_global_full_io.hpp:38 CommandlineArguments
                "n_groups":                       1,
                "parameters":                     "false (set by --parameters)",
                "group_layout":                   "empty (auto-split if n_groups>1)",
                "nptype":                         "no_nested_parallelism",
                "diffgroup":                      -1,
                "restart":                        0,
                "restart_file_identifier":        "''",
                "restart_per_group":              "empty",
                "restart_identifier_per_group":   "empty",
                "interactive":                    False,
                "io_pairs":                       "empty",
                "input_file_name":                "''",
                "output_file_identifier":         "''",
            },
            "nptype_enum_values": {
                # snake_case (C++ enum) -> CLI-string alias (camelCase)
                "no_nested_parallelism":        "noNestedParallelism (default if --nptype omitted)",
                "every_group_read_input_file":  "everyGroupReadInputFile — single input, output gets _group_<N> suffix",
                "separate_input_files":         "separateInputFiles — N input/output pairs required",
                "nested_multiscale":            "nestedMultiscale — N input/output pairs required",
                "diffgroup0":                   "Sets nptype=no_nested_parallelism + diffgroup=0; first of two paired runs whose vectors/matrices/results will be diff'd",
                "diffgroup1":                   "Sets nptype=no_nested_parallelism + diffgroup=1; second of paired runs. Suffixes other than '0' or '1' rejected by CLI11 ValidationError",
            },
            "restart_special_values": {
                "last_possible":  "Sentinel string accepted by --restart; passes -1 internally, meaning 'restart from the last available checkpoint'",
                "<a>,<b>,<c>":    "Comma-separated per-group restart steps. Only meaningful with --nptype=separateInputFiles (one entry per group)",
            },
            "legacy_cli_syntax": {
                "description": (
                    "adapt_legacy_cli_arguments rewrites old-style invocations before CLI11 sees them — "
                    "two compat sets:"),
                "single_dash_legacy_names":  ["ngroup", "glayout", "nptype"],
                "nodash_legacy_names":       ["restart", "restartfrom"],
                "explanation": (
                    "single_dash: `4C -ngroup 4 ...` rewritten to `4C --ngroup=4 ...`. "
                    "nodash: `4C restart=10 input.yaml output.pre` rewritten to `4C --restart=10 input.yaml output.pre`. "
                    "Lets old scripts keep working but is invisible in --help."),
            },
            "build_options_affecting_runtime": {
                "FOUR_C_ENABLE_FE_TRAPPING": (
                    "If defined at compile time, main.cpp calls "
                    "feenableexcept(FE_INVALID | FE_DIVBYZERO) — the OS "
                    "kills the process via SIGFPE on the first NaN or "
                    "division-by-zero (informative message). Useful for "
                    "debugging numerical drift; will crash production runs "
                    "that intentionally use NaN sentinels."),
                "FOUR_C_ENABLE_CORE_DUMP": (
                    "If defined: run() is called WITHOUT a try/catch around "
                    "Core::Exception, so any throw triggers a core dump "
                    "(for post-mortem in gdb). If NOT defined (default): "
                    "Core::Exception is caught, stack trace printed via "
                    "err.what_with_stacktrace(), and MPI_Abort(MPI_COMM_WORLD, "
                    "EXIT_FAILURE) is called."),
            },
            "cmake_build_config_options": {
                "description": (
                    "Configure-time CMake cache variables defined in "
                    "cmake/setup_global_options.cmake. All take the "
                    "FOUR_C_* prefix and are surfaced via "
                    "four_c_process_global_option."),
                "options": {
                    "FOUR_C_BUILD_SHARED_LIBS": (
                        "bool, default ON. Force-syncs the legacy CMake "
                        "BUILD_SHARED_LIBS via FORCE cache writes if "
                        "only the legacy var is set — emits a CMake "
                        "WARNING in that case. Both names point to "
                        "the same value."),
                    "FOUR_C_ENABLE_DEVELOPER_MODE": (
                        "bool, default OFF. Optimizes the setup for "
                        "iterative development cycles."),
                    "FOUR_C_ENABLE_WARNINGS_AS_ERRORS": (
                        "bool, default OFF. Adds -Werror to the "
                        "private compile interface when ON."),
                    "FOUR_C_ENABLE_NATIVE_OPTIMIZATIONS": (
                        "bool, default OFF. Adds -march=native; "
                        "incompatible with portable binaries / "
                        "container images run on heterogeneous "
                        "hardware."),
                    "FOUR_C_ENABLE_ADDRESS_SANITIZER": (
                        "bool, default OFF. Adds -fsanitize=address "
                        "to both compile and link. FATAL_ERRORs at "
                        "configure time if the compiler+linker "
                        "don't accept the flag."),
                    "FOUR_C_ENABLE_COVERAGE": (
                        "bool, default OFF. LLVM source-based "
                        "coverage: -fprofile-instr-generate + "
                        "-fcoverage-mapping + -Wl,--build-id=sha1. "
                        "FATAL_ERROR at configure time if compiler "
                        "doesn't support."),
                    "FOUR_C_ENABLE_CORE_DUMP": (
                        "bool, default OFF. See "
                        "build_options_affecting_runtime."),
                    "FOUR_C_ENABLE_FE_TRAPPING": (
                        "bool, DEFAULT ON. Adds -ftrapping-math to "
                        "the compile interface. FATAL_ERRORs at "
                        "configure time if the compiler does not "
                        "support -ftrapping-math (most GCC/Clang "
                        "do; some Intel ICX / older Clang versions "
                        "don't). When OFF, instead adds "
                        "-fno-trapping-math. See "
                        "build_options_affecting_runtime for the "
                        "runtime behavior."),
                    "FOUR_C_ENABLE_IWYU": (
                        "bool, default OFF. Enables include-what-"
                        "you-use linter. FATAL_ERROR if iwyu "
                        "binary not found; user can override via "
                        "FOUR_C_IWYU_EXECUTABLE CMake variable."),
                    "FOUR_C_ENABLE_PYTHON_BINDINGS": (
                        "bool, default OFF. Builds the py4C "
                        "pybind11 bindings. Gated by cmake/"
                        "setup_py4C.cmake: requires BOTH "
                        "FOUR_C_BUILD_SHARED_LIBS=ON AND "
                        "FOUR_C_WITH_PYBIND11=ON, and is "
                        "INCOMPATIBLE with "
                        "FOUR_C_ENABLE_ADDRESS_SANITIZER=ON — "
                        "each violation FATAL_ERRORs at "
                        "configure time. The Python package "
                        "name written into the build dir is "
                        "literally 'py4C' (set via "
                        "FOUR_C_PYTHON_BINDINGS_PROJECT_NAME); "
                        "pyproject.toml.in and __init__.py.in "
                        "are configure_file'd from "
                        "utilities/py4C/src/config/ into "
                        "${PROJECT_BINARY_DIR}/py4C/."),
                    "FOUR_C_WITH_PYBIND11": (
                        "bool. Toggles the project-level "
                        "pybind11 dependency. Hard precondition "
                        "for FOUR_C_ENABLE_PYTHON_BINDINGS."),
                    "FOUR_C_ENABLE_ASSERTIONS": (
                        "bool, default OFF — but FORCE-set to ON "
                        "when build type is DEBUG (line 252-255: "
                        "explicit FORCE cache write). Adds "
                        "-D_GLIBCXX_ASSERTIONS for libstdc++ "
                        "assertions in addition to 4C's own "
                        "assertions."),
                    "FOUR_C_ENABLE_METADATA_GENERATION": (
                        "bool, default ON. Invokes Python after "
                        "build to generate metadata; requires "
                        "Python on the build host."),
                    "FOUR_C_ENABLE_LINKER_DETECTION": (
                        "bool, default ON. Defined in "
                        "cmake/checks/01_detect_linkers.cmake. "
                        "Probes `ld.mold`, `ld.lld`, `ld.gold`, "
                        "`ld.bfd` in that preference order via "
                        "find_program + four_c_check_compiles "
                        "with -fuse-ld=<name>. First linker that "
                        "passes the link-test wins. Populates "
                        "cache vars FOUR_C_LINKER_PROGRAM_<name> "
                        "(absolute path to ld.<name>) and "
                        "FOUR_C_LINKER_FUNCTIONAL_<name> (bool) "
                        "for each candidate tried. When OFF, the "
                        "user's manually-supplied linker flags "
                        "are used unchanged. If no linker is "
                        "functional, FATAL_ERROR with text "
                        "'Failed to find any working linker. "
                        "Please check your compiler and any "
                        "manually added flags.' OpenMPI / Ubuntu "
                        "20.04 quirk: faster linkers can fail "
                        "with mpic++; cmake then retries each "
                        "linker with `-lopen-pal` added "
                        "(populating FOUR_C_LINKER_FUNCTIONAL_"
                        "WITH_OPEN_PAL_<name>) before falling "
                        "back to the next candidate."),
                    "FOUR_C_CXX_FLAGS": (
                        "string, default empty. Expert setting; "
                        "additional C++ compile flags appended at "
                        "the END of the compile interface (so they "
                        "DO override earlier defaults). "
                        "separate_arguments-split."),
                    "FOUR_C_CXX_LINKER_FLAGS": (
                        "string, default empty. Expert setting; "
                        "additional linker flags appended at the "
                        "end."),
                },
                "build_type_optimization_flags": {
                    "DEBUG":          "-O0 -g (+ forces ENABLE_ASSERTIONS=ON)",
                    "RELEASE":        "-O3 -funroll-loops",
                    "RELWITHDEBINFO": "-O3 -g -funroll-loops",
                },
                "Signal": (
                    "[Performance] FOUR_C_ENABLE_FE_TRAPPING defaults "
                    "to ON in setup_global_options.cmake:195. Compilers "
                    "that don't accept -ftrapping-math (some Intel "
                    "ICX builds, older Clang on certain platforms) "
                    "FATAL_ERROR at CMake configure time with "
                    "'Option FOUR_C_ENABLE_FE_TRAPPING is ON but the "
                    "compiler does not support this feature. "
                    "Specifically, the compiler does not support "
                    "-ftrapping-math, which is necessary to generate "
                    "code that can safely use the floating-point "
                    "trapping mechanism.' Users on such compilers "
                    "must explicitly cmake -DFOUR_C_ENABLE_FE_TRAPPING="
                    "OFF. Plus three other configure-time pitfalls: "
                    "(a) DEBUG build type silently FORCEs "
                    "FOUR_C_ENABLE_ASSERTIONS=ON even if the user "
                    "explicitly passed -DFOUR_C_ENABLE_ASSERTIONS=OFF "
                    "(lines 251-255: explicit FORCE cache write with "
                    "'Forced ON due to build type DEBUG' help text); "
                    "(b) RELEASE / RELWITHDEBINFO build types add "
                    "-O3 + -funroll-loops directly to "
                    "four_c_private_compile_interface BEFORE "
                    "user FOUR_C_CXX_FLAGS — but FOUR_C_CXX_FLAGS is "
                    "appended at the END so it wins; CMAKE_CXX_FLAGS "
                    "by contrast is added in FRONT and cannot "
                    "override (file's own comment at lines 240-242 "
                    "spells this out); "
                    "(c) BUILD_SHARED_LIBS → FOUR_C_BUILD_SHARED_LIBS "
                    "migration emits a CMake WARNING but does NOT "
                    "fail — users following older 4C docs that "
                    "reference BUILD_SHARED_LIBS get a warning, "
                    "their value is force-synced into the new name, "
                    "and the build proceeds. "
                    "(d) [Integration] FOUR_C_ENABLE_PYTHON_BINDINGS=ON "
                    "has three configure-time prerequisites checked "
                    "by cmake/setup_py4C.cmake (NOT by setup_global_"
                    "options.cmake — easy to miss when reading only "
                    "the option declaration): "
                    "(i) FOUR_C_BUILD_SHARED_LIBS must be ON "
                    "(FATAL_ERROR: '4C Python bindings require to "
                    "build 4C with shared libraries (FOUR_C_BUILD_"
                    "SHARED_LIBS).'); "
                    "(ii) FOUR_C_WITH_PYBIND11 must be ON "
                    "(FATAL_ERROR: '4C Python bindings require to "
                    "build 4C with pybind11 (FOUR_C_WITH_PYBIND11).'); "
                    "(iii) FOUR_C_ENABLE_ADDRESS_SANITIZER must be "
                    "OFF (FATAL_ERROR: '4C Python bindings are "
                    "currently not compatible with an address "
                    "sanitizer build. Either set FOUR_C_ENABLE_"
                    "ADDRESS_SANITIZER=OFF or FOUR_C_ENABLE_PYTHON_"
                    "BINDINGS=OFF.'). The bindings build outputs a "
                    "pip-installable package at "
                    "${PROJECT_BINARY_DIR}/py4C/, with pyproject.toml "
                    "and __init__.py generated from "
                    "utilities/py4C/src/config/*.in templates. "
                    "(File walk cmake/setup_global_options.cmake + "
                    "cmake/setup_py4C.cmake 2026-06-03.) "
                    "(e) [Performance] When FOUR_C_ENABLE_LINKER_"
                    "DETECTION=ON (default), cmake/checks/"
                    "01_detect_linkers.cmake probes linkers in the "
                    "literal preference order mold > lld > gold > "
                    "bfd; first one that passes "
                    "four_c_check_compiles with "
                    "-fuse-ld=<name> wins. Source comment line "
                    "47-49 documents an OpenMPI / Ubuntu 20.04 "
                    "mpic++ wrapper bug: faster linkers can fail "
                    "with a missing-symbol error from "
                    "libopen-pal — cmake retries each linker with "
                    "`-lopen-pal` added (populating "
                    "FOUR_C_LINKER_FUNCTIONAL_WITH_OPEN_PAL_<name>) "
                    "before falling back to the next candidate. "
                    "To FORCE a specific linker, set "
                    "FOUR_C_ENABLE_LINKER_DETECTION=OFF and pass "
                    "-fuse-ld=<name> via FOUR_C_CXX_LINKER_FLAGS — "
                    "the detection block is skipped entirely. "
                    "When all four candidates fail, configure "
                    "aborts with FATAL_ERROR 'Failed to find any "
                    "working linker. Please check your compiler "
                    "and any manually added flags.' (File walk "
                    "cmake/checks/01_detect_linkers.cmake "
                    "2026-06-03.)"
                ),
            },
            "cmake_test_setup_options": {
                "description": (
                    "Configure-time CMake options + derived "
                    "constants defined in cmake/setup_tests.cmake "
                    "— controls GoogleTest unit-test fetching, "
                    "Google Benchmark micro-benchmark fetching, "
                    "and the test-timeout scaling system."),
                "options": {
                    "FOUR_C_TEST_TIMEOUT_SCALE": (
                        "STRING (integer-valued), default 4 when "
                        "FOUR_C_BUILD_TYPE_UPPER==DEBUG else 1. "
                        "Multiplier applied to all test timeouts. "
                        "Silent 4× scaling in Debug builds — easy "
                        "to miss when comparing CI durations "
                        "between Debug and Release."),
                    "FOUR_C_WITH_GOOGLETEST": (
                        "bool, default ON. Toggles GoogleTest "
                        "v1.15.2 FetchContent (pinned commit "
                        "b514bdc898e2951020cbdca1304b75f5950d1f59) "
                        "and the `unittests` custom target. The "
                        "`full` target depends on `unittests`."),
                    "FOUR_C_WITH_GOOGLE_BENCHMARK": (
                        "bool, default OFF. Toggles Google "
                        "Benchmark v1.9.2 FetchContent (pinned "
                        "commit afa23b7699c17f1e26c88cbf95257b20d"
                        "78d6247) and the `benchmarktests` custom "
                        "target. Implicitly sets "
                        "BENCHMARK_ENABLE_TESTING=OFF to skip "
                        "google-benchmark's own internal tests."),
                    "FOUR_C_ENABLE_FULL_BENCHMARK_TESTS": (
                        "bool, default OFF, ONLY visible when "
                        "FOUR_C_WITH_GOOGLE_BENCHMARK=ON. OFF "
                        "means dry-run mode (10s timeout); ON "
                        "means real benchmark execution (600s "
                        "timeout × FOUR_C_TEST_TIMEOUT_SCALE)."),
                    "FOUR_C_BENCHMARK_TESTS_COLLECTION_FILE": (
                        "PATH, default ${PROJECT_BINARY_DIR}/"
                        "benchmark_test_results.json. Output JSON "
                        "where 4C aggregates benchmark results via "
                        "four_c_collect_benchmark_test_results."),
                    "FOUR_C_ENABLE_FULL_PERFORMANCE_TESTS": (
                        "bool, default OFF. Switches performance "
                        "tests between full and minimal execution. "
                        "Distinct from FOUR_C_ENABLE_FULL_BENCHMARK_"
                        "TESTS — performance tests are 4C-internal, "
                        "benchmark tests use Google Benchmark."),
                    "FOUR_C_PERFORMANCE_TESTS_COLLECTION_FILE": (
                        "PATH, default ${PROJECT_BINARY_DIR}/"
                        "performance_test_results.json."),
                },
                "derived_constants": {
                    "FOUR_C_TEST_GLOBAL_TIMEOUT": (
                        "120 * FOUR_C_TEST_TIMEOUT_SCALE — global "
                        "ctest timeout floor."),
                    "UNITTEST_TIMEOUT": (
                        "10 * FOUR_C_TEST_TIMEOUT_SCALE — per-"
                        "unit-test timeout (set when "
                        "FOUR_C_WITH_GOOGLETEST=ON)."),
                    "BENCHMARK_TEST_TIMEOUT": (
                        "10 (dry-run) or 600 (full) * "
                        "FOUR_C_TEST_TIMEOUT_SCALE."),
                    "FOUR_C_INSTALL_PREFIX": (
                        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_"
                        "DATADIR}/cmake/4C — where the install-"
                        "test harness expects 4CConfig.cmake."),
                },
                "install_test_configure_files": (
                    "Three .in templates configure_file'd at "
                    "configure time into ${PROJECT_BINARY_DIR}/"
                    "tests/install_test/: main.cpp, CMakeLists.txt, "
                    "test_install.sh — used by CI to verify the "
                    "installed 4CConfig.cmake works for downstream "
                    "consumers."),
                "Signal": (
                    "[Integration] cmake/setup_tests.cmake "
                    "FetchContent's GoogleTest at PINNED COMMIT "
                    "b514bdc898e2951020cbdca1304b75f5950d1f59 "
                    "(v1.15.2) and Google Benchmark at PINNED "
                    "COMMIT afa23b7699c17f1e26c88cbf95257b20d78d6247 "
                    "(v1.9.2). Two FATAL_ERROR guards catch "
                    "TARGET-name clashes when 4C is embedded in a "
                    "larger CMake project that has already pulled "
                    "in either library: "
                    "  if(TARGET gtest) → FATAL_ERROR 'A target "
                    "<gtest> has already been included by another "
                    "library. This is not supported.' "
                    "  if(TARGET benchmark_main) → FATAL_ERROR 'A "
                    "target <benchmark_main> has already been "
                    "included by another library. This is not "
                    "supported.' "
                    "Workarounds when integrating 4C into an "
                    "umbrella project: (a) set "
                    "FOUR_C_WITH_GOOGLETEST=OFF and/or "
                    "FOUR_C_WITH_GOOGLE_BENCHMARK=OFF to skip the "
                    "fetch entirely (the parent project must then "
                    "not link against 4C's unit-test executables); "
                    "(b) override the FetchContent source via "
                    "FETCHCONTENT_SOURCE_DIR_GOOGLETEST and "
                    "FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK to "
                    "point at the parent project's pre-included "
                    "copies — note that the FATAL_ERROR fires "
                    "BEFORE the FetchContent_MakeAvailable call, "
                    "so source-dir override alone is not sufficient. "
                    "[Performance] FOUR_C_TEST_TIMEOUT_SCALE "
                    "defaults to 4 in DEBUG builds (line 8-12) "
                    "vs 1 in Release/RelWithDebInfo. This silently "
                    "quadruples ALL test timeouts (UNITTEST_TIMEOUT "
                    "40s, GLOBAL_TIMEOUT 480s, BENCHMARK_TIMEOUT "
                    "2400s in full-benchmark mode) when the user "
                    "switches between Debug and Release without "
                    "changing CMakeCache.txt. CI run-time diffs "
                    "between Debug and Release jobs often surface "
                    "this. (File walk cmake/setup_tests.cmake "
                    "2026-06-03.)"
                ),
            },
            "cmake_dependency_configure_pattern": {
                "description": (
                    "Pattern shared by cmake/configure/configure_"
                    "<Dep>.cmake files (one per external "
                    "dependency: ArborX, Backtrace, Boost, "
                    "CLI11, CLN, FFTW, HDF5, MIRCO, MPI, Qhull, "
                    "Trilinos, VTK, ZLIB, gmsh, deal.II, "
                    "magic_enum, pybind11, ryml). Each file "
                    "controls HOW to obtain that dependency at "
                    "configure time; the higher-level "
                    "FOUR_C_WITH_<Dep> toggle (cmake_install_"
                    "export.dependency_toggles_FOUR_C_WITH_*) "
                    "controls WHETHER to use it at all."),
                "shape": (
                    "(1) Declares a FOUR_C_<DEP>_FIND_INSTALLED "
                    "boolean option (default usually OFF) via "
                    "four_c_process_global_option. "
                    "(2) When ON: find_package(<Dep> HINTS "
                    "${FOUR_C_<DEP>_ROOT}); FATAL_ERROR with a "
                    "per-dep message if find fails. "
                    "(3) When OFF: fetchcontent_declare/"
                    "_makeavailable from a PINNED commit hash, "
                    "then sets FOUR_C_<DEP>_ROOT to "
                    "${CMAKE_INSTALL_PREFIX}. "
                    "(4) four_c_remember_variable_for_install on "
                    "both FOUR_C_<DEP>_FIND_INSTALLED + "
                    "FOUR_C_<DEP>_ROOT so downstream consumers' "
                    "4CConfig.cmake replays the same choice."),
                "Signal": (
                    "[Integration] Two-layer toggle structure to "
                    "be aware of: FOUR_C_WITH_<Dep>=ON enables "
                    "the dependency at all, then FOUR_C_<DEP>_"
                    "FIND_INSTALLED=ON/OFF controls find-vs-fetch. "
                    "Wanting to use a system-installed library "
                    "but forgetting to set FIND_INSTALLED=ON "
                    "silently triggers a fetch+build of a pinned "
                    "vendored version, doubling configure time "
                    "and producing two copies of the dep in the "
                    "build tree. Pinned-commit fetch fallback "
                    "uses fetchcontent_declare + "
                    "fetchcontent_makeavailable; each dep's "
                    "configure_<Dep>.cmake has its own "
                    "GIT_REPOSITORY + GIT_TAG commit hash hard-"
                    "coded (e.g. ArborX is pinned to "
                    "f9244ba03904cc518a54d99e9f87bb42dc9ecaf3 = "
                    "v2.0.1, ARBORX_ENABLE_MPI=ON unconditionally "
                    "forced). To switch fetch source: override "
                    "FETCHCONTENT_SOURCE_DIR_<dep> before "
                    "fetchcontent_makeavailable. (File walk "
                    "cmake/configure/configure_ArborX.cmake "
                    "2026-06-03; pattern verified once, applies "
                    "to all 18 configure_<Dep>.cmake files.)"
                ),
            },
            "cmake_install_export": {
                "description": (
                    "Configure-time surfaces defined in "
                    "cmake/setup_install.cmake — install rules, "
                    "exported 4CTargets, and the 4CConfig.cmake "
                    "consumer file."),
                "exported_targets_namespace": "4C::",
                "config_file_destination": (
                    "${CMAKE_INSTALL_DATADIR}/cmake/4C/4CConfig.cmake"
                    " (plus 4CConfigVersion.cmake)"),
                "version_compatibility": "ExactVersion",
                "dependency_toggles_FOUR_C_WITH_*": [
                    "HDF5", "MPI", "Qhull", "Trilinos", "VTK",
                    "gmsh", "deal.II", "Boost", "ArborX", "FFTW",
                    "CLN", "MIRCO", "Backtrace", "ryml",
                    "magic_enum", "ZLIB", "pybind11", "CLI11",
                ],
                "rolled_up_dependency_target": (
                    "four_c_all_enabled_external_dependencies — "
                    "single CMake target rolling up every "
                    "FOUR_C_WITH_<X>=ON external; downstream "
                    "consumers link via 4C::lib4C only."),
                "Signal": (
                    "[Input] 4CConfig.cmake exports the package "
                    "with COMPATIBILITY ExactVersion (line 100 of "
                    "setup_install.cmake). Downstream "
                    "find_package(4C <version> EXACT) requires "
                    "EXACT FOUR_C_VERSION_MAJOR.MINOR match — "
                    "find_package(4C 1.3) when 4C is installed at "
                    "1.4 FAILS with 'incompatible version' even "
                    "though they may be API-compatible. Use "
                    "find_package(4C) (no version pin) to fall "
                    "back to whatever is installed, or pin to the "
                    "EXACT installed MAJOR.MINOR. The 18-package "
                    "FOUR_C_WITH_<X> boolean surface (HDF5 / MPI "
                    "/ Qhull / Trilinos / VTK / gmsh / deal.II / "
                    "Boost / ArborX / FFTW / CLN / MIRCO / "
                    "Backtrace / ryml / magic_enum / ZLIB / "
                    "pybind11 / CLI11) is set by the parent "
                    "build's FOUR_C_WITH_<X> CMake cache values "
                    "and baked into the exported config — "
                    "downstream cannot RE-enable a dep that was "
                    "OFF at 4C install time. The "
                    "four_c_all_enabled_external_dependencies "
                    "rolled-up target is the canonical downstream "
                    "link edge; downstream projects do "
                    "target_link_libraries(myapp PRIVATE "
                    "4C::lib4C) and inherit the dependency "
                    "transitively. (File walk "
                    "cmake/setup_install.cmake 2026-06-03.)"
                ),
            },
            "additional_io_input_keys": {
                "WRITE_TIMINGS": (
                    "bool — when true, run() writes "
                    "`<output_file_identifier>-timings.yaml` via export_timings() "
                    "after the simulation completes. Read from io_params."),
            },
            "memory_high_water_mark_summary": (
                "After run() completes, main calls get_memory_high_water_mark(comm) "
                "which reads /proc/self/status for VmHWM, MPI-gathers, and prints "
                "'Memory High Water Mark Summary: MinOverProcs [PID] / MeanOverProcs / "
                "MaxOverProcs [PID] / SumOverProcs' in GB. LINUX-ONLY — guarded by "
                "#if defined(__linux__); macOS/Windows runs print 'Memory High Water "
                "Mark summary not available on this operating system.' instead. "
                "Failure to open /proc/self/status (e.g. namespace restrictions in "
                "containers) prints a friendlier 'status file could not be opened "
                "on every proc.' rather than failing — does NOT abort the run."),
            "io_input_keys": {
                # YAML/dat keys read from the input file's I/O block by setup_parallel_output
                "WRITE_TO_SCREEN":     "bool — stream Core::IO::cout to stdout",
                "WRITE_TO_FILE":       "bool — stream Core::IO::cout to log file",
                "PREFIX_GROUP_ID":     "bool — prepend group ID to each line",
                "LIMIT_OUTP_TO_PROC":  "int — limit per-rank output to this MPI rank only",
                "VERBOSITY":           "Core::IO::Verbositylevel enum (e.g. verbose, standard, minimal)",
            },
            "Signal": (
                "[Input] CLI validation in validate_argument_cross_compatibility() "
                "raises FOUR_C_THROW with literal text messages — these are "
                "the verbatim error strings:\n"
                "  - 'When --glayout is provided its number of entries must "
                "equal --ngroup.'\n"
                "  - 'when --ngroup > 1, a nested parallelism type must be "
                "specified via --nptype.'\n"
                "  - 'when using \\'no_nested_parallelism\\' or "
                "\\'everyGroupReadInputFile\\' the number of <input> <output> "
                "pairs must be exactly 1.'\n"
                "  - 'when using \\'separateInputFiles\\' or "
                "\\'nestedMultiscale\\' the number of <input> <output> pairs "
                "must equal --ngroup ...'\n"
                "  - 'When using --nptype other than \\'separateInputFiles\\', "
                "only one restart step and one restartfrom identifier must be given.'\n"
                "  - 'You need to specify a restart step when using restartfrom.'\n"
                "  - 'Positional arguments must be provided as pairs: <input> <output>.'\n"
                "Mixed naming: ENUM values are snake_case in C++ "
                "(no_nested_parallelism) but the CLI string parses the "
                "camelCase ALIAS (everyGroupReadInputFile / separateInputFiles "
                "/ nestedMultiscale). The error messages quote BOTH forms in "
                "the same sentence — user-facing inconsistency. (File walk "
                "apps/global_full/4C_global_full_io.cpp 2026-06-02.)"
            ),
            "output_naming_under_groups": (
                "When --nptype=everyGroupReadInputFile is set, output_identifier "
                "gets _group_<N> suffix appended. If the user's identifier "
                "already ends with -<num> (e.g. 'mysim-001'), the suffix is "
                "inserted as 'mysim_group_<N>_001'. Restart identifier follows "
                "the same convention. Source: update_io_identifiers() switch case."
            ),
        },
        "post_monitor_tool": {
            "description": (
                "The standalone post_monitor CLI binary extracts time-history "
                "of a single node into an ASCII .mon file. Source: "
                "apps/post_monitor/4C_post_monitor.cpp main()."),
            "cli_arguments": {
                "--node": "Required int. Global node id whose history to dump.",
                "--field": ("String, default 'fluid'. Selects which "
                            "discretization the node belongs to. Valid "
                            "vocabulary per ProblemType is hard-coded "
                            "in the main() switch — see "
                            "supported_field_per_problem below."),
            },
            "output_file_suffixes": {
                ".mon": "primary fields per write_mon_file()",
                ".stress.mon": "Cauchy + 2nd-PK stresses",
                ".strain.mon": "Green-Lagrange / Euler-Almansi / Log strains",
                ".plasticstrain.mon": "plastic GL / plastic EA strains",
                ".heatflux.mon": "thermo current + initial heatfluxes (thermo only)",
                ".tempgrad.mon": "thermo current + initial tempgrads (thermo only)",
            },
            "stresstype_straintype_heatfluxtype_enum": [
                "none",
                "ndxyz",
            ],
            "supported_field_per_problem": {
                "fsi / fsi_redmodels": ["fluid", "structure"],
                "structure / loma / fluid / fluid_redmodels / fps3i":
                    ["scatra", "fluid", "structure"],
                "ale": ["ale"],
                "thermo": ["thermo"],
                "red_airways": ["red_airway"],
                "poroelast": ["fluid", "structure"],
                "porofluid_pressure_based": ["porofluid"],
                "porofluid_pressure_based_elast":
                    ["structure", "porofluid"],
                "porofluid_pressure_based_elast_scatra":
                    ["structure", "porofluid", "scatra", "artery_scatra"],
            },
            "stress_strain_groupnames_at_write_time": {
                "stress": ["gauss_cauchy_stresses_xyz", "gauss_2PK_stresses_xyz"],
                "strain": ["gauss_GL_strains_xyz", "gauss_EA_strains_xyz",
                           "gauss_LOG_strains_xyz"],
                "plastic_strain": ["gauss_pl_GL_strains_xyz",
                                   "gauss_pl_EA_strains_xyz"],
                "heatflux": ["gauss_current_heatfluxes_xyz",
                             "gauss_initial_heatfluxes_xyz"],
                "tempgrad": ["gauss_initial_tempgrad_xyz",
                             "gauss_current_tempgrad_xyz"],
            },
            "Signal": (
                "[Output] Seven sharp edges users routinely hit running "
                "post_monitor: "
                "(1) SERIAL-ONLY tool. The source-file header comment says "
                "'Works in seriell version only! Requires to read one "
                "instance of the discretisation!!'; the body counts node "
                "owners across all ranks and FOUR_C_THROW('Found more than "
                "one owner of node {}: {}') if more than one rank owns the "
                "node. Running with mpirun -n>1 errors out at the first "
                "node lookup. "
                "(2) The stresstype / straintype / heatfluxtype enum is "
                "EXACTLY {'none', 'ndxyz'} — any other value (e.g. 'cxyz', "
                "'averaged', '123') triggers FOUR_C_THROW('Cannot deal "
                "with requested <kind> output type: {}'). Common confusion: "
                "4C output formats elsewhere use 'cxyz' for cell-based, but "
                "post_monitor accepts only the nodal 'ndxyz' form. "
                "(3) FSI + --field=ale is explicitly REJECTED with "
                "FOUR_C_THROW('There is no ALE output. Displacements of "
                "fluid nodes can be printed.') — there's even a leftover "
                "FsiAleMonWriter ctor call after the throw that is dead "
                "code. Use --field=fluid to get the fluid-side ALE "
                "displacements. "
                "(4) ProblemType red_airways silently NO-OPS if "
                "--field != 'red_airway'. The main() case has an if-check "
                "and no else clause — wrong field value writes no .mon "
                "file and prints no error. "
                "(5) Stress / strain time-history is DEAD CODE in this "
                "tool. write_mon_stress_file, write_mon_strain_file, "
                "write_mon_pl_strain_file are defined on MonWriter but "
                "main() never calls them — only thermo's heatflux and "
                "tempgrad are auto-invoked. The .stress.mon / .strain.mon "
                "files are produced only if the user calls the methods "
                "programmatically, NOT from the CLI. "
                "(6) CLI default --field=fluid is silently applied to "
                "structural / thermo / scatra runs where it would be "
                "rejected — easy oversight: the first FOUR_C_THROW the "
                "user sees is 'Node {} does not belong to fluid field!' "
                "even though they're running a structure problem. "
                "(7) ProblemType gas_fsi / biofilm_fsi / thermo_fsi are "
                "in the dispatch but throw FOUR_C_THROW('not implemented "
                "yet') — the tool's coverage is narrower than the full "
                "ProblemType set. Default unknown ProblemType triggers "
                "FOUR_C_THROW('problem type {} not yet supported'). "
                "(File walk apps/post_monitor/4C_post_monitor.cpp "
                "2026-06-03.)"
            ),
        },
        "post_processor_tool": {
            "description": (
                "The standalone post_processor CLI binary — the bigger "
                "sibling of post_monitor. Reads native 4C output and "
                "writes per-field visualization files (Ensight .case or "
                "ParaView VTU/VTI). Source: apps/post_processor/"
                "4C_post_processor.cpp main() + run_ensight_vtu_filter()."),
            "cli_arguments": {
                "--filter": ("String, default 'ensight'. CASE-SENSITIVE "
                             "enum: {'ensight', 'vtu', 'vtu_node_based', "
                             "'vti'}. Any other value FOUR_C_THROWs "
                             "'Unknown filter {} given, supported "
                             "filters: [ensight|vtu|vti]'."),
            },
            "supported_problem_types_in_dispatch": [
                "fsi", "fsi_redmodels", "gas_fsi", "thermo_fsi",
                "biofilm_fsi", "structure", "polymernetwork",
                "fluid", "fluid_redmodels", "fluid_ale",
                "particle", "pasi", "ale", "lubrication",
                "cardiac_monodomain", "scatra",
                "fsi_xfem", "fpsi_xfem", "fluid_xfem",
                "loma", "elch", "art_net", "thermo",
                "red_airways", "poroelast", "poroscatra",
                "fpsi", "fbi", "fps3i", "ehl", "none",
            ],
            "filter_writer_classes": [
                "StructureFilter (also used for art_net, red_airways)",
                "FluidFilter (also used for porofluid)",
                "XFluidFilter (XFEM-only)",
                "AleFilter",
                "MortarFilter (structure problem with do_mortar_interfaces)",
                "InterfaceFilter (fsi_xfem boundary discretizations)",
                "ThermoFilter (uses heatfluxtype + tempgradtype)",
                "LubricationFilter",
                "AnyFilter (ProblemType::none — write whatever vectors exist)",
            ],
            "filter_result_tags_per_writer": {
                "StructureFilter": (
                    "~50 tags. Primary: displacement, prolongated_"
                    "gauss_2PK_stresses_xyz, prolongated_gauss_GL_"
                    "strains_xyz, material_displacement (if struct_"
                    "mat_disp='yes'). Contact: activeset, contact"
                    "owner, nor/tan contactstress, slave/master"
                    "forces (+lm/g suffixes), interfacetraction, "
                    "wear, poronopen_lambda. Spring/dashpot: gap, "
                    "curnormals, springstress. Error norms: L2_norm, "
                    "H1_norm, Energy_norm. 1D artery: one_d_artery_"
                    "{pressure,flow,area}, forward/backward speed[0]. "
                    "Reduced airway: pnp/p_nonlin, NodeIDs, radii, "
                    "scatraO2np, PO2, dVO2, AcinarPO2, acini_vnp, "
                    "qin_np/qout_np, x_np, open, p_extnp/n, generations, "
                    "elemVolume[0]np, elemRadius_current. FSI: "
                    "Add_Forces, fsilambda, fpilambda_ps/pf. Biofilm: "
                    "str_growth_displ. Poro: porosity_p1. SSI: nodal_"
                    "stresses_xyz. EHL: fluid_force, normal/tangential_"
                    "contact, active, slip. Plus Gauss-point post-stress "
                    "and rotation R."),
                "FluidFilter": (
                    "~40 tags. Primary: velnp (-> 'velocity'), pressure, "
                    "scalar_field, residual. Averaged: averaged_pressure/"
                    "velnp/scanp. Filtered: filteredvel, fsvelaf. ALE: "
                    "dispnp, idispnfull, traction. Wall-shear: wss, "
                    "wss_mean. XWall: xwall_enrvelnp, xwall_tauw, par_vel. "
                    "FSI volume-constraint: Add_Forces. Poro: convel, "
                    "gridv. Adjoint: adjoint_velnp/pressure. Meshfree: "
                    "velatmeshfreenodes, pressureatmeshfreenodes. FSI "
                    "Lagrange mul.: fsilambda. Biofilm: fld_growth_displ. "
                    "HDG: velnp_hdg, pressure_hdg, tracevel_hdg. XFluid "
                    "level-set: fluid_levelset_boundary + phinp_0..19."),
                "XFluidFilter": (
                    "5 tags ONLY (XFEM-specific naming): velocity_"
                    "smoothed, pressure_smoothed, averaged_velnp, "
                    "averaged_pressure, fsvelocity."),
                "MortarFilter": (
                    "8 tags: displacement, nor/tan contactstress, "
                    "interface traction, slave/master forces (+nor/tan "
                    "suffixes)."),
                "InterfaceFilter": (
                    "7 tags (interface-side FSI accessors): idispnp/n, "
                    "ivelnp/n/nm, iaccn, itrueresnp."),
                "AleFilter": (
                    "3 tags: dispnp (-> 'displacement'), det_j, "
                    "element_quality."),
                "LubricationFilter": (
                    "5 tags: prenp (-> 'pressure'), height, no_gap_DBC, "
                    "dispnp, viscosity."),
                "ThermoFilter": (
                    "Primary: temperature (NOT 'tempnp'). Optional "
                    "Gauss-point post: gauss_{current,initial}_heatfluxes_"
                    "xyz → 'heatflux' (nodebased), gauss_{current,initial}_"
                    "tempgrad_xyz → 'tempgrad' (nodebased). Plus "
                    "displacement (TSI), and SLM-specific: phase, "
                    "conductivity, capacity."),
                "AnyFilter": (
                    "Writes all dof + node + element results blindly "
                    "(no tag whitelist)."),
            },
            "structure_filter_one_time_step_subset": (
                "StructureFilter::write_all_results_one_time_step "
                "(line 173) writes ONLY displacement + node results, "
                "NOT the full ~50-tag set. Used for partial restart-"
                "style writes. Users expecting stresses/strains/contact "
                "tags from a per-step call get only displacement."),
            "structure_stress_filter_internals": {
                "post_stress_stresstype_enum": [
                    "ndxyz",
                    "cxyz",
                    "cxyz_ndxyz",
                    "nd123",
                    "c123",
                    "c123_nd123",
                ],
                "post_stress_dispatch": {
                    "ndxyz": "write_stress(..., nodebased)",
                    "cxyz": "write_stress(..., elementbased)",
                    "cxyz_ndxyz": (
                        "write_stress(..., nodebased) then PostResult "
                        "reset then write_stress(..., elementbased) "
                        "— dual write"),
                    "nd123": "write_eigen_stress(..., nodebased)",
                    "c123": "write_eigen_stress(..., elementbased)",
                    "c123_nd123": (
                        "write_eigen_stress(..., nodebased) then "
                        "PostResult reset then write_eigen_stress("
                        "..., elementbased) — dual write"),
                },
                "special_field_subclasses_in_file": [
                    "WriteNodalStressStep (6-component nodal stress, "
                    "via Core::FE::extrapolate_gauss_point_quantity_"
                    "to_nodes)",
                    "WriteElementCenterStressStep (6-component element-"
                    "center stress, via Core::FE::evaluate_gauss_point_"
                    "quantity_at_element_center)",
                    "WriteElementCenterRotation (9-component element-"
                    "center rotation tensor R, only triggered when "
                    "groupname=='rotation'; comment 'pfaller may17')",
                    "WriteNodalEigenStressStep (num_df_map = "
                    "{1,1,1,3,3,3} — 3 eigenvalues + 3 eigenvector "
                    "columns × 3 components; uses symmetric_eigen_"
                    "problem)",
                    "WriteElementCenterEigenStressStep (same shape as "
                    "WriteNodalEigenStressStep but at element centers)",
                ],
                "write_stress_groupname_vocab": [
                    "gauss_2PK_stresses_xyz",
                    "gauss_cauchy_stresses_xyz",
                    "gauss_GL_strains_xyz",
                    "gauss_EA_strains_xyz",
                    "gauss_LOG_strains_xyz",
                    "gauss_pl_GL_strains_xyz",
                    "gauss_pl_EA_strains_xyz",
                    "rotation",
                ],
                "write_eigen_stress_groupname_vocab": [
                    "gauss_2PK_stresses_xyz",
                    "gauss_cauchy_stresses_xyz",
                    "gauss_GL_strains_xyz",
                    "gauss_EA_strains_xyz",
                    "gauss_LOG_strains_xyz",
                    "gauss_pl_GL_strains_xyz",
                    "gauss_pl_EA_strains_xyz",
                ],
                "eigen_output_naming_pattern": (
                    "For each groupname, write_eigen_stress emits 6 "
                    "names: <base>_eigenval{1,2,3} (1 component each) "
                    "and <base>_eigenvec{1,2,3} (3 components each). "
                    "Both nodal_ and element_ prefixes applied via "
                    "stresskind dispatch."),
            },
            "thermo_heatflux_filter_internals": {
                "post_heatflux_heatfluxtype_enum": [
                    "ndxyz",
                    "cxyz",
                    "cxyz_ndxyz",
                ],
                "post_heatflux_dispatch": {
                    "ndxyz": "write_heatflux(..., nodebased)",
                    "cxyz": "write_heatflux(..., elementbased)",
                    "cxyz_ndxyz": (
                        "write_heatflux(..., nodebased) then "
                        "PostResult reset then write_heatflux("
                        "..., elementbased) — dual write"),
                },
                "special_field_subclasses_in_file": [
                    "WriteNodalHeatfluxStep (numdf-aware 1/2/3 "
                    "components; uses Thermo::postproc_thermo_"
                    "heatflux action via dis->evaluate; "
                    "components averaged across adjoining elements "
                    "via /adjele)",
                    "WriteElementCenterHeatfluxStep (numdf-aware "
                    "1/2/3 components; passes 'eleheatflux' "
                    "vector to elements; FOUR_C_THROW if returned "
                    "vector is nullptr)",
                ],
                "write_heatflux_groupname_vocab": [
                    "gauss_initial_heatfluxes_xyz",
                    "gauss_current_heatfluxes_xyz",
                    "gauss_initial_tempgrad_xyz",
                    "gauss_current_tempgrad_xyz",
                ],
                "element_action_routed": (
                    "Thermo::postproc_thermo_heatflux — routed via "
                    "Teuchos::ParameterList p.set<Thermo::Action>("
                    "'action', ...). 'heatfluxtype' is passed AGAIN "
                    "as a parameter string ('ndxyz' or 'cxyz') to "
                    "tell the element which output shape to fill."),
                "numdf_per_dim": (
                    "WriteNodalHeatfluxStep + WriteElementCenter"
                    "HeatfluxStep::numdf() return 1/2/3 from "
                    "problem()->num_dim(). FOUR_C_THROW('Cannot "
                    "handle dimension {}') for any other dim. "
                    "Average is per-element-incidence (adjele = "
                    "lnode->num_element() in the nodal averaging "
                    "loop) — boundary nodes with fewer adjacent "
                    "elements get the same /adjele divisor as "
                    "interior nodes."),
            },
            "Signal": (
                "[Output] Six sharp edges in post_processor most users "
                "hit at least once: "
                "(1) --filter is case-sensitive enum {'ensight', 'vtu', "
                "'vtu_node_based', 'vti'}. Common mistakes: 'VTU' "
                "(uppercase), 'paraview', 'vtkhdf' — all FOUR_C_THROW "
                "'Unknown filter <X> given'. "
                "(2) On problemtype scatra / cardiac_monodomain / elch "
                "with num_discr() == 1, the tool SILENTLY DOES NOTHING "
                "(comment in source: 'runtime output is used for scatra'). "
                "No .case file appears, no error, no warning. The "
                "runtime VTU output written during the solve is the only "
                "result; users running post_processor expecting "
                "additional output get nothing. "
                "(3) The fluid case has [[fallthrough]] to fluid_redmodels "
                "which has [[fallthrough]] to fluid_ale. A `fluid` "
                "problem with num_discr()==2 and disc[1].name()=='xfluid' "
                "writes THREE filters (XFluidFilter for disc[1] + the "
                "fluid_redmodels artery branch's StructureFilter on the "
                "same disc + FluidFilter for disc[0]). If disc[1] is NOT "
                "named 'xfluid', the fluid_redmodels artery branch still "
                "writes StructureFilter on disc[1], which is wrong for a "
                "pure ALE fluid. "
                "(4) fsi_xfem / fpsi_xfem branch has an INVERTED-LOGIC "
                "test: `disname.compare(1, 12, \"boundary_of_\")` "
                "returns 0 (falsy) when the substring at offset 1 IS "
                "'boundary_of_', so the InterfaceFilter branch runs ONLY "
                "for discretizations whose name does NOT match. Discs "
                "literally named like '_boundary_of_fluid' fall to the "
                "FOUR_C_THROW 'You try to postprocess a discretization "
                "with name {X}, maybe you should add it here?'. The fix "
                "would be `== 0` or `starts_with` — upstream bug worth "
                "reporting. "
                "(5) fsi_xfem ALE branch (case Core::ProblemType::fsi_xfem "
                "around line 305-310) is DEAD CODE: 'ale' name matches "
                "and prints the header but the AleFilter constructor + "
                "WriteFiles call are COMMENTED OUT. ALE fields in "
                "fsi_xfem problems produce no output despite the visual "
                "indicator. "
                "(6) ProblemType::none uses AnyFilter and writes "
                "'whatever vectors exist' in the first discretization. "
                "This is the right escape hatch for ad-hoc field dumps "
                "but offers no diagnostic if the discretization is "
                "missing — user sees an empty .case. "
                "(7) FluidFilter::write_all_results has a HARDCODED "
                "`int num_levelsets = 20;` (line 271 of "
                "single_field_writers.cpp) which unconditionally loops "
                "writing phinp_0..phinp_19 from the XFluid level-set "
                "store. Users with fewer than 20 level-sets get "
                "silent no-ops for the missing tags; users with MORE "
                "than 20 lose level-sets 20+ from .case output. "
                "Recompile-only knob — no runtime override. "
                "(8) XFluidFilter uses the `_smoothed` naming "
                "convention (velocity_smoothed, pressure_smoothed) "
                "for its 4-DOF-per-node fixed-size Paraview vectors, "
                "NOT the raw XFEM runtime names ('velocity', "
                "'pressure'). Source comment (line 286-294) "
                "explains XFEM has changing DOF counts so restart "
                "vectors and Paraview vectors are different sizes. "
                "Users grepping a runtime XFEM .out for 'velocity' "
                "see the raw name; users opening the post_processor "
                ".case in ParaView see velocity_smoothed — and "
                "looking for the wrong name gives 'field not found'. "
                "(9) StructureFilter + ThermoFilter post-stress / "
                "post-heatflux paths call BOTH alternatives "
                "(gauss_cauchy_stresses_xyz AND gauss_2PK_stresses_xyz; "
                "{current,initial}_heatfluxes / tempgrad pairs) even "
                "though only ONE is present in the result archive at "
                "a time. Comments at lines 142-159 / 362-379 spell "
                "this out: 'only one function call to PostStress is "
                "really postprocessing Gauss point stresses'. The "
                "non-present tag's write_result is silently a no-op. "
                "Plus 5 strain types are tried "
                "(GL/EA/LOG/pl_GL/pl_EA) — users counting writes "
                "from log lines see 'attempted N writes, got M' "
                "without an error indicator. "
                "(12) Dead wrapper script: apps/post_processor/"
                "scripts/post_gid ships in every build "
                "(installed alongside post_ensight / post_vti / "
                "post_vtu by create_post_scripts.cmake's "
                "copy_script() invocations) but invokes "
                "`post_processor --filter=gid $@` — and 'gid' is "
                "NOT in the post_processor filter enum (which is "
                "{'ensight', 'vtu', 'vtu_node_based', 'vti'} per "
                "edge 1). Every invocation of `./post_gid <file>` "
                "FOUR_C_THROWs 'Unknown filter gid given, "
                "supported filters: [ensight|vtu|vti]' at the "
                "filter-dispatch line in apps/post_processor/"
                "4C_post_processor.cpp. The wrapper is dead "
                "code — likely a leftover from when 4C had a "
                "Ciarlet-Geuzaine GiD output backend that was "
                "removed without updating the CMake glue. Users "
                "expecting GiD output get no help text, just the "
                "filter-enum error. Workaround: drop the GiD "
                "format entirely and use ensight / vtu / vti, OR "
                "edit post_processor source to add a 'gid' "
                "branch (requires an actual GiD writer "
                "implementation which the 4C codebase no longer "
                "ships). "
                "(11) ThermoFilter::post_heatflux dispatches on a "
                "3-VALUE heatfluxtype enum {'ndxyz', 'cxyz', "
                "'cxyz_ndxyz'} — NO eigen variants. Three-tool "
                "asymmetry users routinely confuse: "
                "post_monitor accepts {'none', 'ndxyz'} (2 values, "
                "tick #54); post_processor structure_stress accepts "
                "{'ndxyz', 'cxyz', 'cxyz_ndxyz', 'nd123', 'c123', "
                "'c123_nd123'} (6 values, edge 10); post_processor "
                "thermo_heatflux accepts {'ndxyz', 'cxyz', "
                "'cxyz_ndxyz'} (3 values, this edge). All three "
                "tools share the literal 'ndxyz' / 'cxyz' tokens "
                "but ONLY post_processor structure_stress accepts "
                "'nd123' / 'c123' eigen variants. Unknown enum "
                "values in thermo_heatflux trigger FOUR_C_THROW("
                "'Unknown heatflux/tempgrad type'). The 4 "
                "groupnames write_heatflux accepts are "
                "{gauss_{initial,current}_{heatfluxes,tempgrad}_xyz} "
                "and any other groupname FOUR_C_THROWs 'trying to "
                "write something that is not a heatflux or a "
                "temperature gradient'. The nodal-averaging loop "
                "(WriteNodalHeatfluxStep::operator()) divides each "
                "summed Gauss-point contribution by "
                "lnode->num_element() — boundary nodes (fewer "
                "adjacent elements) get the same divisor as "
                "interior nodes, which means the implied "
                "consistent-projection weights are wrong at the "
                "domain boundary. Result: visualized boundary "
                "heatfluxes are biased; the bias is largest for "
                "coarse meshes. "
                "(10) StructureFilter::post_stress dispatches on a "
                "6-VALUE stresstype enum {'ndxyz', 'cxyz', "
                "'cxyz_ndxyz', 'nd123', 'c123', 'c123_nd123'} — "
                "WIDER than post_monitor's {'none', 'ndxyz'}. The two "
                "tools have ASYMMETRIC vocabularies and users "
                "routinely cross-confuse them. Any value outside the "
                "6-set FOUR_C_THROWs 'Unknown stress/strain type'. "
                "The '*_ndxyz' / '*_123' compound forms are DUAL-"
                "WRITE paths: write nodal first, PostResult reset, "
                "then element-center. Eigen variants (nd123/c123/"
                "c123_nd123) route to write_eigen_stress which emits "
                "<base>_eigenval{1,2,3} + <base>_eigenvec{1,2,3} per "
                "groupname (6 outputs per call). The eigen path is "
                "missing the 'rotation' groupname that write_stress "
                "supports — asking for principal rotation tensors is "
                "silently undefined. Also note write_eigen_stress's "
                "final else clause throws 'Unknown heatflux type' "
                "(line 636) — a verbatim copy-paste error from "
                "ThermoFilter, never updated; the message is "
                "misleading for structure dispatch. "
                "(File walks apps/post_processor/4C_post_processor.cpp + "
                "4C_post_processor_single_field_writers.cpp + "
                "4C_post_processor_structure_stress.cpp + "
                "4C_post_processor_thermo_heatflux.cpp + "
                "scripts/create_post_scripts.cmake + scripts/post_gid "
                "2026-06-03.)"
            ),
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURAL MECHANICS (structure, structure_new, solid_3D_ele)
    # ═══════════════════════════════════════════════════════════════════════
    "structural_mechanics": {
        "description": "Full nonlinear structural mechanics — the core of 4C",
        "problemtype": "Structure",
        "yaml_section": "STRUCTURAL DYNAMIC",

        "time_integration": {
            "Statics": "Static analysis (one step, equilibrium)",
            "GenAlpha": "Generalized-alpha implicit time integration",
            "GenAlphaLieGroup": "Generalized-alpha for SO(3) rotation group (beams/shells)",
            "OneStepTheta": "One-step-theta implicit scheme",
            "ExplEuler": "Explicit forward Euler",
            "CentrDiff": "Explicit central differences (wave propagation)",
            "AdamsBashforth2": "Explicit 2nd order Adams-Bashforth",
            "AdamsBashforth4": "Explicit 4th order Adams-Bashforth",
        },

        "kinematics": {
            "linear": "Small strain / linear kinematics (ε = sym(∇u))",
            "nonlinearTotLag": "Total Lagrangian / finite deformation (F = I + ∇u)",
        },

        "element_types": {
            "3D_solid": {
                "SOLID HEX8": "8-node hexahedron (Q1, standard or F-bar or EAS)",
                "SOLID HEX20": "20-node hexahedron (serendipity Q2)",
                "SOLID HEX27": "27-node hexahedron (full Q2)",
                "SOLID TET4": "4-node tetrahedron (P1)",
                "SOLID TET10": "10-node tetrahedron (P2)",
                "SOLID WEDGE6": "6-node wedge/prism",
                "SOLID WEDGE15": "15-node wedge (P2)",
                "SOLID PYRAMID5": "5-node pyramid",
                "SOLIDSCATRA HEX8": "8-node hex with scalar transport coupling (for TSI)",
            },
            "2D_wall": {
                "SOLID QUAD4": "4-node quadrilateral (plane "
                                "strain/stress, EAS option). NOTE: "
                                "'WALL QUAD4' is the legacy name "
                                "and is NOT registered in 4C "
                                "2026.3 (eletype 'WALL' raises "
                                "'Unknown type WALL' from "
                                "parobjectfactory.cpp). Use "
                                "SOLID QUAD4 + THICKNESS + "
                                "PLANE_ASSUMPTION.",
                "SOLID QUAD8": "8-node serendipity quad",
                "SOLID QUAD9": "9-node full biquadratic quad",
                "SOLID TRI3":  "3-node triangle (registered as "
                                "SOLID, not WALL)",
                "SOLID TRI6":  "6-node quadratic triangle",
            },
            "1D_beam": {
                "BEAM3R": "Simo-Reissner beam (shear-deformable, geometrically exact)",
                "BEAM3K": "Kirchhoff beam (shear-rigid, inextensible option)",
                "BEAM3EB": "Euler-Bernoulli beam (classical)",
            },
            "shell": {
                "SHELL7P": "7-parameter shell (EAS, ANS options, thickness locking-free)",
                "SHELL_KL_NURBS": "Kirchhoff-Love NURBS shell (isogeometric)",
            },
            "other": {
                "MEMBRANE": "Membrane element (no bending stiffness)",
                "TRUSS3": "Truss element (axial force only)",
                "TORSION3": "Torsional spring element",
                "RIGIDSPHERE": "Rigid sphere for DEM contact",
            },
        },

        "element_technologies": {
            "none": "Standard displacement-based formulation",
            "fbar": "F-bar method (volumetric locking treatment for hex8)",
            "eas_mild": "Enhanced Assumed Strain (mild enrichment, 7 modes for hex8)",
            "eas_full": "Enhanced Assumed Strain (full enrichment, 21 modes for hex8)",
            "shell_ans": "Assumed Natural Strain for shells (shear locking treatment)",
            "shell_eas": "EAS for shells",
            "shell_eas_ans": "Combined EAS + ANS for shells",
        },

        "nonlinear_solvers": {
            "newtonfull": "Full Newton-Raphson (assemble tangent every iteration)",
            "newtonmod": "Modified Newton (reuse tangent, cheaper per iteration)",
            "newtonls": "Newton with line search (backtracking)",
            "newtonuzawalin": "Linear Uzawa for constrained problems",
            "newtonuzawanonlin": "Nonlinear Uzawa",
            "ptc": "Pseudo-transient continuation (robust for difficult convergence)",
            "nox_nln": "NOX nonlinear solver framework (Trilinos)",
        },

        "wall_element_params": "KINEM linear/nonlinear, EAS none/full, THICK 1.0, STRESS_STRAIN plane_strain/plane_stress, GP 2 2",

        "pitfalls": [
            (
                "[API] SOLID QUAD4 (2D structural) needs: "
                "'MAT 1 KINEM nonlinear THICKNESS 1.0 "
                "PLANE_ASSUMPTION plane_strain'. DO NOT "
                "write 'WALL QUAD4', 'THICK', or "
                "'STRESS_STRAIN' — those are the legacy "
                "keywords. Signal: 4C 2026.3 rejects "
                "'WALL' with 'Unknown type WALL of finite "
                "element' from parobjectfactory.cpp:153; "
                "'THICK' / 'STRESS_STRAIN' raise 'unknown "
                "parameter' from input_spec_builders.cpp. "
                "Verified empirically; see Tier-2 fixture "
                "structural_2d_solid_quad4_not_wall. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] SOLID elements (3D + 2D) need: "
                "MAT <id> KINEM <linear or nonlinearTotLag>. "
                "Signal: writing KINEM nonlinear (without "
                "the TotLag suffix in 3D) is silently "
                "accepted in some 4C versions but produces "
                "an updated-Lagrangian formulation instead "
                "of total-Lagrangian — stress and strain "
                "are referred to the wrong configuration. "
                "Use nonlinearTotLag for finite-strain "
                "problems unless you specifically want UL. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] SOLIDSCATRA element is REQUIRED for "
                "TSI coupling — plain SOLID cannot couple "
                "with the thermal field. Signal: a TSI "
                "problem with SOLID elements (not "
                "SOLIDSCATRA) raises 'no SCATRA "
                "discretisation found' from "
                "4C_tsi_factory.cpp at setup; the "
                "structure has no SCATRA-side mass matrix "
                "to clone into a thermal discretisation. "
                "Replace 'SOLID' with 'SOLIDSCATRA' for "
                "all elements in the structural mesh. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For statics: MAXITER = 1 for "
                "linear problems, 10+ for nonlinear. "
                "Signal: MAXITER = 1 on a nonlinear "
                "problem stops after the first Newton "
                "iterate with residual still O(1) — "
                "result looks like the linear solution "
                "but is wrong for KINEM nonlinear. "
                "MAXITER > 1 on a linear problem wastes "
                "iterations (residual is already at "
                "tolerance after step 1). Match MAXITER "
                "to KINEM. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] PREDICT: TangDis is "
                "RECOMMENDED for nonlinear Newton "
                "convergence — uses the tangent "
                "stiffness to predict the next "
                "iterate. Signal: PREDICT: ConstDis "
                "(constant displacement) on a "
                "geometrically-nonlinear problem "
                "gives slow Newton convergence (5-10 "
                "iters per step vs 2-3 with TangDis); "
                "the tangent predictor jumps closer to "
                "the equilibrium each step. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] Body forces: DESIGN SURF NEUMANN "
                "(2D) or DESIGN VOL NEUMANN (3D) with "
                "NUMDOF matching the spatial dimension. "
                "Signal: a body-force section with "
                "NUMDOF = 6 on a 3D solid raises 'NUMDOF "
                "mismatch — expected 3 got 6'; gravity "
                "and per-volume forces use 3 components "
                "in 3D (FX, FY, FZ) and 2 in 2D. "
                "NUMDOF = 6 is for beam DOFs. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Beam elements need SPECIAL BEAM3* "
                "type — NOT SOLID or WALL. Signal: "
                "writing 'SOLID LINE2' or 'WALL LINE2' "
                "for beam elements raises 'Unknown type' "
                "from parobjectfactory.cpp — the SOLID/"
                "WALL factories only register volume/"
                "surface element families. Use BEAM3R / "
                "BEAM3K / BEAM3EB with the appropriate "
                "LINE2/LINE3/LINE4 cell type and TRIADS. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MATERIALS (120+ models)
    # ═══════════════════════════════════════════════════════════════════════
    "materials": {
        "description": "120+ material models spanning all physics disciplines",

        "basic_structural": {
            "MAT_Struct_StVenantKirchhoff": {
                "params": "YOUNG, NUE, DENS",
                "use": "Linear elastic (small strain) or geometric nonlinear",
            },
            "MAT_Struct_ThermoStVenantK": {
                "params": "YOUNG (array), NUE, DENS, THEXPANS, INITTEMP, THERMOMAT",
                "use": "Linear elastic with thermal expansion coupling (for TSI)",
                "notes": "THERMOMAT links to a MAT_Fourier for thermal properties",
            },
        },

        "hyperelastic": {
            "MAT_ElastHyper": "Toolbox: combine summands (NeoHooke + volumetric, etc.)",
            "summands": {
                "coupNeoHooke": "Neo-Hooke (coupled form): W = C1*(I1-3) + 1/(2*D1)*(J-1)^2",
                "couploganeohooke": "Logarithmic Neo-Hooke: W = mu/2*(I1-3) - mu*ln(J) + lam/2*ln(J)^2",
                "coupMooneyRivlin": "Mooney-Rivlin (coupled): W = C1*(I1-3) + C2*(I2-3)",
                "isoNeoHooke": "Isochoric Neo-Hooke (incompressible split)",
                "isoOgden": "Isochoric Ogden (stretch-based)",
                "isoYeoh": "Isochoric Yeoh (polynomial in I1)",
                "coupBlatzKo": "Blatz-Ko (compressible rubber-like)",
                "coupSimoPister": "Simo-Pister model",
                "coupAnisoExpo": "Anisotropic exponential fiber model (soft tissue)",
                "coupAnisoNeoHooke": "Anisotropic Neo-Hooke fiber",
            },
            "volumetric": {
                "volOgden": "Ogden volumetric penalty",
                "volPenalty": "Standard penalty: κ/2*(J-1)^2",
                "volSussmanBathe": "Sussman-Bathe volumetric",
            },
        },

        "viscoelastic": {
            "MAT_ViscoElastHyper": "Viscohyperelastic with Maxwell branches",
            "generalizedMaxwell": "Generalized Maxwell (Standard Linear Solid)",
            "fractionalSLS": "Fractional Standard Linear Solid",
        },

        "plasticity": {
            "MAT_PlLinElast": "Small-strain von Mises plasticity (YOUNG, NUE, YIELD, SATHARDENING, etc.)",
            "MAT_PlNlnLogNeoHooke": "Finite strain von Mises + logarithmic Neo-Hooke",
            "MAT_PlDruckPrag": "Drucker-Prager plasticity (pressure-dependent yield)",
            "MAT_PlGTN": "Gurson-Tvergaard-Needleman (ductile damage)",
            "MAT_CrystPlast": "Crystal plasticity (single crystal, multiple slip systems)",
            "MAT_PlElastHyper": "Hyperelastic + finite strain von Mises (semi-smooth Newton)",
        },

        "biological": {
            "MAT_ConstraintMixture": "Constrained mixture model for arterial growth/remodeling",
            "MAT_GrowthRemodelElastHyper": "Growth and remodeling hyperelastic",
            "MAT_Muscle_Combo": "Active strain muscle model (combo)",
            "MAT_Muscle_Giantesio": "Giantesio active strain muscle",
            "MAT_Myocard": "Myocardial tissue with electrophysiology (FHN, TenTusscher, etc.)",
        },

        "fluid": {
            "MAT_Fluid": "Newtonian fluid (DYNVISCOSITY, DENSITY)",
            "MAT_CarreauYasuda": "Carreau-Yasuda shear-thinning",
            "MAT_HerschelBulkley": "Herschel-Bulkley yield stress fluid",
            "MAT_Sutherland": "Temperature-dependent viscosity (Sutherland law)",
        },

        "thermal": {
            "MAT_Fourier": "Fourier heat conduction (CAPA=heat capacity, CONDUCT=conductivity)",
            "MAT_Soret": "Soret effect (thermodiffusion coupling)",
        },

        "scalar_transport": {
            "MAT_scatra": "General scalar transport (DIFFUSIVITY parameter)",
            "MAT_scatra_reaction": "Reactive scalar transport",
            "MAT_scatra_chemotaxis": "Chemotactic scalar transport",
        },

        "porous_media": {
            "MAT_FluidPoro": "Darcy fluid in porous medium",
            "MAT_StructPoro": "Structural skeleton for poroelasticity",
            "phase_laws": "Linear, tangent, constraint, by-function",
            "permeability_laws": "Constant, exponential",
        },

        "particle": {
            "MAT_Particle_SPH_Fluid": "SPH fluid particle",
            "MAT_Particle_DEM": "DEM particle",
            "MAT_Particle_PD": "Peridynamic particle (bond-based)",
        },

        # Beam material names from 4C 2026.3 schema (MATERIALS
        # section enum). Catalog previously had wrong delimiter:
        # 'MAT_Beam_Reissner_ElastHyper' (underscore-separated)
        # is NOT a valid 4C material name — real format is
        # 'MAT_BeamReissnerElastHyper' (CamelCase, only one
        # underscore between MAT and the beam family). Wrong
        # names fail at YAML parse with input_spec_builders.cpp
        # 'Could not match this input'. Verified 2026-06-01.
        "beam": {
            "MAT_BeamReissnerElastHyper": "Simo-Reissner beam hyperelastic (default for BEAM3R)",
            "MAT_BeamReissnerElastHyper_ByModes": "Reissner beam, parametrized by deformation modes",
            "MAT_BeamReissnerElastPlastic": "Reissner beam with plasticity",
            "MAT_BeamKirchhoffElastHyper": "Kirchhoff beam hyperelastic (default for BEAM3K)",
            "MAT_BeamKirchhoffElastHyper_ByModes": "Kirchhoff beam, parametrized by deformation modes",
            "MAT_BeamKirchhoffTorsionFreeElastHyper": "Kirchhoff torsion-free hyperelastic (default for BEAM3EB)",
            "MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes": "Torsion-free Kirchhoff, parametrized by modes",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # FLUID MECHANICS (fluid, fluid_ele, fluid_turbulence)
    # ═══════════════════════════════════════════════════════════════════════
    "fluid": {
        "description": "Incompressible Navier-Stokes with stabilized FEM",
        "problemtype": "Fluid",
        "yaml_section": "FLUID DYNAMIC",

        # FLUID DYNAMIC/TIMEINTEGR enum (4C 2026.3 schema —
        # 4C_schema.json). NOTE: different from STRUCTURAL
        # DYNAMIC/DYNAMICTYPE — the fluid enum has Af_/Np_
        # prefixes on Gen_Alpha and underscores on
        # One_Step_Theta. Verified 2026-06-01.
        "time_integration": {
            "Af_Gen_Alpha": "Alpha-form generalized-alpha (alpha-f weighting on the residual)",
            "Np_Gen_Alpha": "N+1 generalized-alpha (default for incompressible NS)",
            "BDF2": "2nd order backward difference formula",
            "One_Step_Theta": "One-step-theta (note underscores — NOT 'OneStepTheta')",
            "Stationary": "Steady-state RANS or Stokes",
        },

        "stabilization": {
            "SUPG": "Streamline upwind Petrov-Galerkin",
            "GLS": "Galerkin least squares",
            "VMS": "Variational multiscale (recommended)",
            "PSPG": "Pressure stabilization Petrov-Galerkin",
        },

        "turbulence_models": [
            "Dynamic Smagorinsky (LES)",
            "Dynamic Vreman (LES)",
            "k-epsilon (RANS, via additional scatra equations)",
        ],

        "ale": "ALE formulation for moving meshes (ale2, ale3 elements)",

        "pitfalls": [
            "[Syntax] Fluid uses its own element section "
            "'FLUID ELEMENTS' (NOT 'STRUCTURE'). The dynamics-"
            "control section is 'FLUID DYNAMIC' (not 'FLUID' or "
            "'FLUID_DYN'). Wrong section name is rejected at "
            "YAML parse with 'PROC 0 ERROR ... Section ... is "
            "not a valid section name.' from "
            "core/io/src/4C_io_input_file.cpp. Signal: stderr "
            "contains the offending section name + 'not a valid "
            "section name'. (Verified empirically 2026-06-01 — "
            "'FLUID' was rejected with this exact diagnostic; "
            "'FLUID DYNAMIC' was accepted. Same family as "
            "scatra_section_name_required fixture; no separate "
            "Tier-2 fixture added to avoid duplication.)",
            "[Numerical] Stabilization parameters (SUPG, PSPG, "
            "GRAD-DIV) need tuning at high Reynolds. Default "
            "values in FLUID DYNAMIC/STABILIZATION are tuned "
            "for moderate Re; for Re > 1000 the residual-based "
            "tau parameter benefits from increasing TAU_TYPE / "
            "TAU_DEF or switching to a more dissipative variant. "
            "Signal: integrated kinetic energy in the FLUID "
            "discretization grows non-physically as Re is "
            "increased without stabilisation re-tuning. (Claim "
            "inherited.)",
            "[Integration] ALE (arbitrary Lagrangian-Eulerian) "
            "mesh movement requires a SEPARATE ALE problem set "
            "up alongside the fluid problem in PROBLEM TYPE: "
            "'Fluid_Ale'. The mesh motion equation (typically "
            "elastic) is solved each step on the same "
            "discretization. Signal: PROBLEMTYPE: 'Fluid_Ale' "
            "is the enum value 4C expects; the ALE DYNAMIC "
            "section is required. (Claim inherited.)",
            "[Numerical] X-wall functions: extended near-wall "
            "treatment for high-Re flows where direct DNS-"
            "resolved boundary layers are infeasible. Activated "
            "via FLUID DYNAMIC/WALL_NORMAL_NODE_DISTANCE and "
            "related XWALL_* keys. Signal: in a turbulent "
            "channel flow benchmark, the near-wall velocity "
            "profile matches the log-law slope (1/0.41 × ln(y+) "
            "+ 5.0) within ~5% with x-wall enabled; without, "
            "the law-of-the-wall is over-resolved at the wall "
            "and diverges in the log-region. (Claim inherited.)",
            "[API] 4C time-integration enum naming is "
            "SECTION-DEPENDENT — the same conceptual scheme "
            "has different spellings in different YAML sections. "
            "FLUID DYNAMIC/TIMEINTEGR accepts {Af_Gen_Alpha, "
            "Np_Gen_Alpha, BDF2, One_Step_Theta, Stationary} "
            "(underscored, with Af_/Np_ prefixes on Gen-Alpha). "
            "SCALAR TRANSPORT DYNAMIC/TIMEINTEGR accepts "
            "{Gen_Alpha, BDF2, One_Step_Theta, Stationary} "
            "(underscored, no prefix). STRUCTURAL DYNAMIC/"
            "DYNAMICTYPE accepts {GenAlpha, GenAlphaLieGroup, "
            "OneStepTheta, Statics, CentrDiff, AdamsBashforth2, "
            "AdamsBashforth4, ExplicitEuler} (CamelCase, no "
            "underscores). THERMAL DYNAMIC/DYNAMICTYPE accepts "
            "{GenAlpha, OneStepTheta, Statics, Undefined} "
            "(CamelCase). The earlier catalog used bare "
            "'GenAlpha' / 'OneStepTheta' uniformly across "
            "physics — wrong for fluid + scatra. Signal: wrong "
            "enum value fails at YAML parse with "
            "input_spec_builders.cpp 'Could not match this "
            "input'. Verified empirically against 4C 2026.3 "
            "schema 2026-06-01.",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SCALAR TRANSPORT (scatra, scatra_ele)
    # ═══════════════════════════════════════════════════════════════════════
    "scalar_transport": {
        "description": "Convection-diffusion / scalar transport — the workhorse for Poisson, heat, electrochemistry",
        "problemtype": "Scalar_Transport",
        "yaml_section": "SCALAR TRANSPORT DYNAMIC",

        # SCALAR TRANSPORT DYNAMIC/TIMEINTEGR enum
        # (4C 2026.3 schema). Different from STRUCTURAL/THERMAL
        # DYNAMIC/DYNAMICTYPE (which uses CamelCase). The
        # scatra TIMEINTEGR has underscores.
        "time_integration": ["Gen_Alpha", "BDF2", "One_Step_Theta", "Stationary"],

        "physics_variants": {
            "standard": "Pure convection-diffusion-reaction",
            "electrochemistry": "Nernst-Planck ion transport (elch, elch_diffcond, elch_scl)",
            "cardiac_monodomain": "Cardiac electrophysiology (FHN, TenTusscher models)",
            "level_set": "Level-set advection + reinitialization",
            "porous_media": "Scalar transport in porous media",
            "growth_remodel": "Growth and remodeling scalar transport",
        },

        "elements": "TRANSP QUAD4/8/9 (2D), TRANSP HEX8/20/27 (3D), TRANSP TRI3/6, TRANSP TET4/10",

        "pitfalls": [
            "[Syntax] Top-level section name must be the full "
            "spelling 'SCALAR TRANSPORT DYNAMIC' (matches "
            "scalar_transport.yaml_section above), NOT the "
            "abbreviation 'SCATRA DYNAMIC' that internal Kratos / "
            "4C application names suggest. 4C rejects 'SCATRA "
            "DYNAMIC' at YAML parse time with 'PROC 0 ERROR ... "
            "Section \"SCATRA DYNAMIC\" is not a valid section "
            "name.' from core/io/src/4C_io_input_file.cpp. "
            "Signal: \"SCATRA DYNAMIC\" + \"not a valid section "
            "name\" + \"4C_io_input_file\" all appear in 4C "
            "stderr. Element section is similarly TRANSPORT "
            "ELEMENTS (not STRUCTURE or FLUID). (Verified "
            "empirically 2026-06-01.)",
            "[Integration] Material: MAT_scatra with DIFFUSIVITY "
            "parameter (and CAPACITY for transient). Without a "
            "MAT_scatra entry for the elements' MAT id, the "
            "solver setup fails when the constitutive law is "
            "looked up. Signal: 4C ERROR mentioning 'Material' / "
            "'MAT_scatra' and the missing MAT id. (Claim "
            "inherited — not yet empirically verified.)",
            "[Syntax] For Poisson via scatra: TIMEINTEGR "
            "Stationary, VELOCITYFIELD zero (the field must be "
            "set, not omitted), source applied via DESIGN SURF "
            "NEUMANN. Signal: omitting VELOCITYFIELD may surface "
            "later as an undefined-field error during element "
            "initialisation, not at parse time. (Claim "
            "inherited — not yet fully verified; partial probe "
            "saw a TRANSPORT-ELEMENTS-empty error before "
            "VELOCITYFIELD was checked.)",
            "[Syntax] For scatra heat: same skeleton as Poisson "
            "but T_left/T_right via DESIGN LINE DIRICH (linewise "
            "Dirichlet) rather than NEUMANN. Signal: temperature "
            "field shows the prescribed BC at the boundaries; "
            "swapping DIRICH ↔ NEUMANN produces a constant flux "
            "but no enforced temperature. (Claim inherited.)",
            "[API] IO/RUNTIME VTK OUTPUT/SCATRA may crash 4C — "
            "the scatra-specific RUNTIME VTK path has known "
            "issues for some element types; safer to omit the "
            "RUNTIME VTK SCATRA subsection and convert results "
            "to .vtu via the post_vtu post-processor after the "
            "run completes. Signal: crash inside 4C's "
            "RUNTIME_VTK output code rather than diverged "
            "physics; switching to post_vtu sidesteps it. "
            "(Claim inherited.)",
            "[Syntax] Field name in scatra VTU output is "
            "phi_1, phi_2, ... (not 'temperature' or 'u' as "
            "users sometimes assume). When loading the result "
            "in ParaView, the array name must match. Signal: "
            "ParaView (or paraview-python) opening the "
            "post_vtu output of a SCATRA DYNAMIC run reports "
            "'no array named temperature' when the user "
            "expects 'phi_1'. (Claim inherited.)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # THERMAL (thermo)
    # ═══════════════════════════════════════════════════════════════════════
    "thermal": {
        "description": "Thermal analysis (standalone or coupled via TSI/STI/SSTI)",
        "problemtype": "Thermo",
        "yaml_section": "THERMAL DYNAMIC",
        "time_integration": ["Statics", "GenAlpha", "OneStepTheta"],
        "boundary_conditions": {
            "DESIGN SURF THERMO DIRICH CONDITIONS": "Prescribed temperature",
            "DESIGN SURF THERMO NEUMANN CONDITIONS": "Prescribed heat flux",
            "ThermoConvections": "Convective heat transfer BC (h*(T-T_inf))",
            "ThermoRobin": "Robin BC for thermal",
        },
        "pitfalls": [
            (
                "[Syntax] Use THERMO not THERMAL in section "
                "names: 'DESIGN SURF THERMO DIRICH', "
                "'DESIGN SURF THERMO NEUMANN', "
                "'DESIGN VOL THERMO DIRICH'. Signal: "
                "writing 'DESIGN SURF THERMAL DIRICH' "
                "raises 'unknown section' from "
                "input_spec_builders.cpp at parse — the "
                "vocabulary uses THERMO consistently "
                "across all conditions; THERMAL appears "
                "only in section names like 'THERMAL "
                "DYNAMIC'. (Audit 2026-06-02.)"
            ),
            (
                "[Input] For TSI: thermal field is "
                "SOLVED BY 4C, not prescribed externally. "
                "Signal: prescribing temperature via a "
                "Dirichlet on every node (instead of "
                "letting 4C solve the heat equation) "
                "defeats the coupling — there is no "
                "feedback from structure to thermal. Use "
                "thermal source terms (Joule heating, "
                "mechanical dissipation) and BCs only on "
                "physical heat-input/output boundaries. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] INITIALFIELD: field_by_function "
                "with INITFUNCNO pointing to a FUNCT for "
                "initial temperature. Signal: omitting "
                "INITIALFIELD defaults to T = 0, which "
                "for a problem with INITTEMP > 0 in the "
                "structural material gives spurious "
                "thermal-strain at t = 0 (the difference "
                "T - INITTEMP drives the contraction). "
                "Set INITIALFIELD to match INITTEMP. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-PHYSICS COUPLING
    # ═══════════════════════════════════════════════════════════════════════
    "tsi": {
        "description": "Thermo-Structure Interaction — the key multi-physics coupling in 4C",
        "problemtype": "Thermo_Structure_Interaction",
        "yaml_sections": ["STRUCTURAL DYNAMIC", "THERMAL DYNAMIC", "TSI DYNAMIC"],

        # COUPALGO enum values from 4C 2026.3 schema
        # (TSI DYNAMIC/COUPALGO). Verified via 4C_schema.json
        # 2026-06-01. Names that LOOK obvious are NOT — the
        # underscore between 'iterstagg' and the variant is
        # load-bearing, 'fixedrel' is actually 'fixedrelax',
        # and the monolithic variant is 'tsi_monolithic'
        # (NOT bare 'monolithic'). Wrong values fail at
        # YAML parse with input_spec_builders.cpp
        # 'Could not match this input'.
        "coupling_algorithms": {
            "tsi_oneway": "One-way: thermal → structural (no feedback)",
            "tsi_sequstagg": "Sequential staggered (solve thermal, then structural, once per step)",
            "tsi_iterstagg": "Iterative staggered (iterate until convergence)",
            "tsi_iterstagg_aitken": "Iterative staggered with Aitken acceleration",
            "tsi_iterstagg_aitkenirons": "Aitken-Irons variant",
            "tsi_iterstagg_fixedrelax": "Fixed relaxation iterative staggered",
            "tsi_monolithic": "Simultaneous solve of all fields (TSI DYNAMIC/MONOLITHIC section)",
        },

        "requirements": [
            "SOLIDSCATRA elements (NOT plain SOLID — the SCATRA coupling is needed). "
            "Accepts HEX8, HEX27, TET4, TET10, NURBS27, QUAD4, QUAD9, TRI3, TRI6.",
            "MAT_Struct_ThermoStVenantK (structural material with thermal expansion)",
            "MAT_Fourier (thermal material linked via THERMOMAT parameter)",
            "CLONING MATERIAL MAP: SRC_FIELD structure → TAR_FIELD thermo",
            "Two LINEAR_SOLVERs: one for thermal, one for structural",
            "FUNCT for INITIALFIELD: SYMBOLIC_FUNCTION_OF_SPACE_TIME for initial temperature",
        ],

        "pitfalls": [
            (
                "[Input] Without CLONING MATERIAL MAP, 4C "
                "crashes at initialization. Signal: TSI setup "
                "phase aborts with 'cannot clone material for "
                "thermo field' from "
                "4C_adapter_str_factory.cpp; the thermal "
                "discretisation has no way to inherit the "
                "structural cell topology + nodes. Standard "
                "form: SRC_FIELD: structure, SRC_MAT: <struct_"
                "mat_id>, TAR_FIELD: thermo, TAR_MAT: <thermo_"
                "mat_id>. (Audit 2026-06-02.)"
            ),
            (
                "[Input] THEXPANS in MAT_Struct_"
                "ThermoStVenantK is the thermal-expansion "
                "coefficient — UNITS must match the "
                "temperature units used elsewhere. Signal: "
                "a 4C input with INITTEMP in Kelvin and "
                "THEXPANS in 1/Celsius produces displacement "
                "that differs from analytic by exactly the "
                "T_reference offset (273.15) times "
                "alpha*length — easily mistaken for boundary-"
                "condition error. Use consistent units (all "
                "SI, all CGS, etc.) throughout. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] INITTEMP is the reference "
                "temperature for ZERO thermal strain. Signal: "
                "omitting INITTEMP defaults to 0 — a heated "
                "specimen at T = 300 K with no INITTEMP "
                "specified produces unrealistically large "
                "thermal strains as if it started from "
                "absolute zero; expansion u = alpha * "
                "DeltaT * L where DeltaT = T - 0 instead of "
                "T - T_ref. Set INITTEMP to the stress-free "
                "temperature (room temperature for typical "
                "experiments). (Audit 2026-06-02.)"
            ),
            (
                "[Input] TSI DYNAMIC controls the COUPLING "
                "(time step, ITEMAX, COUPALGO); the per-"
                "field STRUCTURAL DYNAMIC and THERMAL "
                "DYNAMIC sections control the individual "
                "field solvers. Signal: setting NUMSTEP in "
                "STRUCTURAL DYNAMIC but not in TSI DYNAMIC "
                "is silently ignored — TSI DYNAMIC's "
                "NUMSTEP wins and the structural section's "
                "value is unused. Always set time-loop "
                "controls in TSI DYNAMIC; use per-field "
                "DYNAMIC sections for tolerances and "
                "predictor type only. (Audit 2026-06-02.)"
            ),
            (
                "[Input] For one-way TSI (no feedback): "
                "ITEMAX = 1 (only one coupling iteration "
                "needed). Signal: in TSI_DYNAMIC, ITEMAX > "
                "1 on a one-way problem still converges but "
                "wastes wall-clock — each extra iteration "
                "recomputes the second field with unchanged "
                "inputs. Conversely, ITEMAX = 1 on a TWO-"
                "way TSI_DYNAMIC problem stops before "
                "convergence and yields a partly-converged "
                "solution that looks like the right answer "
                "but has 5-20% error on the coupled "
                "response. Match ITEMAX to coupling type. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] SOLIDSCATRA elements REQUIRE 'TYPE "
                "Undefined' in the element definition. "
                "Signal: omitting TYPE or writing 'TYPE "
                "Std' triggers a RUNTIME FOUR_C_THROW "
                "'TYPE ... not valid for SOLIDSCATRA "
                "elements' at problem setup (TYPE is a "
                "free-form schema string, so the YAML "
                "parser does NOT reject it). Full format: "
                "<id> SOLIDSCATRA HEX8 <n1..n8> MAT <id> "
                "KINEM nonlinear TYPE Undefined. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] For one-way thermal -> structural "
                "TSI: MUST add TSI DYNAMIC/PARTITIONED "
                "section with COUPVARIABLE: Temperature. "
                "Signal: without it, 4C defaults to "
                "displacement coupling (structural -> "
                "thermal), which is BACKWARDS for heating "
                "problems — the result is zero "
                "displacement everywhere because the "
                "structural field gets no thermal forcing "
                "input. Sanity check: a heated bar should "
                "expand; if it doesn't, COUPVARIABLE is "
                "likely missing or set to Displacement. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Monolithic TSI requires the "
                "Belos iterative solver with a block "
                "preconditioner (NOT UMFPACK). Signal: "
                "writing 'SOLVER: UMFPACK' for a "
                "tsi_monolithic problem aborts with "
                "'monolithic TSI requires Belos' from "
                "4C_tsi_monolithic.cpp at setup; the "
                "monolithic Jacobian is too large and ill-"
                "conditioned for a direct solver. For "
                "simple one-way problems, use partitioned "
                "tsi_oneway with UMFPACK (much simpler "
                "setup). (Audit 2026-06-02.)"
            ),
            (
                "[Input] Volume-level thermal Dirichlet: "
                "use DESIGN VOL THERMO DIRICH CONDITIONS + "
                "DVOL-NODE TOPOLOGY to prescribe "
                "temperature on all nodes in a volume "
                "region. Signal: applying a Dirichlet to "
                "the surface-only set when you want a "
                "constant-temperature volume leaves "
                "interior nodes free — temperature "
                "develops a non-uniform interior profile "
                "instead of staying clamped. For "
                "uniform-T initial condition over a "
                "region, prefer INITIALFIELD + FUNCT over "
                "DIRICH BCs (more efficient). (Audit "
                "2026-06-02.)"
            ),
            "[API] TSI DYNAMIC/COUPALGO is an enum of exactly 7 "
            "values: tsi_oneway, tsi_sequstagg, tsi_iterstagg, "
            "tsi_iterstagg_aitken, tsi_iterstagg_aitkenirons, "
            "tsi_iterstagg_fixedrelax, tsi_monolithic. The "
            "earlier catalog had FOUR wrong names: "
            "'tsi_iterstaggaitken' (missing underscore — real: "
            "'tsi_iterstagg_aitken'), 'tsi_iterstaggaitkenirons' "
            "(missing underscore — real: 'tsi_iterstagg_"
            "aitkenirons'), 'tsi_iterstaggfixedrel' (wrong stem "
            "— real: 'tsi_iterstagg_fixedrelax'), and "
            "'monolithic' (missing 'tsi_' prefix — real: "
            "'tsi_monolithic'). Signal: invalid COUPALGO value "
            "in YAML produces 'PROC 0 ERROR' from "
            "input_spec_builders.cpp with 'Could not match this "
            "input' and the offending YAML block echoed. "
            "Verified empirically against 4C 2026.3 schema "
            "2026-06-01.",
            "[API] SOLIDSCATRA elements accept exactly 11 TYPE "
            "values: Undefined, AdvReac, CardMono, GR, NLS, "
            "Chemo, ChemoReac, ElchDiffCond, ElchElectrode, "
            "Loma, Std (4C_solid_scatra_ele_lib.cpp). For "
            "TSI specifically, use 'TYPE Undefined' — the "
            "SCATRA half is cloned into a thermal "
            "discretization, the SCATRA impl in the structure "
            "is therefore unused. The schema's TYPE field is a "
            "free-form string, so invalid TYPE values fail at "
            "RUNTIME (FOUR_C_THROW 'not valid for SOLIDSCATRA "
            "elements') rather than at YAML parse time. "
            "SOLIDSCATRA also supports QUAD4, QUAD9, TRI3, "
            "TRI6, HEX27, TET4, TET10, NURBS27 — not just HEX8. "
            "Signal: 4C stderr emits 'TYPE <bad value> not "
            "valid for SOLIDSCATRA elements' at the first "
            "time step (NOT at YAML parse) when the TYPE "
            "string is not in the 11-value enum; correct it "
            "to one of the listed values. Verified "
            "2026-06-01.",
        ],
    },

    "fsi": {
        "description": "Fluid-Structure Interaction — partitioned and monolithic coupling",
        "problemtype": "Fluid_Structure_Interaction",

        "partitioned_algorithms": {
            "Dirichlet-Neumann": "Standard: displacement/velocity/force coupling at interface",
            "DirichletNeumannSlideALE": "Sliding interface variant",
            "relaxation": ["Fixed", "Aitken", "Steepest descent", "Chebyshev", "NLCG"],
            "MFNK": "Matrix-free Newton-Krylov (advanced, robust)",
        },

        "monolithic_algorithms": {
            "fluid_split": "Monolithic with fluid-based splitting",
            "structure_split": "Monolithic with structure-based splitting",
            "mortar": "Mortar-based monolithic (non-matching meshes)",
            "xfem": "XFEM-based monolithic (no mesh conformity needed)",
        },

        "required_sections": [
            "PROBLEM TYPE", "PROBLEM SIZE",
            "STRUCTURAL DYNAMIC", "STRUCTURAL DYNAMIC/GENALPHA",
            "FLUID DYNAMIC", "ALE DYNAMIC",
            "FSI DYNAMIC", "FSI DYNAMIC/MONOLITHIC SOLVER",
            "MATERIALS", "CLONING MATERIAL MAP",
            "STRUCTURE GEOMETRY", "FLUID GEOMETRY",
            "DESIGN FSI COUPLING LINE CONDITIONS (2D) or SURF CONDITIONS (3D)",
        ],

        "ale_boundary_conditions": {
            "rules": [
                "ALL walls with no-slip fluid BC: apply ALE Dirichlet (fix mesh)",
                "Inflow boundary: apply ALE Dirichlet (fix mesh)",
                "Outflow boundary: apply ALE Dirichlet (fix mesh)",
                "Cylinder/obstacle surfaces: apply ALE Dirichlet (fix mesh)",
                "FSI interface: do NOT apply ALE Dirichlet (mesh moves with structure)",
            ],
            "common_mistake": (
                "Forgetting ALE Dirichlet on some outer boundary causes the "
                "ALE mesh to distort freely, leading to inverted elements."
            ),
        },

        "valid_2d_elements": {
            "FLUID": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "SOLID (structure)": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "notes": (
                "QUAD4 most validated. TRI3 less accurate for "
                "pressure. NOTE: legacy 'WALL' eletype was "
                "renamed to 'SOLID' in 4C 2026.3 — see the [API] "
                "pitfall in SOL_MECH for the parobjectfactory.cpp "
                "error you get if you write 'WALL QUAD4'."
            ),
        },

        "pitfalls": [
            (
                "[Reference] FSI is the most complex problem "
                "type in 4C — three coupled fields (structure "
                "+ fluid + ALE), each needs its own DYNAMIC "
                "section + SOLVER, plus FSI DYNAMIC/MONOLITHIC "
                "or PARTITIONED SOLVER + CLONING MATERIAL MAP. "
                "Signal: an FSI input missing any of the "
                "required sections (PROBLEM TYPE / STRUCTURAL "
                "DYNAMIC / FLUID DYNAMIC / ALE DYNAMIC / FSI "
                "DYNAMIC / CLONING MATERIAL MAP) aborts at "
                "setup with 'missing required section' from "
                "4C_io_input_file.cpp — work from a tutorial "
                "instead of greenfield. (Audit 2026-06-02.)"
            ),
            (
                "[Input] FSI fluid elements MUST set "
                "NA: ALE (not Euler) in the FLUID GEOMETRY "
                "ELEMENT_BLOCKS entry. Signal: leaving NA: "
                "Euler triggers 'fluid element type "
                "incompatible with ALE mesh motion' at setup, "
                "OR (worse) the simulation runs but the fluid "
                "mesh does NOT move with the structure — "
                "interface velocities mismatch and Newton "
                "diverges within ~10 steps. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] ALE Dirichlet BCs MUST be applied on "
                "ALL outer fluid boundaries except the FSI "
                "interface (where the mesh follows the "
                "structure). Signal: missing ALE Dirichlet on "
                "an outflow / outer wall lets the ALE mesh "
                "drift freely there, producing inverted "
                "elements within ~5-20 steps and "
                "'det(J) < 0' from the ALE solver — "
                "simulation aborts. The ALE Dirichlet pins "
                "the mesh at fluid-domain edges. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] CLONING MATERIAL MAP is REQUIRED: it "
                "maps the fluid material ID to a derived ALE "
                "(St. Venant-Kirchhoff pseudo-) material. "
                "Signal: missing CLONING MATERIAL MAP aborts "
                "with 'cannot clone material for ALE field' "
                "from 4C_adapter_fld_base_algorithm. Standard "
                "form: SRC_FIELD: fluid, SRC_MAT: <fluid_id>, "
                "TAR_FIELD: ale, TAR_MAT: <ale_id>. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] SHAPEDERIVATIVES: true is REQUIRED "
                "in FSI DYNAMIC/MONOLITHIC SOLVER for "
                "monolithic schemes — accounts for the "
                "derivative of the fluid residual w.r.t. ALE "
                "displacement in the Jacobian. Signal: with "
                "SHAPEDERIVATIVES: false, the monolithic "
                "Newton iteration is missing a term and "
                "shows linear (not quadratic) convergence; "
                "for partitioned algorithms the flag is "
                "irrelevant. (Audit 2026-06-02.)"
            ),
            (
                "[Input] Each FSI field (structure, fluid, "
                "ALE) needs its OWN SOLVER N entry, "
                "referenced by LINEAR_SOLVER: N in the "
                "respective DYNAMIC section. Signal: "
                "referencing a SOLVER that is not defined "
                "raises 'SOLVER N not found' at setup; "
                "reusing one SOLVER for all three fields is "
                "ALLOWED but typically suboptimal (e.g. "
                "structure benefits from CG+ML, fluid from "
                "GMRES+ILU, ALE from direct UMFPACK). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] FSI coupling-condition sections "
                "differ by spatial dimension: 2D uses "
                "DESIGN FSI COUPLING LINE CONDITIONS, 3D "
                "uses DESIGN FSI COUPLING SURF CONDITIONS. "
                "Signal: a 2D problem with SURF CONDITIONS "
                "(or vice versa) silently has ZERO coupling "
                "nodes — the FSI interface is degenerate and "
                "structure / fluid evolve independently; "
                "neither one diverges, but the structural "
                "deformation does not affect the flow. "
                "Sanity: count DOF-coupling rows in the "
                "Jacobian. (Audit 2026-06-02.)"
            ),
            (
                "[Input] Field NUMDOF: structure uses NUMDOF "
                "matching dimension (2 or 3); fluid uses "
                "NUMDOF = dim + 1 (extra DOF is pressure). "
                "Signal: a structural Dirichlet with "
                "NUMDOF=3 on a 2D problem (or NUMDOF=2 on a "
                "3D problem) aborts at setup with 'invalid "
                "NUMDOF' — the field's DOF count is fixed by "
                "the physics. Fluid always +1 vs structure "
                "for the pressure unknown. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] DESIGN LINE DIRICH CONDITIONS in "
                "FSI applies to ALL discretisations "
                "containing a node — structure AND fluid AND "
                "ALE. Signal: a shared node between "
                "structure (NUMDOF=2) and fluid (NUMDOF=3) "
                "hit by a Dirichlet with NUMDOF=2 raises a "
                "'NUMDOF mismatch' from 4C_dofset.cpp. "
                "Workarounds: (a) offset structural mesh "
                "slightly to avoid shared nodes, "
                "(b) mortar coupling with non-conforming "
                "meshes, (c) remove structural Dirichlet "
                "and rely on FSI coupling. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] DESIGN FLUID LINE LIFT&DRAG does NOT "
                "exist in 4C for 2D. Signal: writing it in a "
                "2D FSI input raises 'unknown section "
                "DESIGN FLUID LINE LIFT&DRAG' from "
                "4C_io_input_spec_builders.cpp. For 2D "
                "lift/drag, set LIFTDRAG: true in FLUID "
                "DYNAMIC — 4C computes it automatically from "
                "the no-slip boundaries. SURF LIFT&DRAG "
                "exists for 3D only. (Audit 2026-06-02.)"
            ),
            (
                "[Syntax] IO section has NO EVERY_ITERATION "
                "parameter — that is not valid in 4C. Signal: "
                "writing 'EVERY_ITERATION: true' in IO "
                "aborts with 'unknown parameter "
                "EVERY_ITERATION' at parse time. Use "
                "RESULTSEVERY in each field's DYNAMIC section "
                "(STRUCTURAL DYNAMIC, FLUID DYNAMIC, ALE "
                "DYNAMIC) to control output frequency per "
                "field. (Audit 2026-06-02.)"
            ),
            (
                "[Syntax] FUNCT with "
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME + VARIABLE "
                "requires COMPONENT: 0 in the same list "
                "item. Signal: without COMPONENT, the "
                "VARIABLE definition is silently ignored and "
                "the function returns wrong values — an "
                "inflow ramp stays stuck at 0 instead of "
                "ramping up. Compare evaluated function "
                "output against an analytic expression to "
                "catch the silent miss. SYMBOLIC_FUNCTION_OF_"
                "TIME (pure time) does NOT need COMPONENT. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] Monolithic FSI requires SEPARATE "
                "nodes at the FSI interface — structure and "
                "fluid must NOT share nodes. Signal: a single "
                "Gmsh mesh shares nodes, and 4C reports 'no "
                "FSI interface nodes found' or runs without "
                "coupling (fluid and solid never exchange "
                "forces). Post-process Gmsh to duplicate "
                "interface nodes and remap fluid connectivity, "
                "OR use mortar coupling "
                "(iter_mortar_monolithicfluidsplit) which "
                "handles non-matching meshes natively. "
                "(Audit 2026-06-02.)"
            ),
            "[API] 4C 2026.3 2D structural element name is "
            "'SOLID QUAD4' (NOT 'WALL QUAD4'). The eletype "
            "string 'WALL' triggers 'PROC 0 ERROR ... Unknown "
            "type WALL of finite element' from "
            "core/comm/src/4C_comm_parobjectfactory.cpp:153. "
            "The legacy WALL eletype was replaced by the "
            "unified SOLID eletype + cell-type variants. "
            "Real syntax in tests/input_files/contact2D_*.4C.yaml: "
            "'1 SOLID QUAD4 ... MAT 1 KINEM nonlinear "
            "THICKNESS 1.0 PLANE_ASSUMPTION plane_strain' — "
            "note THICKNESS (not THICK) and PLANE_ASSUMPTION "
            "(not STRESS_STRAIN). Signal: stderr contains "
            "'Unknown type \\'WALL\\' of finite element'; "
            "swapping to 'SOLID QUAD4' + THICKNESS + "
            "PLANE_ASSUMPTION lets the discretization reach "
            "fill_complete. (Verified empirically 2026-06-01 "
            "— Tier-2 fixture structural_2d_solid_quad4_not_wall "
            "in scripts/tier2_fixtures/fourc/.)",
            (
                "[Output] IO/RUNTIME VTK OUTPUT/STRUCTURE may "
                "CONFLICT with FSI (INT_STRATEGY override). "
                "Signal: an FSI input with that section "
                "aborts with 'inconsistent integration "
                "strategy' from FSI setup phase; removing "
                "the section and using post_vtu after the "
                "simulation succeeds. The override happens "
                "inside the FSI adapter, not the user input. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Output] 2D fluid VTK output may show NaN "
                "pressure and garbage vz component — this "
                "is a VTK output artifact, NOT divergence. "
                "Signal: ParaView opening the IO/RUNTIME VTK "
                "OUTPUT FLUID .pvd shows pressure = NaN over "
                "the entire 2D FLUID3 domain while the "
                "simulation logs report convergence; the "
                "native HDF5 .result files contain the "
                "correct pressure. Check vx/vy (correct in "
                "2D) and convergence logs (residual "
                "decreasing) to confirm — the issue is "
                "output, not solve. (Audit 2026-06-02.)"
            ),
            (
                "[Mesh] For complex FSI geometries (e.g. flag "
                "attached to cylinder): offset the flag "
                "slightly (e.g. 0.1mm gap) to avoid Gmsh "
                "fragment operations that create non-quad-"
                "meshable surfaces. Signal: a flag glued to "
                "a cylinder produces a degenerate "
                "intersection edge that Gmsh can only mesh "
                "with TRI3 (not QUAD4) — typically 100x more "
                "elements than a clean offset geometry; or "
                "Gmsh aborts with 'cannot quad-mesh non-"
                "planar fragment'. A 0.1mm gap is a "
                "negligible geometric approximation that "
                "vastly simplifies meshing. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] FSI SLAVE interface CANNOT carry "
                "Dirichlet BCs. Signal: with "
                "iter_monolithicstructuresplit "
                "(structure=slave), a structural Dirichlet "
                "on a node that also belongs to the FSI "
                "coupling interface aborts with 'slave node "
                "carries Dirichlet' from 4C_fsi_monolithic_"
                "structuresplit.cpp. Fix: switch to "
                "iter_monolithicfluidsplit (structure=master) "
                "or exclude the overlapping nodes from the "
                "FSI interface. (Audit 2026-06-02.)"
            ),
            (
                "[Output] IO/RUNTIME VTK OUTPUT/ALE does NOT "
                "exist — it crashes 4C. Signal: writing /ALE "
                "as a subsection causes an immediate parse "
                "failure with 'unknown subsection ALE in "
                "IO/RUNTIME VTK OUTPUT' from "
                "4C_io_input_spec_builders.cpp. Only "
                "/STRUCTURE and /FLUID subsections are valid "
                "for FSI VTK output. For ALE fields, use "
                "post_processor --filter=vtu on native "
                "output instead. (Audit 2026-06-02.)"
            ),
            (
                "[Input] Valid COUPALGO values for monolithic "
                "FSI: iter_monolithicfluidsplit "
                "(structure=master, recommended), "
                "iter_monolithicstructuresplit "
                "(structure=slave), "
                "iter_mortar_monolithicfluidsplit (non-"
                "matching meshes), "
                "iter_sliding_monolithicfluidsplit (sliding "
                "interface). For partitioned: "
                "iter_stagg_AITKEN_rel_force (default), "
                "iter_stagg_fixed_rel_force. Signal: a "
                "mis-spelled COUPALGO value aborts with "
                "'unknown coupling algorithm' from "
                "4C_fsi_adapter.cpp — copy verbatim from "
                "this list. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Inflow ramp rate affects FSI "
                "stability. Signal: a fast inflow ramp "
                "(e.g. step or 1s rise) over a flexible "
                "structure produces Newton divergence "
                "within ~10 time steps even at laminar "
                "Re — the structural response cannot follow "
                "the fluid forcing transient. For initial "
                "testing, use a slow ramp (5-10s period, "
                "e.g. cos(pi*t/5)) rather than the standard "
                "Turek-Hron 2s ramp. Once stable, gradually "
                "decrease the ramp period. (Audit "
                "2026-06-02.)"
            ),
        ],
    },

    "ssi": {
        "description": "Structure-Scalar Interaction (e.g., battery electrode mechanics)",
        "problemtype": "Structure_Scalar_Interaction",
        "coupling_types": ["OneWay_ScatraToSolid", "OneWay_SolidToScatra",
                          "IterStagg", "IterStaggFixedRel", "IterStaggAitken", "Monolithic"],
    },

    "ssti": {
        "description": "Structure-Scalar-Thermo Interaction (three-field coupling)",
        "problemtype": "Structure_Scalar_Thermo_Interaction",
        "coupling": "Monolithic (all three fields simultaneously)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CONTACT MECHANICS
    # ═══════════════════════════════════════════════════════════════════════
    "contact": {
        "description": "Contact mechanics with multiple enforcement methods",
        "methods": {
            "penalty": "Penalty method (simple, parameter-dependent)",
            "lagrange": "Lagrange multiplier (exact enforcement, saddle-point)",
            "nitsche": "Nitsche method (consistent, no extra DOFs)",
            "mortar": "Mortar method (surface integration, non-matching meshes)",
        },
        "variants": ["Standard contact", "Self-contact (binary tree search)",
                     "Wear contact", "Friction (Coulomb)",
                     "TSI contact", "Poro contact", "FSI contact", "SSI contact"],
        "constitutive_laws": ["Linear", "Cubic", "Power law", "Broken rational",
                              "MIRCO (microscale)", "Python surrogate"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PARTICLE METHODS
    # ═══════════════════════════════════════════════════════════════════════
    "particles": {
        "description": "Particle methods: SPH, DEM, Peridynamics",
        "problemtype": "Particle",

        "sph": {
            "kernels": ["CubicSpline (default)", "QuinticSpline"],
            "eos": ["GenTait (generalized Tait)", "IdealGas"],
            "momentum": ["Adami formulation", "Monaghan formulation"],
            "density": ["Summation", "Integration", "Predict-Correct"],
            "boundary": ["Adami boundary particles", "Virtual wall particles"],
            "extra_physics": ["Surface tension (CSF)", "Phase change", "Temperature"],
        },

        "dem": {
            "contact_normal": ["LinearSpring", "LinearSpringDamp", "Hertz",
                               "LeeHerrmann", "KuwabaraKono", "Tsuji"],
            "contact_tangential": ["None", "LinearSpringDamp"],
            "rolling": ["None", "Viscous", "Coulomb"],
            "adhesion": ["None", "VdWDMT", "RegDMT"],
        },

        "peridynamics": {
            "dimensions": ["3D (Peridynamic_3D)", "2D Plane Stress (Peridynamic_2DPlaneStress)",
                          "2D Plane Strain (Peridynamic_2DPlaneStrain)"],
            "features": ["Bond-based PD", "Damage via critical stretch criterion",
                        "Volume correction factor", "Pre-crack definition via line segments"],
            "material": "MAT_ParticlePD: INITRADIUS, INITDENSITY, YOUNG, CRITICAL_STRETCH",
            "input_section": "PARTICLE DYNAMIC/PD",
            "key_params": {
                "INTERACTION_HORIZON": "delta = m * dx (typically m=3, so horizon = 3*particle_spacing)",
                "PERIDYNAMIC_GRID_SPACING": "dx (particle spacing, must match actual particle grid)",
                "PD_DIMENSION": "Peridynamic_2DPlaneStrain / Peridynamic_2DPlaneStress / Peridynamic_3D",
                "PRE_CRACKS": "Line segments: 'x1 y1 x2 y2 ; x3 y3 x4 y4' — bonds crossing these are pre-broken",
                "NORMALCONTACTLAW": "NormalLinearSpring (for impactor-body contact)",
                "NORMAL_STIFF": "Contact stiffness (e.g., 1.0e4)",
            },
            "particle_grid_generation": {
                "description": "PD requires a REGULAR GRID of particles with sufficient resolution",
                "pattern": "Loop over nx*ny (2D) or nx*ny*nz (3D) with uniform spacing dx",
                "spacing": "dx should be chosen based on the problem scale; horizon = m*dx (m=3 typical)",
                "notches_cracks": "Skip particles inside notch gaps OR use PRE_CRACKS line segments",
                "example": "for iy in range(ny): for ix in range(nx): particles.append((ix*dx, iy*dx, 0.0))",
                "convergence": "PD converges as dx→0 AND m→∞ (delta-convergence AND m-convergence)",
            },
        },

        "time_integration": ["Semi-implicit Euler (SemiImplicitEuler)", "Velocity Verlet (VelocityVerlet)"],

        "vtk_output": {
            "description": "CRITICAL: Particle VTK output must be explicitly configured",
            "yaml_section": """
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 10
IO/RUNTIME VTK OUTPUT/PARTICLES:
  PARTICLE_OUTPUT: true
  DISPLACEMENT: true
  VELOCITY: true
  ACCELERATION: false
  OWNER: true""",
            "output_format": "VTP (VTK PolyData) files, one per time step, with PVD time series",
            "pitfall": "Without IO/RUNTIME VTK OUTPUT/PARTICLES section, 4C produces NO particle output files!",
        },

        "mandatory_sph_section": {
            "description": "Even for PURE peridynamics, the SPH section is MANDATORY in 4C",
            "reason": "The PD implementation lives inside the SPH interaction framework. Without SPH section, pd_neighbor_pairs=0 → no PD forces computed",
            "yaml": """
PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: 1.0
  BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
  TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity""",
        },

        "impactor_setup": {
            "description": "Rigid impactor as boundary phase particles",
            "material": "MAT_ParticleSPHBoundary: INITRADIUS, INITDENSITY",
            "phase_mapping": "PHASE_TO_MATERIAL_ID: 'boundaryphase 1 pdphase 2'",
            "velocity": "Applied via FUNCT + DIRICHLET_BOUNDARY_CONDITION on boundaryphase",
        },

        "pitfalls": [
            (
                "[Input] PARTICLE DYNAMIC/SPH section is "
                "MANDATORY even for PURE peridynamics — the "
                "PD implementation lives inside the SPH "
                "interaction framework. Signal: omitting SPH "
                "section gives pd_neighbor_pairs = 0 at "
                "runtime (visible in stderr) and zero "
                "displacement, with NO error message — 4C "
                "happily runs a no-force simulation. Add the "
                "SPH block with KERNEL: QuinticSpline, "
                "KERNEL_SPACE_DIM: Kernel2D (or 3D), "
                "INITIALPARTICLESPACING matching dx. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Output] IO/RUNTIME VTK OUTPUT/PARTICLES "
                "must be added for ParaView output (VTP "
                "files). Signal: a PD simulation runs to "
                "completion but no .vtp / .pvd files are in "
                "the output directory — 4C produces native "
                "files only, no particle output unless the "
                "PARTICLES subsection is configured with "
                "PARTICLE_OUTPUT: true. (Audit 2026-06-02.)"
            ),
            (
                "[Input] PD requires a REGULAR particle grid "
                "(uniform spacing in all directions). "
                "Signal: a non-uniform / refined particle "
                "set produces visibly anisotropic wave "
                "propagation in PD (waves travel faster in "
                "dense regions) and wrong fracture patterns "
                "— PD bond stiffness depends on uniform "
                "spacing dx. Generate particles on a regular "
                "grid (e.g. nx*ny loop with uniform dx). "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] INTERACTION_HORIZON must equal m * "
                "dx where m is the horizon ratio (typically "
                "3). Signal: setting INTERACTION_HORIZON < "
                "2*dx in PARTICLE DYNAMIC / PD gives a "
                "MAT_ParticlePD model with each particle "
                "only seeing 1-2 neighbours — bond count is "
                "too sparse, stiffness is mesh-dependent "
                "and convergence as dx -> 0 fails. m=3 is "
                "the minimum for delta-convergence to "
                "classical elasticity. (Audit 2026-06-02.)"
            ),
            (
                "[Input] PERIDYNAMIC_GRID_SPACING in the "
                "input must EXACTLY match the actual "
                "particle spacing in the mesh. Signal: a "
                "mismatch (e.g. PERIDYNAMIC_GRID_SPACING: "
                "0.1 but actual particles at 0.05 spacing) "
                "produces wrong volume corrections at the "
                "horizon — fracture stress is off by 2x or "
                "more vs analytic Griffith load. Verify dx "
                "by computing min pairwise distance between "
                "first 10 particles. (Audit 2026-06-02.)"
            ),
            (
                "[Input] PRE_CRACKS uses semicolon-separated "
                "line segments: 'x1 y1 x2 y2 ; x3 y3 x4 y4'. "
                "Signal: mis-formatted PRE_CRACKS in PARTICLE "
                "DYNAMIC / PD (e.g. comma separator, or "
                "missing semicolons between segments) parses "
                "as ONE crack with concatenated endpoints — "
                "MAT_ParticlePD bonds across all spurious "
                "segments break instead of just the intended "
                "ones; the initial damage pattern visualised "
                "in ParaView from the PARTICLES VTK output "
                "reveals the wrong geometry. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] PDBODYID must be specified for PD "
                "phase particles (e.g. PDBODYID 0). Signal: "
                "omitting PDBODYID gives all PD particles "
                "the default body ID -1; force assembly is "
                "applied across body boundaries that should "
                "be separate, producing non-physical "
                "coupling between bodies (e.g. an impactor "
                "experiences PD bonds with its target). "
                "Each distinct body needs a unique PDBODYID. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Input] Boundary phase particles (impactor) "
                "need TYPE boundaryphase; PD particles need "
                "TYPE pdphase. Signal: swapping TYPE between "
                "impactor and target makes 4C apply "
                "boundary-phase contact law where PD bonds "
                "are expected and vice versa — the impactor "
                "either passes through the target (no "
                "contact reaction) or sticks to it (no "
                "rebound). Verify TYPE per phase in the "
                "PHASE_TO_MATERIAL_ID mapping. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] CFL condition for PD: dt < "
                "0.5 * dx / c_wave where c_wave = sqrt(E/"
                "rho). Signal: dt > CFL gives NaN within "
                "~10 time steps (typical 'energy not "
                "conserved' message); reducing dt by 2x at "
                "a time until stable. For PD with damage, "
                "safety factor 0.3 is more conservative "
                "than 0.5 because cracks reduce effective "
                "stiffness and increase wave speed. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] BINNING STRATEGY's "
                "BIN_SIZE_LOWER_BOUND must be > horizon for "
                "correct neighbour search. Signal: too "
                "small a bin (< horizon) misses neighbour "
                "pairs at bin boundaries — pd_neighbor_pairs "
                "drops below the expected ~ 4*pi*delta^2 / "
                "dx^2 per particle, fracture pattern "
                "develops spurious gaps at bin boundaries. "
                "Set BIN_SIZE_LOWER_BOUND >= horizon, "
                "ideally 1.5 * horizon. (Audit 2026-06-02.)"
            ),
            (
                "[Input] DOMAINBOUNDINGBOX must enclose ALL "
                "particles INCLUDING the impactor motion "
                "range. Signal: an impactor moving outside "
                "the original bounding box triggers "
                "'particle out of domain' from 4C particle "
                "engine — simulation aborts mid-run. Set "
                "the bbox larger than the initial particle "
                "extent by at least the maximum expected "
                "impactor displacement over the simulation. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # POROUS MEDIA
    # ═══════════════════════════════════════════════════════════════════════
    "porous_media": {
        "description": "Biot poroelasticity and porous flow",
        "problem_types": {
            "Poroelasticity": "Biot consolidation (structure + fluid in pores)",
            "Poroelastic_scalar_transport": "Poro + scalar transport",
            "porofluid_pressure_based": "Pressure-based porous flow (standalone)",
        },
        "coupling": ["Monolithic", "Partitioned", "1-way", "2-way"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CARDIOVASCULAR / BIOMEDICAL
    # ═══════════════════════════════════════════════════════════════════════
    "cardiovascular": {
        "description": "Cardiovascular and biomedical simulation capabilities",
        "models": {
            "0D_windkessel": "4-element Windkessel for arterial pressure",
            "arterial_network": "1D arterial blood flow network (artery elements)",
            "reduced_airways": "Reduced lung airways with acinus elements",
            "cardiac_monodomain": "Cardiac electrophysiology (FHN, TenTusscher, etc.)",
        },
        "applications": "Arterial hemodynamics, cardiac mechanics, lung ventilation",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # BEAM INTERACTION
    # ═══════════════════════════════════════════════════════════════════════
    "beam_interaction": {
        "description": "Beam-to-beam, beam-to-solid, beam-to-sphere contact and meshtying",
        "contact_pairs": [
            "Beam-to-beam (point coupling, tangent smoothing)",
            "Beam-to-solid volume meshtying (Gauss point, mortar)",
            "Beam-to-solid surface meshtying",
            "Beam-to-solid surface contact",
            "Beam-to-sphere contact",
        ],
        "cross_linking": "Pin-jointed, rigid-jointed, truss links (biopolymer networks)",
        "brownian_dynamics": "Stochastic dynamics of beam networks",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # LINEAR SOLVERS
    # ═══════════════════════════════════════════════════════════════════════
    "solvers": {
        "direct": {
            "UMFPACK": "Serial direct solver (recommended for small problems)",
            "SuperLU": "Parallel direct solver (SuperLU_Dist)",
            "MUMPS": "Parallel direct solver (MPI, recommended for large problems)",
            "KLU2": "Serial direct solver (alternative to UMFPACK)",
        },
        "iterative": {
            "CG": "Conjugate gradient (symmetric positive definite systems only)",
            "GMRES": "Generalized minimal residual (non-symmetric systems)",
            "BiCGSTAB": "Bi-conjugate gradient stabilized (non-symmetric, lower memory)",
        },
        "preconditioners": {
            "ILU": "Incomplete LU factorization (Ifpack package)",
            "MueLu": "Algebraic multigrid (MueLu, recommended for large problems)",
            "Block_Teko": "Block preconditioning for multi-field problems (Teko package)",
        },
        "nonlinear": {
            "NOX": "Trilinos NOX framework (Newton + line search + PTC + convergence tests)",
        },
        "yaml_example": """
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
SOLVER 2:
  SOLVER: "Belos"
  SOLVER_XML_FILE: "iterative_gmres_template.xml"
  AZPREC: "MueLu"
  MUELU_XML_FILE: "elasticity_template.xml"
  NAME: "iterative_solver"
""",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT FILE FORMAT
    # ═══════════════════════════════════════════════════════════════════════
    "input_format": {
        "description": "YAML-based input files (.4C.yaml) — can use inline mesh or Exodus file",

        "mandatory_sections": [
            "PROBLEM SIZE (DIM: 2 or 3)",
            "PROBLEM TYPE (PROBLEMTYPE: Structure/Scalar_Transport/Fluid/...)",
            "Dynamics section matching problem type (STRUCTURAL DYNAMIC, etc.)",
            "At least one SOLVER",
            "MATERIALS",
            "Mesh (NODE COORDS + ELEMENTS, or STRUCTURE GEOMETRY with FILE)",
        ],

        "boundary_conditions": {
            "structural": {
                "DESIGN POINT/LINE/SURF/VOL DIRICH CONDITIONS": "Prescribed displacement",
                "DESIGN POINT/LINE/SURF/VOL NEUMANN CONDITIONS": "Applied force/traction/body force",
            },
            "thermal": {
                "DESIGN SURF/VOL THERMO DIRICH CONDITIONS": "Prescribed temperature",
                "DESIGN SURF/VOL THERMO NEUMANN CONDITIONS": "Applied heat flux",
            },
            "bc_format": """
DESIGN SURF DIRICH CONDITIONS:
  - E: 1            # Design entity ID
    NUMDOF: 3       # Number of DOFs per node
    ONOFF: [1, 1, 0] # Which DOFs are constrained (1=yes, 0=no)
    VAL: [0.0, 0.0, 0.0]  # Prescribed values
    FUNCT: [0, 0, 0]       # Time function IDs (0=constant)
""",
        },

        "topology_sections": {
            "DNODE-NODE TOPOLOGY": "Map single nodes to design nodes (for point BCs)",
            "DLINE-NODE TOPOLOGY": "Map nodes to design lines (for line BCs in 2D)",
            "DSURF-NODE TOPOLOGY": "Map nodes to design surfaces (for surface BCs in 3D)",
            "DVOL-NODE TOPOLOGY": "Map nodes to design volumes (for volume BCs)",
        },

        "functions": """
# --- Simple space-time function (no time-varying sub-variables) ---
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "sin(2*pi*x)*cos(pi*t)"
  # Supports: x, y, z, t as variables

# --- Function with VARIABLE (e.g. ramp-up) ---
# IMPORTANT: COMPONENT: 0 is REQUIRED when using VARIABLE/multifunction.
# Without COMPONENT, the VARIABLE definition is NOT parsed correctly
# and the function silently returns wrong values.
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*U_bar*y*(H-y)/(H*H)*a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "multifunction"
    NUMPOINTS: 3
    TIMES: [0, 2, 10000]
    DESCRIPTION: ["0.5*(1-cos(pi*t/2))", "1.0"]

# --- Pure time function (no COMPONENT needed) ---
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_TIME: "a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "multifunction"
    NUMPOINTS: 3
    TIMES: [0, 1, 10000]
    DESCRIPTION: ["0.5*(1.0-cos((t*pi)/1.0))", "1.0"]

# --- Linear interpolation (piecewise linear in time) ---
FUNCT4:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1*a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "linearinterpolation"
    NUMPOINTS: 3
    TIMES: [0, 1, 101]
    VALUES: [0, 1, 100]
""",

        "inline_mesh_example": """
NODE COORDS:
  - "NODE 1 COORD 0.000000 0.000000 0.0"
  - "NODE 2 COORD 1.000000 0.000000 0.0"
TRANSPORT ELEMENTS:
  - "1 TRANSP QUAD4 1 2 3 4 MAT 1 TYPE Std"
""",

        # 2026-06-01 (critic-audit #5): renamed from
        # 'general_pitfalls' so the verify_signal_clauses /
        # orphan / parse-discipline harnesses (which key off
        # the literal 'pitfalls' field) can see these entries.
        # Combined with the fourc backend exposing
        # 'input_format' as a [Reference] PhysicsCapability,
        # users now reach them via discover + knowledge +
        # prepare_simulation.
        "pitfalls": [
            # ExodusII block IDs
            "[API] CRITICAL: meshio (Python) writes ExodusII "
            "element block IDs starting at 0 (0-indexed), but "
            "4C YAML ELEMENT_BLOCKS use 1-indexed IDs. Signal: "
            "4C stderr emits the cryptic 'Pressure map empty' "
            "(or analogous map-empty errors for transport / "
            "fluid / structure) at problem-setup time — the "
            "block-ID mismatch makes 4C find zero elements of "
            "the expected family. Fix: after writing with "
            "meshio, patch with netCDF4 — "
            "import netCDF4; ds = netCDF4.Dataset("
            "'mesh.e', 'r+'); ds.variables['eb_prop1'][:] += "
            "1; ds.close(). Verify with: python3 -c \"import "
            "meshio; m = meshio.read('mesh.e'); print([c.type "
            "for c in m.cells])\". (Audit 2026-06-02.)",

            # FUNCT COMPONENT requirement
            "[Syntax] SYMBOLIC_FUNCTION_OF_SPACE_TIME with "
            "VARIABLE/multifunction REQUIRES 'COMPONENT: 0' "
            "in the same list item. Signal: omitting COMPONENT "
            "does NOT raise an error at parse time — the "
            "VARIABLE definition is silently ignored and the "
            "function returns the WRONG values (the variable "
            "expression evaluates to 0 everywhere). A "
            "Dirichlet BC driven by such a function stays "
            "stuck at 0 instead of ramping up; comparing "
            "results vs an analytic ramp exposes the silent "
            "miss. SYMBOLIC_FUNCTION_OF_TIME (pure time "
            "functions) do NOT need COMPONENT. (Audit "
            "2026-06-02.)",

            # Shared-node NUMDOF conflict
            "[API] In multi-physics problems (FSI, TSI, SSI), "
            "DESIGN ... DIRICH CONDITIONS apply to ALL "
            "discretisations containing a node. Signal: a node "
            "shared between structure (NUMDOF=2 in 2D) and "
            "fluid (NUMDOF=3) hit by a Dirichlet with "
            "NUMDOF=2 raises 'inconsistent NUMDOF on shared "
            "node' (or equivalent dof_check failure) from "
            "4C_io_input_spec.cpp during setup. Solutions: "
            "(a) use separate node sets per discretisation, "
            "(b) offset meshes to avoid shared nodes at "
            "Dirichlet boundaries, (c) use mortar coupling "
            "with non-matching meshes. (Audit 2026-06-02.)",

            # Invalid section names
            "[Syntax] 4C is STRICT about section names. Common "
            "invalid sections: EVERY_ITERATION (not a valid IO "
            "parameter), DESIGN FLUID LINE LIFT&DRAG (does not "
            "exist for 2D), DESIGN THERMO LINE DIRICH "
            "CONDITIONS (wrong — must be DESIGN LINE THERMO "
            "DIRICH CONDITIONS). Signal: 4C aborts with "
            "'unknown section' or 'Could not match this input' "
            "from 4C_io_input_spec_builders.cpp at parse time, "
            "echoing the offending YAML block. Check valid "
            "names with: 4C --parameters | grep DESIGN. "
            "(Audit 2026-06-02.)",

            # Output
            "[API] 4C writes native .control/.mesh/.result "
            "files. To get VTU output for ParaView, either: "
            "(a) add IO/RUNTIME VTK OUTPUT sections "
            "(recommended), or (b) run post_vtu --file="
            "output_prefix AFTER the simulation. Signal: "
            "after PROBLEMTYPE / STRUCTURAL DYNAMIC / FLUID "
            "DYNAMIC / SCATRA DYNAMIC run completes, looking "
            "for a .vtu / .pvd output file in the results "
            "directory finds nothing — 4C only produced "
            ".control / .mesh / .result; either the IO/"
            "RUNTIME VTK OUTPUT section is missing or "
            "post_vtu was not invoked. The native files are "
            "HDF5-readable but not directly ParaView-"
            "loadable. (Audit 2026-06-02.)",

            # 2026-06-01: .dat extension rejected
            "[Syntax] 4C 2026.3.0-dev accepts ONLY .yaml / .yml / .json input "
            "files. Passing a legacy .dat-format file (or any other extension) "
            "is rejected at file-open time with 'Cannot infer format of input "
            "file ... Only .yaml, .yml, and .json are supported.' from "
            "core/io/src/4C_io_input_file.cpp. Note: the section-name "
            "vocabulary (PROBLEM TYPE, STRUCTURAL DYNAMIC, DESIGN SURF NEUMANN, "
            "MAT_scatra, etc.) is unchanged — those are still valid as YAML "
            "keys; only the overall file format moved from dat-style "
            "section-header text to YAML mapping syntax. Signal: 4C ERROR "
            "from 4C_io_input_file.cpp with 'Cannot infer format' and "
            "'Only .yaml, .yml, and .json' substrings when an unsupported "
            "extension is passed on the CLI. (Verified empirically 2026-06-01.)",

            "[Syntax] 4C validates enum-like keys against an allowed set at "
            "input-parse time. Mis-spelling a PROBLEMTYPE value (e.g. "
            "'Hyperelasticity' instead of 'Structure', or a typo like "
            "'Scalar_Tranzport') triggers 'PROC 0 ERROR ... Could not match "
            "this input' from core/io/src/4C_io_input_spec_builders.cpp, "
            "with the offending YAML block echoed in the message. Signal: "
            "the substrings 'Could not match this input', 'PROBLEMTYPE', and "
            "'input_spec_builders' all appear in 4C stderr when the value is "
            "not in the allowed enum set. (Verified empirically 2026-06-01.)",

            # THICKNESS parameter for 2D plane strain
            "[Input] For 2D plane-strain SOLID elements, "
            "THICKNESS is the out-of-plane depth (unit "
            "thickness), NOT the element width. Almost always "
            "THICKNESS: 1.0. Signal: THICKNESS set to the "
            "element edge length (or some geometric width) "
            "silently scales ALL forces and stresses by that "
            "factor — total reaction force at a fixed edge "
            "is off by exactly THICKNESS, no error from 4C. "
            "Sanity: integrate sigma_xx over a cross-section "
            "and compare to applied force / THICKNESS. NOTE: "
            "the legacy keyword 'THICK' was renamed to "
            "'THICKNESS' along with the WALL -> SOLID eletype "
            "change in 4C 2026.3. (Audit 2026-06-02.)",

            # 2D VTK output artifacts — applies to fluid AND porofluid
            "[Output] In 2D simulations, fluid AND porofluid "
            "VTK output may show NaN for pressure and garbage "
            "for the z-velocity component. Signal: opening "
            "the IO/RUNTIME VTK OUTPUT FLUID .vtu in "
            "ParaView for a 2D FLUID3 / POROFLUIDMULTIPHASE "
            "run shows pressure = NaN everywhere "
            "(white/uncolored field) while the simulation "
            "actually converged — the issue is a VTK output "
            "artifact for 2D problems, NOT divergence. "
            "Native HDF5 .result files contain the correct "
            "pressure. Affects fluid, poro, and FSI in 2D; "
            "3D output is unaffected. (Audit 2026-06-02.)",

            # Poro-specific
            "[Numerical] 4C poro uses a DYNAMIC formulation "
            "(with inertia) even for quasi-static problems — "
            "the structural momentum balance retains the "
            "rho*a term. Signal: a step-load applied to a 1D "
            "consolidation column shows elastic-wave "
            "ringing (oscillating pressure / displacement at "
            "frequency ~ c_p/H where c_p = sqrt(E/rho)) — "
            "NOT the smooth Terzaghi consolidation curve. "
            "Fix: ramp the load over a time >> 10 * H / "
            "sqrt(E/rho) (10x wave traversal time) so the "
            "elastic transient damps before consolidation "
            "begins. (Audit 2026-06-02.)",


            # 2D structural element types (post-WALL→SOLID
            # rename in 4C 2026.3).
            "[API] 2D structural elements are 'SOLID QUAD4 / "
            "QUAD8 / QUAD9' and 'SOLID TRI3 / TRI6'. The "
            "legacy 'WALL' eletype was renamed to 'SOLID' — "
            "the new naming covers BOTH 2D (with "
            "PLANE_ASSUMPTION) and 3D (no PLANE_ASSUMPTION) "
            "under one factory string. Signal: writing "
            "'WALL QUAD4' / 'WALL TRI3' raises 'Unknown type "
            "WALL of finite element' from parobjectfactory."
            "cpp:153 at problem setup; the fix is replacing "
            "every 'WALL' eletype string with 'SOLID' and "
            "adding PLANE_ASSUMPTION (plane_strain or "
            "plane_stress) for 2D. (Audit 2026-06-02.)",

            # FSI mesh requirements
            "[API] For monolithic FSI: the structure and "
            "fluid meshes MUST have SEPARATE nodes at the "
            "FSI interface (NOT shared conforming nodes). "
            "Signal: a single Gmsh mesh used for both phases "
            "shares interface nodes, and the FSI coupling "
            "operator detects only zero interface DOFs — "
            "either 4C aborts with 'no FSI interface nodes "
            "found' or the simulation runs without coupling "
            "(fluid and solid never exchange forces, "
            "deformation stays zero). Post-process Gmsh to "
            "duplicate interface nodes, remap connectivity. "
            "Alternative: mortar coupling "
            "(iter_mortar_monolithicfluidsplit) handles "
            "non-matching meshes natively. (Audit "
            "2026-06-02.)",

            # Large inline YAML performance
            "[Performance] For meshes with > 200 nodes, use "
            "an ExodusII mesh file (.e) instead of inline "
            "NODE COORDS + ELEMENTS sections. Signal: an "
            "inline YAML with > 1000 lines takes 30+ seconds "
            "to parse — the MCP stdio transport times out at "
            "60s, and even direct CLI 4C startup is "
            "noticeably slow. Use meshio to write the mesh "
            "to .e format, then reference it with "
            "STRUCTURE GEOMETRY: FILE: mesh.e. (Audit "
            "2026-06-02.)",

            # FSI + runtime VTK
            "[Output] IO/RUNTIME VTK OUTPUT/STRUCTURE may be "
            "INCOMPATIBLE with FSI — FSI overrides "
            "INT_STRATEGY internally. Signal: a structural "
            "VTK section in an FSI input causes 4C to abort "
            "with 'inconsistent integration strategy' or "
            "similar error from the FSI setup phase; removing "
            "the IO/RUNTIME VTK OUTPUT/STRUCTURE section and "
            "using post_vtu after the simulation succeeds. "
            "(Audit 2026-06-02.)",

            # GPU / hardware acceleration
            "[Hardware] 4C linear algebra is CPU-ONLY "
            "(Epetra-based, Trilinos 16.2.0). Epetra does NOT "
            "support GPU execution. Signal: setting "
            "CUDA_VISIBLE_DEVICES, KOKKOS_NUM_DEVICES, or any "
            "GPU-targeted environment variable has zero effect "
            "on 4C runtime — wall-clock for assembly and "
            "linear solves stays identical. Tpetra "
            "(GPU-capable via Kokkos CUDA/HIP/SYCL backends) "
            "is not yet integrated. Plan compute on CPU only. "
            "(Audit 2026-06-02.)",

            # ArborX optional GPU component
            "[Hardware] The ONLY GPU-accelerated component in "
            "4C is ArborX (optional, OFF by default), used "
            "for geometric search (bounding-volume-hierarchy "
            "queries in contact / particle problems). Enable "
            "with cmake flag -DFOUR_C_WITH_ARBORX=ON and a "
            "Kokkos GPU backend in Trilinos. Signal: even "
            "with ArborX-on, the LINEAR SOLVER wall-clock is "
            "unchanged — only the contact-search phase "
            "shrinks; for problems dominated by linear solve "
            "(most), ArborX gives < 5% total speedup. (Audit "
            "2026-06-02.)",

            # MPI parallelism
            "[Hardware] 4C uses MPI for domain decomposition. "
            "Standard invocation: mpirun -np N 4C input.4C."
            "yaml. Signal: forgetting mpirun on a multi-CPU "
            "machine restricts 4C to a single rank — wall-"
            "clock is N-fold higher than expected for a "
            "well-decomposable problem and CPU utilisation "
            "is < 1/N on the system monitor. MPI is the "
            "primary parallelism mechanism; thread-level "
            "parallelism uses OpenMP (set OMP_NUM_THREADS). "
            "Mixing both (mpirun + OMP_NUM_THREADS) is "
            "supported but oversubscribes if "
            "N_mpi * N_omp > N_cores. (Audit 2026-06-02.)",
        ],

        "element_type_per_physics": {
            "FLUID (2D)": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "FLUID (3D)": ["HEX8", "HEX20", "HEX27", "TET4", "TET10", "NURBS27"],
            "SOLID (2D structure)": ["QUAD4", "QUAD8", "QUAD9",
                                     "TRI3", "TRI6"],
            # NOTE: 4C 2026.3 unified the legacy WALL 2D eletype
            # into the SOLID eletype factory. Writing 'WALL QUAD4'
            # raises 'Unknown type WALL of finite element' from
            # parobjectfactory.cpp:153 — see SOL_MECH [API] pitfall.
            "SOLID (3D structure)": ["HEX8", "HEX20", "HEX27", "TET4", "TET10",
                                     "WEDGE6", "PYRAMID5"],
            "TRANSP (scalar transport)": ["QUAD4", "QUAD9", "HEX8", "HEX27",
                                          "TRI3", "TRI6", "TET4", "TET10"],
            "SOLIDSCATRA (TSI/SSI)": ["HEX8", "TET4", "TET10", "HEX27"],
            "ALE (2D)": ["QUAD4", "TRI3"],
            "ALE (3D)": ["HEX8", "TET4"],
            "PORO (2D)": ["WALLQ4PORO", "WALLQ9PORO"],
            "PORO (3D)": ["SOLIDH8PORO", "SOLIDT4PORO", "SOLIDH27PORO"],
            "BEAM": ["BEAM3R LINE2", "BEAM3EB LINE2", "BEAM3R LINE3"],
            "ARTERY": ["ARTERY LINE2"],
            "notes": (
                "QUAD4 is the workhorse element for most 2D problems.  "
                "HEX8 for 3D.  Higher-order elements (QUAD9, HEX27) give "
                "better accuracy but are slower.  TRI3/TET4 are available "
                "but less accurate for pressure in fluid problems."
            ),
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CELL TYPES
    # ═══════════════════════════════════════════════════════════════════════
    "cell_types": {
        "1D": ["line2", "line3", "line4", "line5", "line6", "point1"],
        "2D": ["quad4", "quad6", "quad8", "quad9", "tri3", "tri6"],
        "3D": ["hex8", "hex16", "hex18", "hex20", "hex27", "tet4", "tet10",
               "wedge6", "wedge15", "pyramid5"],
        "NURBS": ["nurbs2", "nurbs3 (1D)", "nurbs4", "nurbs9 (2D)",
                  "nurbs8", "nurbs27 (3D)"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # XFEM
    # ═══════════════════════════════════════════════════════════════════════
    "xfem": {
        "description": "Extended Finite Element Method for interface problems",
        "capabilities": [
            "Level-set based interfaces (weak Dirichlet, Neumann, Navier slip, two-phase)",
            "Surface-based interfaces (displacement, FSI, FPI)",
            "Robin conditions (Dirichlet/Neumann)",
            "Edge stabilization",
            "Semi-Lagrangean time integration",
        ],
        "applications": "Fluid-XFEM, FSI-XFEM (no mesh conformity at interface)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ALL 40 PROBLEM TYPES
    # ═══════════════════════════════════════════════════════════════════════
    "all_problem_types": {
        "Structure": "Structural mechanics",
        "Scalar_Transport": "Convection-diffusion / scalar transport",
        "Thermo": "Pure thermal analysis",
        "Fluid": "Incompressible Navier-Stokes",
        "Fluid_Ale": "Fluid on ALE mesh",
        "Ale": "Pure ALE mesh movement",
        "Fluid_Structure_Interaction": "FSI (standard)",
        "Fluid_Structure_Interaction_XFEM": "FSI with XFEM",
        "Thermo_Structure_Interaction": "TSI",
        "Structure_Scalar_Interaction": "SSI (electrode mechanics, etc.)",
        "Structure_Scalar_Thermo_Interaction": "SSTI (three-field)",
        "Scalar_Thermo_Interaction": "STI",
        "Fluid_Beam_Interaction": "3D fluid + 1D beam",
        "Fluid_Porous_Structure_Interaction": "FPSI",
        "Particle": "SPH / DEM / Peridynamics",
        "Particle_Structure_Interaction": "PASI",
        "Poroelasticity": "Biot poroelasticity",
        "Poroelastic_scalar_transport": "Poro + scalar",
        "Level_Set": "Level-set interface tracking",
        "Low_Mach_Number_Flow": "Variable-density flow",
        "Lubrication": "Thin film lubrication",
        "Elastohydrodynamic_Lubrication": "EHL coupling",
        "Electrochemistry": "Nernst-Planck electrochemistry",
        "ArterialNetwork": "1D arterial blood flow",
        "ReducedDimensionalAirWays": "Lung airways",
        "Cardiac_Monodomain": "Cardiac electrophysiology",
        "Biofilm_Fluid_Structure_Interaction": "Biofilm FSI",
        "Gas_Fluid_Structure_Interaction": "Gas + FSI",
        "Polymer_Network": "Polymer network",
    },
}
