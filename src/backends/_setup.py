"""Install, setup and build-configuration knowledge.

WHY THIS MODULE EXISTS
    Everything else in OASiS's pitfall database describes what happens once a
    backend runs. This file describes the part before that: getting the backend
    installed, letting OASiS find it, and knowing which of our claims survive on
    a machine whose build options differ from the one they were checked on.

    Two failure classes live here and nowhere else.

    (1) SETUP. A route that works, the route that looks right and does not, and
        what the user sees when discovery picks up the wrong thing. Every
        message below was produced by running the command, not read from a
        manual.

    (2) BUILD CONFIGURATION. A claim can be true on one build of the same
        version and false on another. deal.II is the extreme case: `Assert` is
        compiled out in Release, so an entire family of pitfalls whose signal is
        an assertion message cannot fire on a Release build while remaining
        correct on the Debug build most users have. A claim like that is not
        wrong — it is SCOPED, and the scope has to travel with it.

HOW ENTRIES ARE WRITTEN
    Same `[Category]` convention as the rest of the pitfall DB, with a second
    tag naming the sub-kind, as in `src/backends/_cross.py`:

        [Integration][Install]      obtaining a working install
        [Integration][Discovery]    how OASiS locates it; env vars
        [Integration][FirstRun]     what breaks on the first real run
        [Integration][BuildConfig]  claims conditional on build options
        [Integration][Portability]  version ranges; clean-environment evidence

    A `Signal:` clause must quote a string the software actually emits. Where a
    signal was produced by deliberately breaking something, the entry says how
    it was broken so a reader can reproduce it.

NO MACHINE PATHS
    Install knowledge has to talk about locations, but one machine's layout is
    wrong everywhere else. Entries therefore give the COMMAND that finds a path
    (`readlink -f "$(command -v febio4)"`) rather than the path itself, and use
    `<angle-bracket>` placeholders for anything host-specific.

VERIFICATION SCOPE
    Checked on one Linux host: glibc 2.31, CPython 3.12 for the solver venv.
    Where a second configuration could not be built, the entry says which
    configuration it was checked on instead of stating the claim flat.
"""
from __future__ import annotations


# ── shared probes ────────────────────────────────────────────────────────
#
# Commands a user (or an agent) can run to answer "what have I actually got?".
# Kept together so the per-backend entries can reference them by name instead
# of repeating a shell line four times.

CONFIG_PROBES: dict[str, dict[str, str]] = {
    "dealii_build_type": {
        "what": "Release or Debug — decides whether Assert-based pitfalls fire",
        "command": (
            "grep -m1 CMAKE_BUILD_TYPE <dealii-build-dir>/detailed.log  "
            "# source build\n"
            "# or, for any install, test for the debug library:\n"
            "ls <dealii-prefix>/lib/libdeal_II.g.so 2>/dev/null "
            "&& echo DEBUG_AVAILABLE || echo RELEASE_ONLY"
        ),
        "reading": (
            "libdeal_II.g.so present => a Debug library exists and Assert can "
            "fire. Only libdeal_II.so => Release-only; every Assert is a no-op."
        ),
    },
    "dealii_features": {
        "what": "which optional dependencies this deal.II was built against",
        "command": (
            "grep -E '^#define DEAL_II_WITH|^/\\* #undef DEAL_II_WITH' "
            "$(find <dealii-prefix> -path '*/deal.II/base/config.h' | head -1)"
        ),
        "reading": (
            "`#define DEAL_II_WITH_X` = ON. `/* #undef DEAL_II_WITH_X */` = "
            "OFF. This file is the ONLY reliable source — see the "
            "header-probe pitfall below."
        ),
    },
    "dolfinx_scalar": {
        "what": "real or complex PETSc scalars",
        "command": (
            "<fenics-python> -c \"import dolfinx, numpy as np; "
            "print(np.dtype(dolfinx.default_scalar_type).name)\""
        ),
        "reading": (
            "float64 = real build. complex128 = complex build. The two are "
            "SEPARATE installs; a single environment has one of them."
        ),
    },
    "febio_build": {
        "what": "which linear solvers this FEBio was compiled with",
        "command": "<febio-binary> -info 2>&1 | head -30",
        "reading": (
            "The line `Default linear solver: skyline` means no vendor solver "
            "was linked. A build with MKL reports a pardiso-family default. "
            "-info does NOT print a feature list — do not grep it for MMG."
        ),
    },
    "sparta_packages": {
        "what": "which optional SPARTA packages are compiled in",
        "command": "cd <sparta-src> && make ps",
        "reading": (
            "Prints `Installed YES/NO: package <NAME>` per package. A package "
            "listed NO means its commands and its `kk`-suffixed styles do not "
            "exist in the binary."
        ),
    },
    "kratos_glibc": {
        "what": "whether a Kratos wheel can actually load on this host",
        "command": (
            "ldd --version | head -1   # host glibc\n"
            "objdump -T <site-packages>/KratosMultiphysics/.libs/"
            "Kratos.cpython-*.so | grep -o 'GLIBC_2\\.[0-9]*' | sort -uV | tail -1"
        ),
        "reading": (
            "If the highest GLIBC_ symbol the wheel requires exceeds the host "
            "glibc, the import fails at runtime even though pip installed it "
            "without complaint."
        ),
    },
}


# ── deal.II ──────────────────────────────────────────────────────────────

_DEALII = {
    "verified_on": (
        "deal.II 9.8.0-pre, source build, CMAKE_BUILD_TYPE=Release, "
        "Release-only (no debug library). Features ON: ARPACK ASSIMP GMSH GSL "
        "KOKKOS LAPACK METIS MUPARSER OPENCASCADE TASKFLOW TBB UMFPACK ZLIB "
        "THREADS. Features OFF: MPI P4EST PETSC SLEPC TRILINOS COMPLEX_VALUES "
        "HDF5 SUNDIALS SYMENGINE CGAL ADOLC MUMPS SCALAPACK VTK 64BIT_INDICES."
    ),
    "install_route": (
        "Two routes work. Pick by whether you need a Debug build.\n"
        "  (a) Distribution package — fastest, but check what version it is:\n"
        "        sudo apt install -y libdeal.ii-dev\n"
        "      Ubuntu 20.04 ships 9.1.1, which is old enough that many current "
        "APIs are absent. Check before relying on it:\n"
        "        grep DEAL_II_PACKAGE_VERSION /usr/include/deal.II/base/config.h\n"
        "  (b) Source build — needed for a recent version or a Debug library:\n"
        "        git clone https://github.com/dealii/dealii.git\n"
        "        cmake -S dealii -B dealii/build "
        "-DCMAKE_BUILD_TYPE=DebugRelease\n"
        "        cmake --build dealii/build -j$(nproc)\n"
        "      DebugRelease builds BOTH libdeal_II.so and libdeal_II.g.so, "
        "which is what you want if you care about assertion messages."
    ),
    "pitfalls": [
        "[Integration][Discovery] The deal.II path OASiS discovers is the "
        "install ROOT, but CMake needs the directory that actually holds "
        "deal.IIConfig.cmake — and for a SOURCE build that is "
        "<root>/build/lib/cmake/deal.II, not <root>/lib/cmake/deal.II. Point "
        "find_package at the bare source root and CMake does not fail: it "
        "walks past it and picks up whatever system deal.II exists, silently. "
        "Signal: the configure step prints `-- Using the deal.II-9.1.1 "
        "installation found at /usr` while you believed you were building "
        "against your own newer tree; the first compile error is then a "
        "missing post-9.1 header or symbol rather than anything about paths. "
        "Reproduce: point DEAL_II_DIR at a source tree, at an empty directory, "
        "or unset it entirely — all three produce that same line on a host "
        "with a distribution deal.II present. Defense: pass the config "
        "directory explicitly, and read back the version CMake reports: "
        "`cmake -DDEAL_II_DIR=<root>/build/lib/cmake/deal.II ...` then confirm "
        "the `Using the deal.II-<version>` line names the version you meant.",

        "[Integration][Discovery] DEAL_II_DIR and DEALII_ROOT are OPTIONAL "
        "overrides, not requirements — with neither set, OASiS searches conda "
        "envs, then ~/dealii and similar source dirs, then system paths. But "
        "the override is accepted on an is-a-directory test ALONE: it is not "
        "checked for deal.II content. Signal: with DEAL_II_DIR set to any "
        "existing directory, `discover(query='list')` reports deal.II as "
        "`available` and names that directory back to you; the run fails much "
        "later, at compile time, with an error that says nothing about "
        "DEAL_II_DIR. Defense: after setting either variable, confirm "
        "`<dir>/include/deal.II/base/config.h` or "
        "`<dir>/lib/cmake/deal.II/deal.IIConfig.cmake` exists.",

        "[Integration][BuildConfig] BUILD TYPE INVERTS A WHOLE FAMILY OF "
        "PITFALLS — check it before trusting any assertion-based claim. "
        "deal.II compiles `Assert(...)` out entirely in Release; only "
        "`AssertThrow(...)` survives. So every pitfall whose signal is an "
        "ExcMessage / ExcIndexRange / ExcDimensionMismatch / ExcDivideByZero "
        "text is: TRUE on a Debug build, and UNOBSERVABLE on a Release build "
        "— where the same mistake produces a wrong number instead of a "
        "message. Signal, Debug: `An error occurred in line <N> of file <...>` "
        "followed by `The violated condition was:` and the exception name. "
        "Signal, Release: no message at all. Demonstrated: reading index 7 of "
        "a `Vector<double>` of size 3 returns 0.0 and the program exits 0 on "
        "the Release build; the Debug build aborts on ExcIndexRange. Check "
        "which you have with CONFIG_PROBES['dealii_build_type'] BEFORE acting "
        "on an assertion-based pitfall.",

        "[Integration][BuildConfig] A Release deal.II cannot report a CG "
        "breakdown, because the breakdown guard is an `Assert`. In "
        "`lac/solver_cg.h` the divide-by-zero checks are "
        "`Assert(std::abs(...) != 0., ExcDivideByZero())` — compiled out in "
        "Release. What a Release build emits instead is the AssertThrow at the "
        "end of the same solve(): "
        "`AssertThrow(solver_state == SolverControl::success, "
        "SolverControl::NoConvergence(it, worker.residual_norm))`. Signal, "
        "Release, singular matrix: `The violated condition was: solver_state "
        "== SolverControl::success` with `Additional information: Iterative "
        "method reported convergence failure in step 1. The residual in the "
        "last step was nan.` — note `nan`. Signal, Release, merely too few "
        "iterations: the same violated-condition text but a FINITE residual "
        "value. That residual is the whole diagnostic on a Release build: nan "
        "means singular or non-SPD, finite means raise the iteration cap. On a "
        "Debug build you would instead get ExcDivideByZero naming the "
        "breakdown directly. Check which build you have with "
        "CONFIG_PROBES['dealii_build_type'] before deciding which of those two "
        "diagnostics you are entitled to expect.",

        "[Integration][BuildConfig] DO NOT detect deal.II features by testing "
        "whether a header exists — on a source build it always does. The "
        "source tree ships `lac/petsc_vector.h`, `lac/trilinos_vector.h`, "
        "`lac/slepc_solver.h`, `base/mpi.h` and `distributed/tria.h` "
        "regardless of which optional dependencies were enabled; each is "
        "guarded internally by `#ifdef DEAL_II_WITH_...`, so it compiles to "
        "nothing when the feature is off. Signal: all five headers present in "
        "the source tree while config.h shows `/* #undef DEAL_II_WITH_PETSC */`, "
        "`/* #undef DEAL_II_WITH_TRILINOS */`, `/* #undef DEAL_II_WITH_MPI */`, "
        "`/* #undef DEAL_II_WITH_P4EST */`, `/* #undef DEAL_II_WITH_SLEPC */`. "
        "Only config.h is authoritative — and on a source build config.h is a "
        "GENERATED file that lives under the BUILD tree, not the source tree. "
        "Use CONFIG_PROBES['dealii_features'].",

        "[Integration][BuildConfig] Any deal.II claim that names PETSc, "
        "Trilinos, MPI, p4est, SLEPc, HDF5, SUNDIALS, SymEngine, CGAL or "
        "complex values is conditional on that feature being compiled in, and "
        "a large fraction of installs do not have it. It was not possible to "
        "observe those code paths on the build checked here, because all of "
        "the above are OFF in its config.h; claims about them are therefore "
        "recorded as UNVERIFIED-ON-THIS-BUILD rather than confirmed. Signal of "
        "the absence, at compile time rather than run time: a program using "
        "`PETScWrappers::` or `TrilinosWrappers::` fails to link, and "
        "`Utilities::MPI::` calls degrade to the serial stubs. Check with "
        "CONFIG_PROBES['dealii_features'] before selecting a parallel or "
        "external-solver template.",

        "[Integration][BuildConfig] UMFPACK availability is per-build, and "
        "`SparseDirectUMFPACK` is the default direct solver in many templates. "
        "It is ON in the build checked here (`#define DEAL_II_WITH_UMFPACK`), "
        "but conda-forge and some distribution packages have shipped without "
        "it. Signal when absent: deal.II raises its needs-UMFPACK exception "
        "from the SparseDirectUMFPACK constructor rather than failing to "
        "compile, so the failure appears at run time on an otherwise clean "
        "build. Check with CONFIG_PROBES['dealii_features'] and fall back to "
        "SolverCG with PreconditionSSOR if the feature is off.",
    ],
}


# ── FEniCSx (dolfinx) ────────────────────────────────────────────────────

_FENICS = {
    "verified_on": (
        "dolfinx 0.10.0 in two separate conda environments on the same host: "
        "one real-scalar (PETSc ScalarType float64, petsc4py 3.24.4, PETSc "
        "3.24.5) and one complex-scalar (complex128). slepc4py 3.24.3, gmsh "
        "4.15.1 and pyvista 0.47.1 present; adios2 absent."
    ),
    "install_route": (
        "conda-forge, in a DEDICATED environment. dolfinx is not pip "
        "installable in any practical form.\n"
        "  REAL scalars (what almost every template assumes):\n"
        "    conda create -n ofa-fenicsx -y -c conda-forge "
        "fenics-dolfinx pyvista python=3.12\n"
        "  COMPLEX scalars (Helmholtz, time-harmonic Maxwell) — a SEPARATE "
        "environment, not a flag:\n"
        "    conda create -n ofa-fenicsx-complex -y -c conda-forge "
        "'fenics-dolfinx=*=*complex*' python=3.12\n"
        "Do not install dolfinx into the environment OASiS itself runs in; it "
        "pulls its own MPI and PETSc and will fight whatever is already there."
    ),
    "pitfalls": [
        "[Integration][Discovery] FENICS_PYTHON is OPTIONAL. Unset, OASiS "
        "searches conda envs whose NAME contains `fenics` or `dolfinx` "
        "(case-insensitive) and uses the first whose `import dolfinx` "
        "succeeds. Two behaviours follow, and they differ. Set it to a Python "
        "that lacks dolfinx and OASiS is honest — Signal: "
        "`dolfinx import failed at <path>:` followed by the real ImportError "
        "traceback, and the backend reports `not_installed`. Set it to a path "
        "that does not exist and OASiS silently ignores it and falls back to "
        "the discovered conda env — Signal: the backend reports `available` "
        "naming an interpreter you did not choose. A typo in FENICS_PYTHON is "
        "therefore invisible. Defense: confirm the reported interpreter path "
        "is the one you meant before running anything.",

        "[Integration][BuildConfig] REAL vs COMPLEX is a property of the "
        "INSTALL, not of the script, and a name-based env search cannot tell "
        "them apart. Both builds are `dolfinx 0.10.0` and both satisfy `import "
        "dolfinx`; if the complex env sorts first, OASiS will pick it for a "
        "real-valued problem. Signal, running a real-valued template on a "
        "complex build: assembly succeeds and the solution array has dtype "
        "complex128, so every downstream comparison against a float reference "
        "either warns about discarding the imaginary part or silently keeps a "
        "zero imaginary component. Signal, the reverse — a Helmholtz template "
        "on a real build: the form fails where it multiplies by 1j. Verified "
        "on this host by querying both environments: "
        "`np.dtype(dolfinx.default_scalar_type).name` is `float64` in one and "
        "`complex128` in the other. Defense: run "
        "CONFIG_PROBES['dolfinx_scalar'] against the interpreter OASiS "
        "reports, and choose the environment to match the physics.",

        "[Integration][FirstRun] dolfinx compiles every form at first use and "
        "caches the result under `$XDG_CACHE_HOME/fenics` (i.e. "
        "`~/.cache/fenics` unless XDG_CACHE_HOME is set). The first run of a "
        "script pays a C compile; later runs with the SAME form do not. "
        "Measured on this host as a whole-script wall-clock difference of "
        "seconds versus effectively zero for a small Poisson problem. Two "
        "consequences. (1) A first run that looks hung is usually compiling — "
        "the cache directory fills with `libffcx_forms_<hash>.c`, `.o` and "
        "`.so` files while it works. (2) The cache is keyed by a hash of the "
        "generated code, so a stale entry from a DIFFERENT dolfinx version can "
        "be present but unusable after an upgrade. Signal of a stale cache: an "
        "ImportError or undefined-symbol error naming a file under the fenics "
        "cache directory rather than anything in your script. Defense: "
        "`rm -rf ~/.cache/fenics` after upgrading dolfinx; it only costs one "
        "recompile.",

        "[Integration][Portability] dolfinx 0.10.0 made "
        "`petsc_options_prefix` a REQUIRED keyword-only argument of "
        "`dolfinx.fem.petsc.LinearProblem`. Code written against 0.9.x "
        "constructs LinearProblem without it and stops at the constructor. "
        "Signal: `TypeError: LinearProblem.__init__() missing 1 required "
        "keyword-only argument: 'petsc_options_prefix'`. Version range: the "
        "argument does not exist in 0.9.x and is mandatory in 0.10.0, so a "
        "single call cannot satisfy both — branch on "
        "`dolfinx.__version__` if you must support the pair.",

        "[Integration][Portability] Optional companions are per-environment "
        "and their absence is not visible until the call that needs them. On "
        "the environment checked here slepc4py, gmsh and pyvista are all "
        "present, so any claim that SLEPc must be installed separately is "
        "false FOR THIS INSTALL — the conda-forge fenics-dolfinx package "
        "pulled it in. adios2 is absent, and `from dolfinx.io import "
        "VTXWriter` still imports successfully, so an import check does NOT "
        "prove VTX output will work. Defense: probe the specific module you "
        "need in the specific interpreter OASiS reports, rather than assuming "
        "either presence or absence: "
        "`<fenics-python> -c 'import slepc4py, adios2'`.",
    ],
}


# ── 4C ───────────────────────────────────────────────────────────────────

_FOURC = {
    "verified_on": (
        "4C 2026.2.0-dev, source build against a prebuilt dependency bundle, "
        "Trilinos and Open MPI, single-rank runs."
    ),
    "install_route": (
        "Source build only — there is no binary distribution. The dependency "
        "set (Trilinos, Kokkos, qhull, MueLu, NOX, HDF5, Open MPI) is the "
        "hard part; build it once into a prefix and reuse it.\n"
        "    git clone https://github.com/4C-multiphysics/4C.git\n"
        "    cmake -S 4C -B 4C/build --preset=<preset>\n"
        "    cmake --build 4C/build -j$(nproc)\n"
        "The result is <4C-root>/build/4C plus lib4C.so beside it."
    ),
    "pitfalls": [
        "[Integration][Discovery] FOURC_BINARY is accepted WITHOUT being "
        "checked — any existing file is taken as 4C. Point it at the wrong "
        "executable and OASiS reports `available` naming that file, then hands "
        "it your input deck. Signal: `discover(query='list')` shows 4C "
        "available at a path that is not a 4C build; the run afterwards "
        "produces no 4C banner and no result files, and whether it 'fails' "
        "depends entirely on what that other program does with the arguments. "
        "Reproduce by setting FOURC_BINARY to any executable on PATH. Defense: "
        "a real 4C binary answers `--help` with the banner line `4C - "
        "Multiphysics` and a `Comprehensive Computational Community Code` "
        "subtitle; anything else is the wrong file. Note 4C has NO version "
        "flag — `4C --version` and `4C -v` both answer `The following argument "
        "was not expected: --version` / `Run with --help for more "
        "information.`; the version is printed in the banner of an actual run.",

        "[Integration][Discovery] Neither FOURC_ROOT nor FOURC_BINARY is "
        "required. With both unset OASiS falls back to a list of conventional "
        "locations and will find a build under a home-directory 4C checkout. "
        "That means a WRONG FOURC_ROOT is not an error either — it is ignored "
        "in favour of the fallback. Signal: with FOURC_ROOT pointed at an "
        "unrelated directory the backend still reports "
        "`4C at <conventional-path>`, i.e. the variable you set had no effect "
        "and the message does not say so. Defense: read the path OASiS reports "
        "back rather than assuming your variable chose it. FOURC_ROOT has a "
        "second, separate job — it is where reference input decks are looked "
        "up, under <FOURC_ROOT>/tests/input_files — so setting it wrong costs "
        "you the examples surface even when the binary resolves.",

        "[Integration][FirstRun] LD_LIBRARY_PATH is REQUIRED ONLY IF the build "
        "did not embed a RUNPATH, and the commonly-quoted value is incomplete "
        "on its own. 4C links against both its own lib4C.so, which sits in the "
        "BUILD directory, and ~50 Trilinos/Kokkos libraries in the dependency "
        "prefix. Check which case you are in: "
        "`readelf -d <4C-binary> | grep RUNPATH`. If a RUNPATH covering both "
        "directories is present, no environment variable is needed at all — "
        "verified here by running the binary with LD_LIBRARY_PATH unset. If it "
        "is absent you need BOTH directories. Signal with neither: "
        "`<4C-binary>: error while loading shared libraries: lib4C.so: cannot "
        "open shared object file: No such file or directory`. Signal with only "
        "the dependency prefix: the same sentence for `lib4C.so`. Signal with "
        "only the build directory: the same sentence for "
        "`libteuchoscomm.so.16` — a Trilinos library, which is the clue that "
        "the dependency prefix is the missing half. Defense: "
        "`export LD_LIBRARY_PATH=<4C-build-dir>:<deps-prefix>/lib` — build "
        "directory FIRST.",

        "[Integration][FirstRun] The output prefix is a MANDATORY positional "
        "argument, and omitting it crashes rather than printing usage. 4C "
        "takes `4C <input> <output-prefix>`; called with only the input file "
        "it throws from inside command-line parsing and the process dumps "
        "core. Signal: `what():  PROC 0 ERROR in "
        "<...>/4C_global_full_main.cpp, line 457:` immediately followed by "
        "`Please provide both <input> and <output> arguments.`, then a stack "
        "trace through `parse_command_line` and an aborted process. The useful "
        "sentence is the second line; everything after it is noise.",

        "[Integration][FirstRun] When 4C dies while reading its input, the "
        "diagnostic can be lost entirely — MPI_ABORT tears the process down "
        "before the message is flushed. Signal, for BOTH a nonexistent input "
        "file and an existing but empty one: the only output is `MPI_ABORT was "
        "invoked on rank 0 in communicator MPI_COMM_WORLD` / `with errorcode "
        "1.` and the Open MPI explanation that follows it, with nothing naming "
        "the file or the problem. Do not read that as a 4C bug report — treat "
        "errorcode 1 with no 4C banner as 'the input was never successfully "
        "opened or parsed'. Defense: confirm the input path exists and is "
        "non-empty before blaming the deck, and check that a normal run "
        "reaches the banner (`4C` / `version <...>` / `git SHA1`) — if the "
        "banner never appears, the failure is earlier than the physics.",
    ],
}


# ── Kratos ───────────────────────────────────────────────────────────────

_KRATOS = {
    "verified_on": (
        "Kratos 10.4.2 and 10.3.0 wheels for CPython 3.12, on a host with "
        "glibc 2.31. Both were installed into clean virtual environments and "
        "imported."
    ),
    "install_route": (
        "pip, but the version matters more than the command:\n"
        "    python -m pip install KratosMultiphysics-all\n"
        "That metapackage resolves the application set together and, at the "
        "time of checking, lands on the 10.3.x line. Installing "
        "`KratosMultiphysics` alone lets pip take the newest wheel, which is "
        "where the glibc problem below starts. Verify immediately after "
        "installing:\n"
        "    python -c 'import KratosMultiphysics as KM; "
        "print(KM.KratosGlobals.Kernel.Version())'"
    ),
    "pitfalls": [
        "[Integration][Install] THE 10.4.x WHEELS ARE MIS-TAGGED AND WILL "
        "INSTALL ON A HOST THEY CANNOT RUN ON. "
        "`kratosmultiphysics-10.4.2-1-cp312-cp312-manylinux_2_28_x86_64.whl` "
        "declares manylinux_2_28, so pip accepts it on any host with glibc >= "
        "2.28 — but the shipped `Kratos.cpython-*.so` requires GLIBC_2.32 and "
        "GLIBC_2.34. On glibc 2.31 the install reports success and the import "
        "fails. Signal: `ImportError: /lib/x86_64-linux-gnu/libc.so.6: version "
        "`GLIBC_2.32' not found (required by "
        "<site-packages>/KratosMultiphysics/.libs/Kratos.cpython-312-x86_64-"
        "linux-gnu.so)`. CRITICAL — the very next line Kratos prints is "
        "MISLEADING: `Unable to find KratosCore. Please make sure that your "
        "LD_LIBRARY_PATH or DYLD_LIBRARY_PATH environment variable includes "
        "the path to the Kratos libraries.` No value of LD_LIBRARY_PATH can "
        "fix this; the library is found, it just cannot load against this "
        "glibc. Do not spend time on that suggestion. Defense: check the host "
        "glibc with `ldd --version | head -1` and downgrade — the 10.3.0 wheel "
        "is `manylinux2014_x86_64.manylinux_2_17_x86_64` and needs no symbol "
        "above GLIBC_2.16, so it loads on effectively any current Linux: "
        "`python -m pip install 'KratosMultiphysics==10.3.0'`. Use "
        "CONFIG_PROBES['kratos_glibc'] to check a wheel before trusting it.",

        "[Integration][Discovery] Kratos is used through whichever Python is "
        "running OASiS — there is no KRATOS_PYTHON-style override in the "
        "backend's availability check, so 'installed' means 'installed in THAT "
        "interpreter'. A working Kratos in a different interpreter on the same "
        "machine does not help. Signal: `discover(query='list')` reports "
        "kratos `not_installed` with a fragment of the ImportError, while "
        "running the identical import by hand under another interpreter "
        "succeeds — the two are simply different environments. Defense: "
        "install Kratos into the environment OASiS itself runs in, and confirm "
        "with the same interpreter: "
        "`<oasis-python> -c 'import KratosMultiphysics'`.",

        "[Integration][Install] Not every Kratos application has a wheel. The "
        "PFEM applications in particular are absent from PyPI, so an install "
        "plan that names them cannot succeed by pip at all. Signal: "
        "`ERROR: No matching distribution found for "
        "KratosPfemFluidDynamicsApplication` — verified by attempting the "
        "download. Defense: treat the pip route as covering the applications "
        "bundled by `KratosMultiphysics-all` (structural mechanics, fluid "
        "dynamics, convection-diffusion, constitutive laws, contact, DEM, MPM, "
        "geo/poro mechanics, mesh moving, meshing, mapping, FSI, "
        "co-simulation, shallow water, dam, linear solvers); anything outside "
        "that set needs a source build.",
    ],
}


# ── DUNE-fem ─────────────────────────────────────────────────────────────

_DUNE = {
    "verified_on": (
        "dune-fem 2.12.0.2 and its dune-common / geometry / grid / istl / "
        "localfunctions / alugrid siblings, all from PyPI, CPython 3.12. "
        "Checked both inside a conda environment and in a plain venv."
    ),
    "install_route": (
        "pip, from PyPI, into an environment that also has a C++ toolchain, "
        "CMake and mpi4py — DUNE compiles C++ on demand at first use.\n"
        "    python -m pip install dune-fem mpi4py\n"
        "A conda environment is a convenient way to get a matching toolchain, "
        "but the DUNE packages themselves still come from PyPI inside it."
    ),
    "pitfalls": [
        "[Integration][Install] THE CONDA-FORGE ROUTE DOES NOT EXIST. "
        "`conda create -n <env> -c conda-forge dune-fem` cannot succeed, "
        "because conda-forge has no dune-fem package. Signal: "
        "`No match found for: dune-fem. Search: *dune-fem*` from "
        "`conda search -c conda-forge --override-channels dune-fem`; an "
        "install attempt ends in PackagesNotFoundError. PyPI is the working "
        "source — `python -m pip install dune-fem` installs 2.12.0.2 and its "
        "siblings. If some OASiS text still recommends the conda-forge "
        "channel as the supported path and warns against PyPI, that text is "
        "backwards; follow this entry.",

        "[Integration][Install] mpi4py is an UNDECLARED dependency — pip will "
        "not pull it, and DUNE refuses to import without it. In a clean "
        "virtual environment `pip install dune-fem` completes successfully and "
        "the very first `import dune.fem` stops with an instruction rather "
        "than a traceback. Signal, verbatim and easy to miss because it does "
        "not look like an error: `Please run` / `    pip install mpi4py` / "
        "`before rerunning your Dune script.` Defense: install the pair "
        "together — `python -m pip install dune-fem mpi4py`. mpi4py needs an "
        "MPI implementation with a working `mpicc` to build against.",

        "[Integration][FirstRun] DUNE JIT-COMPILES AGAINST WHATEVER PYTHON ITS "
        "CMAKE FINDS, WHICH IS NOT NECESSARILY THE ONE RUNNING IT. The "
        "generated modules are built by a CMake sub-build whose FindPython3 "
        "can pick up a different interpreter that happens to be earlier on "
        "PATH — a conda base environment is the usual culprit. Compilation "
        "then SUCCEEDS and the failure appears at import, which makes it look "
        "like a DUNE bug rather than an environment one. Signal: `ImportError: "
        "<env>/.cache/dune-py/python/dune/generated/<module>.so: undefined "
        "symbol: PyThreadState_GetUnchecked` — that symbol is CPython 3.13+, "
        "so seeing it while running 3.12 means the build used 3.13 headers. "
        "Any `undefined symbol: Py*` from a file under .cache/dune-py is this "
        "same problem. Reproduced in a clean venv on this host with a conda "
        "3.13 first on PATH. Defense: make the intended interpreter "
        "unambiguous to CMake before the first import — set "
        "`Python3_ROOT_DIR` to the environment prefix, set VIRTUAL_ENV (venv) "
        "or CONDA_PREFIX (conda) to that same prefix and clear the other one, "
        "and put its bin directory first on PATH. Then delete the poisoned "
        "cache, because it will otherwise be reused.",

        "[Integration][FirstRun] The DUNE JIT cache is NOT under ~/.cache — it "
        "lives inside the environment prefix, at `<env-prefix>/.cache/dune-py`. "
        "Clearing ~/.cache does nothing for it, which is why a 'stale cache' "
        "is so often not actually cleared. Find it for certain with "
        "`<python> -c 'import dune.packagemetadata as p; "
        "print(p.getDunePyDir())'`. Note the older spelling is deprecated and "
        "says so — Signal: `Call to deprecated function/property "
        "`dune.common.module.get_dune_py_dir`. Use "
        "'dune.packagemetadata.getDunePyDir' instead` — so code using the old "
        "name still works but should be updated. Defense after any interpreter "
        "or DUNE version change: `rm -rf \"$(<python> -c 'import "
        "dune.packagemetadata as p; print(p.getDunePyDir())')\"`.",

        "[Integration][FirstRun] Budget real time for the first solve. Every "
        "grid/space/scheme combination is a separate C++ translation unit "
        "compiled on demand, so a first run of even a small Poisson problem "
        "takes tens of seconds of pure compilation on a warm-toolchain, "
        "cold-cache environment, and a cold toolchain is slower still. A run "
        "that appears hung early on is almost always compiling. Signal that it "
        "is progressing rather than stuck: new files appearing under the "
        "dune-py cache directory located above.",
    ],
}


# ── FEBio ────────────────────────────────────────────────────────────────

_FEBIO = {
    "verified_on": (
        "FEBio 4.12, built from source with USE_MKL=OFF (MKLROOT not found), "
        "USE_HYPRE=OFF, USE_LEVMAR=OFF, USE_MMG=OFF, CMAKE_BUILD_TYPE=Release. "
        "Links libgomp; no MKL library is linked."
    ),
    "install_route": (
        "Two routes, both fine; source is the one that needs no account.\n"
        "  (a) Official binary from https://febio.org/downloads/ — requires a "
        "free registered account, so there is no direct download URL and the "
        "step is interactive. Unpack, then point OASiS at it (below).\n"
        "  (b) Source build — no account, and you control the options:\n"
        "        git clone https://github.com/febiosoftware/FEBio.git\n"
        "        cmake -S FEBio -B FEBio/cbuild -DUSE_MKL=OFF \\\n"
        "              -DCMAKE_EXE_LINKER_FLAGS='-fopenmp -ldl' \\\n"
        "              -DCMAKE_SHARED_LINKER_FLAGS='-fopenmp -ldl'\n"
        "        cmake --build FEBio/cbuild -j$(nproc)\n"
        "      USE_MKL=OFF is what makes this build without an Intel "
        "toolchain; it also changes which linear solvers exist (below)."
    ),
    "pitfalls": [
        "[Integration][Discovery] FEBIO_BINARY is OPTIONAL but is the only "
        "reliable way to choose a build, and it is accepted without "
        "validation. Set it to any existing file and OASiS reports "
        "`available` naming that file, with no check that it is FEBio at all. "
        "Signal of the unvalidated case: `discover(query='list')` reports FEBio "
        "available at a path that is not an FEBio build, and the failure "
        "arrives later as missing output rather than as a complaint about the "
        "variable. Unset, discovery searches a fixed list of CONVENTIONAL "
        "LOCATIONS FIRST — a `FEBio/bin/febio4` or `FEBioStudio/bin/febio4` "
        "under your home directory, then /opt and /usr/local — and only falls "
        "back to PATH afterwards, where it takes `febio4`, then `febio3`, then "
        "`febio`. So putting a new build first on PATH does NOT make it win: "
        "verified by placing a febio4 at the front of PATH and watching "
        "discovery return the home-directory one anyway. That home-directory "
        "entry is very often a SYMLINK to a build tree, which is how people "
        "end up running a binary they deleted and rebuilt somewhere else. "
        "Defense: resolve the path before believing it — "
        "`readlink -f \"$(command -v febio4)\"` — and prove the file answers "
        "as FEBio: `<candidate> -info` on a real FEBio prints `FEBio version "
        " = <version>` and `compiled on <date>`, while an unknown flag is "
        "rejected with `FATAL ERROR: Invalid command line option '<flag>'.` "
        "(`-v` is NOT a valid FEBio flag; use `-info`.)",

        "[Integration][BuildConfig] MKL PRESENCE CHANGES WHICH LINEAR SOLVERS "
        "EXIST, and the way to detect it is the DEFAULT SOLVER LINE, not a "
        "feature list. `febio4 -info` prints a banner, `compiled on <date>`, "
        "`FEBio version  = <version>`, `Starting without configuration file` "
        "and `Default linear solver: skyline` — and nothing else. It does NOT "
        "list build features, so grepping its output for MKL, MMG, HYPRE or "
        "LEVMAR returns nothing whether or not they are compiled in, and any "
        "instruction to detect a feature that way is unusable. Signal of a "
        "no-MKL build: `Default linear solver: skyline`, confirmed here "
        "against a CMakeCache with `USE_MKL:BOOL=OFF` and "
        "`MKLROOT:FILEPATH=MKLROOT-NOTFOUND`, and by `ldd <binary>` showing "
        "libgomp but no MKL library. Consequence to scope: any claim that "
        "recommends or benchmarks the pardiso solver applies only to an "
        "MKL-enabled build. It was not possible to check the MKL-enabled "
        "behaviour here — no such build exists on this host — so pardiso "
        "claims are recorded as unverified rather than confirmed. Defense: run "
        "CONFIG_PROBES['febio_build'] and, for certainty about what was asked "
        "for at configure time, `grep -E '^USE_|^MKLROOT' "
        "<febio-build-dir>/CMakeCache.txt`.",

        "[Integration][FirstRun] FEBio's own errors are clearly worded and "
        "worth reading literally — they are not stack traces. A missing or "
        "unreadable input file stops immediately. Signal: a boxed `ERROR` "
        "banner containing `FATAL ERROR: Failed opening input file <path>`. If "
        "you see that, the problem is the path or permissions, not the model. "
        "Note also that FEBio with no arguments does not exit — it drops into "
        "its own interactive `febio>` prompt after printing the banner, which "
        "in an automated context looks like a hang. Always pass `-i <file>`.",
    ],
}


# ── SPARTA ───────────────────────────────────────────────────────────────

_SPARTA = {
    "verified_on": (
        "SPARTA dated 24 Sep 2025, serial build (`spa_serial`), running on 1 "
        "MPI task. Optional packages FFT and PYTHON both report Installed NO; "
        "KOKKOS is not installed."
    ),
    "install_route": (
        "Source build; SPARTA has no wheel or package.\n"
        "    git clone https://github.com/sparta-sparta/sparta.git\n"
        "    cd sparta/src && make serial     # or: make mpi\n"
        "The binary appears in that same src directory as `spa_serial` (or "
        "`spa_mpi`). It is not installed onto PATH by the build, which is why "
        "SPARTA_BINARY exists."
    ),
    "pitfalls": [
        "[Integration][Discovery] SPARTA_BINARY is OPTIONAL and unvalidated. "
        "Unset, OASiS looks for `spa_serial`, `spa_mpi` or `sparta` on PATH "
        "and then in conventional build locations. Set to any existing file, "
        "it is taken at face value. Signal: `discover(query='list')` reports "
        "SPARTA available at that path AND still advertises its full command "
        "knowledge, because the knowledge base is a data file that loads "
        "whether or not the binary is real — so the reported command count "
        "tells you nothing about the binary. Defense: a genuine SPARTA prints "
        "its dated version as the first line of any run, e.g. "
        "`SPARTA (24 Sep 2025)`, followed by `Running on N MPI task(s)`. "
        "Confirm that before trusting the path.",

        "[Integration][BuildConfig] KOKKOS IS A COMPILE-TIME PACKAGE AND IS "
        "ABSENT FROM A DEFAULT BUILD — every accelerated style depends on it. "
        "Three distinct messages, all captured from real runs on a build "
        "without it. Signal, command-line switch: `ERROR: Cannot use -kokkos "
        "on without KOKKOS installed (../sparta.cpp:381)`. Signal, deck "
        "command: `ERROR: Package kokkos command without KOKKOS package "
        "enabled (../input.cpp:1507)`. Signal, style suffix: SPARTA answers "
        "`ERROR: Unknown command: suffix kk (../input.cpp:244)` — note that "
        "`suffix` is not a SPARTA command at all, so advice to 'use suffix kk' "
        "is wrong regardless of the build; the switch is `-kokkos on` together "
        "with `-sf kk`. Beware a misleading check: Kokkos-named symbols are "
        "present in the binary even when the package is not installed, so "
        "inspecting symbols does not answer the question. Defense: run "
        "CONFIG_PROBES['sparta_packages'] — `make ps` in the source directory "
        "prints `Installed YES/NO: package <NAME>` — and rebuild with the "
        "package if you need it. Scope: only the REFUSAL messages were "
        "observed, on a build without the package. Nothing here claims what "
        "an accelerated run does or how much faster it is.",

        "[Integration][FirstRun] Bundled example decks reference data files "
        "(`species ar.species Ar`, `collide vss air air.vss`, `read_surf "
        "data.circle`) that live in the SPARTA distribution's data/ and "
        "examples/ directories, not in the deck and not in the knowledge base. "
        "A deck run from a scratch directory therefore dies on a missing file "
        "even though the deck itself is correct. Signal: `Cannot open species "
        "file <name>` and its equivalents for the vss and surf files. Defense: "
        "run from a directory that contains the referenced files, or point "
        "SPARTA_DATA_DIR (colon-separated, like PATH) at the distribution's "
        "data and examples directories. Where a task ships its own geometry "
        "with the same filename as a bundled example, put the task directory "
        "FIRST — the two are different geometries and the wrong one will be "
        "picked up silently.",
    ],
}


# ── NGSolve ──────────────────────────────────────────────────────────────

_NGSOLVE = {
    "verified_on": (
        "NGSolve 6.2.2604 (with netgen-mesher 6.2.2604, netgen-occt 7.8.1) "
        "from PyPI wheels on CPython 3.12, plus a clean-environment re-check "
        "at 6.2.2604 and 6.2.2606."
    ),
    "install_route": (
        "pip wheels; no compiler needed.\n"
        "    python -m pip install ngsolve\n"
        "This pulls netgen-mesher, netgen-occt and an OpenBLAS build. Install "
        "it into the SAME interpreter that runs OASiS — see the discovery "
        "pitfall."
    ),
    "pitfalls": [
        "[Integration][Discovery] NGSolve has NO environment-variable "
        "override. The backend uses the interpreter running OASiS and nothing "
        "else, so 'is NGSolve installed' is really 'is NGSolve installed in "
        "THIS interpreter'. An NGSolve in some other environment on the same "
        "machine is invisible. Signal when it is missing from the running "
        "interpreter: the backend reports `not_installed` with `ngsolve import "
        "failed:` and the ModuleNotFoundError; when present it reports "
        "`NGSolve <version> at <interpreter>` — read that interpreter path, it "
        "is the whole answer. Defense: `<oasis-python> -m pip install ngsolve`, "
        "using the interpreter OASiS names, not whichever `pip` is on PATH.",

        "[Integration][Portability] The pinned and newest releases behave the "
        "same on the checked surface, and the surface is not small. A clean "
        "virtual environment was built twice — once at the pinned NGSolve "
        "6.2.2604 and once at the newest available, 6.2.2606 — and the "
        "backend's full tier-2 fixture set was re-run in each. All fixtures "
        "behaved identically to the reference environment in both. So NGSolve "
        "API claims in this catalog are scoped to 6.2.2604 through 6.2.2606 "
        "rather than to a single build, and nothing in them depends on a "
        "package that only exists on the machine they were written on.",

        "[Integration][Portability] The reference environment and a fresh "
        "install do NOT get the same numeric stack, and that difference is not "
        "NGSolve's. Installing today into a clean environment resolves numpy "
        "2.x and a matching scipy, whereas an environment built earlier can "
        "still be on numpy 1.26.x. NGSolve itself was unaffected across that "
        "gap in the re-run described above, but any claim about array "
        "semantics — copy-versus-view, scalar conversion, dtype promotion — is "
        "a numpy claim and inherits the numpy major version, not the solver "
        "version. Defense: record `numpy.__version__` alongside the solver "
        "version when a pitfall is about arrays rather than about NGSolve.",
    ],
}


# ── scikit-fem ───────────────────────────────────────────────────────────

_SKFEM = {
    "verified_on": (
        "scikit-fem 12.0.1 on CPython 3.12, plus clean-environment re-checks "
        "at 12.0.1 and at the newest release, 12.0.2."
    ),
    "install_route": (
        "pip; pure Python, nothing to compile.\n"
        "    python -m pip install scikit-fem meshio\n"
        "meshio is separate and is what writes VTU output. Install into the "
        "interpreter that runs OASiS."
    ),
    "pitfalls": [
        "[Integration][Discovery] scikit-fem has NO environment-variable "
        "override — like NGSolve it is imported from the interpreter running "
        "OASiS. Signal when present: the backend reports `scikit-fem "
        "<version>` with no path, so if you need to know WHICH interpreter "
        "that was, read the path NGSolve or the server reports rather than "
        "guessing. Signal when absent: `skfem import failed:` and the "
        "ModuleNotFoundError. Defense: install with the OASiS interpreter "
        "explicitly — `<oasis-python> -m pip install scikit-fem meshio`.",

        "[Integration][Portability] Checked across two versions and a clean "
        "environment. A fresh virtual environment was built at the pinned "
        "12.0.1 and again at 12.0.2, and the backend's full tier-2 fixture set "
        "was re-run in each; both matched the reference environment "
        "throughout. scikit-fem claims in this catalog are therefore scoped to "
        "12.0.1 through 12.0.2. The caveat is the same as NGSolve's: the clean "
        "environments resolved numpy 2.x while the reference environment is on "
        "numpy 1.26.x, so a pitfall about array behaviour is pinned to a numpy "
        "version and should say so.",
    ],
}


# ── portability evidence ─────────────────────────────────────────────────
#
# "It works here" is not evidence that it works anywhere. The entries above
# make version-range claims; this records what was actually done to earn them,
# so a reader can judge the claims rather than trust them — and so the next
# person to touch this file knows which claims rest on a re-run and which rest
# on a single machine.
#
# Method: build a virtual environment with nothing in it, install ONLY the
# backend under test, and re-run that backend's full tier-2 fixture set. A
# fixture that passes on the development machine and fails on a clean one has
# found either a portability defect or an undeclared dependency. Both are
# things a stranger cloning OASiS hits and we do not.

PORTABILITY_EVIDENCE: dict[str, dict] = {
    "clean_env_pinned": {
        "what": "empty venv, CPython 3.12, pinned solver versions",
        "installed": "scikit-fem 12.0.1, NGSolve 6.2.2604, meshio",
        "result": (
            "Every scikit-fem and NGSolve tier-2 fixture behaved exactly as "
            "in the reference environment. No undeclared dependency surfaced."
        ),
    },
    "clean_env_latest": {
        "what": "empty venv, CPython 3.12, newest solver versions",
        "installed": "scikit-fem 12.0.2, NGSolve 6.2.2606, meshio",
        "result": (
            "Same: every fixture behaved as in the reference environment. "
            "This is what lets the scikit-fem claims be scoped to 12.0.1 "
            "through 12.0.2 and the NGSolve claims to 6.2.2604 through "
            "6.2.2606, instead of to one build."
        ),
        "caveat": (
            "Both clean environments resolved numpy 2.x and a matching "
            "scipy, while the reference environment is on numpy 1.26.x. The "
            "solvers were unaffected across that gap, but a pitfall about "
            "ARRAY behaviour inherits the numpy major version rather than "
            "the solver version and should record it."
        ),
    },
    "clean_env_dune": {
        "what": "empty venv, CPython 3.12, no conda involvement",
        "installed": "dune-fem 2.12.0.2 from PyPI, then mpi4py",
        "result": (
            "Found two defects a same-machine re-run cannot find. (1) "
            "mpi4py is undeclared: the install succeeds and the first import "
            "refuses to proceed without it. (2) With mpi4py added the import "
            "works, but the first JIT-compiled solve then fails to load, "
            "because the sub-build compiled against a different Python that "
            "was earlier on PATH. Both are in the dune pitfalls above."
        ),
    },
    "clean_env_kratos": {
        "what": "empty venv, CPython 3.12, host glibc 2.31",
        "installed": "KratosMultiphysics-all, then KratosMultiphysics 10.4.2",
        "result": (
            "The metapackage installed and imported. Forcing 10.4.2 also "
            "INSTALLED cleanly and then failed to import, which is what "
            "established that the wheel is mis-tagged rather than simply "
            "unavailable — pip's platform check is satisfied and the loader's "
            "is not."
        ),
    },
    "not_tested": {
        "dealii_debug": (
            "No Debug deal.II exists on the machine these entries were "
            "written on, so the Debug half of every build-type claim is "
            "reasoned from deal.II's own source (the Assert macro and the "
            "guards in lac/solver_cg.h) and from what the Release build "
            "demonstrably does NOT do — not from running a Debug build. "
            "Treat the Debug behaviour as sourced, and the Release "
            "behaviour as observed."
        ),
        "dealii_optional_deps": (
            "MPI, PETSc, Trilinos, p4est, SLEPc, HDF5, SUNDIALS, SymEngine "
            "and CGAL are all OFF in the available build, so none of those "
            "code paths could be exercised at all."
        ),
        "febio_with_mkl": (
            "No MKL-enabled FEBio exists here, so what such a build reports "
            "as its default linear solver was not observed. The claim is "
            "limited to what the MKL-less build does report."
        ),
        "sparta_with_kokkos": (
            "The available SPARTA has KOKKOS uninstalled, so only the "
            "REFUSAL messages could be captured. Nothing here claims what an "
            "accelerated run does."
        ),
        "macos_and_windows": (
            "Everything in this file was checked on Linux. None of the "
            "install routes, discovery orders, environment-variable "
            "behaviours or error messages were reproduced on macOS or "
            "Windows, and several are platform-specific by construction — "
            "RUNPATH, glibc symbol versions and manylinux wheel tags have no "
            "macOS equivalent. Treat the macOS notes in "
            "src/core/backend_setup.py as extension points, not findings."
        ),
    },
}


# ── the registry ─────────────────────────────────────────────────────────

SETUP_KNOWLEDGE: dict[str, dict] = {
    "dealii": _DEALII,
    "fenics": _FENICS,
    "fourc": _FOURC,
    "kratos": _KRATOS,
    "dune": _DUNE,
    "febio": _FEBIO,
    "sparta": _SPARTA,
    "ngsolve": _NGSOLVE,
    "skfem": _SKFEM,
}

# Aliases the rest of OASiS uses for the same backend.
_ALIASES = {
    "deal.ii": "dealii", "deal_ii": "dealii", "deal": "dealii",
    "fenicsx": "fenics", "dolfinx": "fenics",
    "4c": "fourc", "four_c": "fourc",
    "dune-fem": "dune", "dunefem": "dune",
    "scikit-fem": "skfem", "scikitfem": "skfem",
}


def get_setup_knowledge(backend: str | None = None) -> dict:
    """Install / setup / build-config knowledge.

    With no argument, returns every backend's entry plus the shared probes —
    which is the right call when the question is "how do I get started" rather
    than "how do I get started with X".
    """
    if not backend:
        return {
            "_how_to_use": (
                "Read `install_route` first, then `pitfalls`. Anything tagged "
                "[Integration][BuildConfig] is a claim that depends on how the "
                "backend was compiled — run the matching probe in "
                "`config_probes` before acting on it. `portability_evidence` "
                "says which of these claims were re-checked in a clean "
                "environment and which were not testable at all."
            ),
            "config_probes": CONFIG_PROBES,
            "portability_evidence": PORTABILITY_EVIDENCE,
            "backends": SETUP_KNOWLEDGE,
        }
    key = backend.strip().lower()
    key = _ALIASES.get(key, key)
    entry = SETUP_KNOWLEDGE.get(key)
    if not entry:
        return {"error": f"No setup knowledge for '{backend}'",
                "known": sorted(SETUP_KNOWLEDGE)}
    return {"backend": key, "config_probes": CONFIG_PROBES,
            "portability_evidence": PORTABILITY_EVIDENCE, **entry}


def get_setup_pitfalls(backend: str) -> list[str]:
    """Just the pitfall strings, for callers that merge them into a larger
    pitfall listing."""
    entry = get_setup_knowledge(backend)
    return list(entry.get("pitfalls", []))
