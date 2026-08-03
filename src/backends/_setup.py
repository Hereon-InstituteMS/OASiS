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
        "what": "which linear solver this FEBio was compiled with",
        # `< /dev/null` is REQUIRED, not tidiness. -info prints and then falls
        # through to FEBio's interactive prompt, so without a closed stdin the
        # command never returns.
        "command": "<febio-binary> -info < /dev/null 2>&1 | head -30",
        "reading": (
            "`Default linear solver: skyline` means no vendor solver was "
            "linked; `Default linear solver: pardiso` means an MKL/PARDISO "
            "build. Those are the only two possibilities — FEBio's NumCore.cpp "
            "selects between exactly those two strings on `#ifdef PARDISO`. "
            "-info does NOT print a feature list, so do not grep it for MKL, "
            "MMG, HYPRE or LEVMAR; those return nothing on every build."
        ),
    },
    "sparta_packages": {
        "what": "which optional SPARTA packages are compiled in",
        "command": "cd <sparta-src> && make ps",
        "reading": (
            "Prints `Installed YES/NO: package <NAME>`. IMPORTANT LIMIT: this "
            "covers ONLY the make-level packages, which are `fft` and `python` "
            "— the Makefile's PACKAGE variable literally lists just those two. "
            "It says nothing about KOKKOS, which is not a make package at all. "
            "For KOKKOS use `sparta_kokkos` below."
        ),
    },
    "sparta_kokkos": {
        "what": "whether this SPARTA binary has KOKKOS acceleration",
        # The runtime refusal is the reliable answer. `make ps` cannot see
        # KOKKOS, and the binary contains Kokkos-named symbols either way
        # because accelerator_kokkos.h defines stub classes when the package
        # is absent.
        "command": "<sparta-binary> -kokkos on < /dev/null 2>&1 | head -3",
        "reading": (
            "`ERROR: Cannot use -kokkos on without KOKKOS installed` means no "
            "KOKKOS. A KOKKOS build accepts the switch. Naming is also a "
            "strong hint: KOKKOS builds come from a separate CMake "
            "configuration and are named `spa_kokkos_omp` / `spa_kokkos_cuda` "
            "rather than `spa_serial` / `spa_mpi`. Do NOT try to answer this "
            "by looking for Kokkos symbols in the binary — they are present "
            "even when the package is absent."
        ),
    },
    "kratos_glibc": {
        "what": "whether a Kratos wheel can actually load on this host",
        # Must sweep EVERY shared object in the package, not just the main
        # extension module. On one release the extension module needed only
        # GLIBC_2.14 while libKratosCore.so needed 2.17 — checking one file
        # answers the wrong question.
        "command": (
            "ldd --version | head -1   # host glibc\n"
            "find <site-packages>/KratosMultiphysics -name '*.so*' "
            "-exec objdump -T {} + 2>/dev/null "
            "| grep -o 'GLIBC_2\\.[0-9]*' | sort -uV | tail -1"
        ),
        "reading": (
            "If the highest GLIBC_ symbol required anywhere in the package "
            "exceeds the host glibc, the import fails at runtime even though "
            "pip installed it without complaint — pip checks the wheel's "
            "declared manylinux tag, not its actual symbol requirements, so a "
            "mis-tagged wheel passes that check."
        ),
    },
}


# ── deal.II ──────────────────────────────────────────────────────────────

_DEALII = {
    "verified_on": (
        "TWO deal.II installs on one host, which is itself the common case "
        "and the source of several traps below. (1) A source build, 9.8.0-pre, "
        "CMAKE_BUILD_TYPE=Release, producing only libdeal_II.so — no debug "
        "library. Features ON: ARPACK ASSIMP BOOST GMSH GSL KOKKOS LAPACK "
        "MAGIC_ENUM METIS MUPARSER OPENCASCADE TASKFLOW TBB UMFPACK ZLIB. "
        "Features OFF: MPI P4EST PETSC SLEPC TRILINOS COMPLEX_VALUES HDF5 "
        "SUNDIALS SYMENGINE CGAL ADOLC MUMPS SCALAPACK VTK ARBORX GINKGO "
        "PSBLAS 64BIT_INDICES. (Note DEAL_II_WITH_THREADS is not an "
        "independently configured feature — config.h defines it automatically "
        "as a backwards-compatibility alias whenever TBB is on, so its "
        "presence in the list says nothing extra.) (2) A distribution deal.II "
        "at /usr, 9.1.1, built DebugRelease, shipping BOTH libdeal.ii.so.9.1.1 "
        "and libdeal.ii.g.so.9.1.1 — so Debug behaviour IS observable on this "
        "host, at the older version."
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
        "which is what you want if you care about assertion messages. It is "
        "also deal.II's own DEFAULT — its setup_cached_variables.cmake sets "
        "CMAKE_BUILD_TYPE to DebugRelease and accepts only Release, Debug or "
        "DebugRelease. So a Release-only tree is one where somebody chose "
        "Release deliberately, and is worth double-checking before you "
        "conclude an assertion 'does not fire' in deal.II generally."
    ),
    "pitfalls": [
        "[Integration][Discovery] THE ONLY RELIABLE CHECK IS THE VERSION LINE "
        "CMAKE PRINTS. When a machine has both a source build and a "
        "distribution deal.II — a very common state — find_package can pick "
        "the distribution one without failing and without saying anything "
        "unusual. Signal of the bad outcome: `-- Using the deal.II-9.1.1 "
        "installation found at /usr` when you meant your own newer tree; the "
        "first compile error is then a missing post-9.1 header or symbol, "
        "which reads as a code problem rather than a path problem. Signal of "
        "the good outcome, and note the WORDING DIFFERS so a grep for the "
        "first string will miss it: `-- Using the deal.II-<version> build "
        "directory found at <path>`. Four routes lead to the bad outcome, all "
        "reproduced: DEAL_II_DIR pointing at a bare SOURCE root; DEAL_II_DIR "
        "pointing at an empty directory; DEAL_II_DIR unset; and — the one that "
        "defeats the obvious defense — DEAL_II_DIR set correctly but the "
        "project's own CMakeLists calling plain `find_package(deal.II 9.0 "
        "REQUIRED)` with no HINTS. That last one matters because DEAL_II_DIR "
        "is NOT a CMake-native variable: find_package consults `deal.II_DIR`, "
        "and DEAL_II_DIR works only because deal.II's own example CMakeLists "
        "forward it. Set on a project that does not, CMake ignores it and "
        "warns `Manually-specified variables were not used by the project: "
        "DEAL_II_DIR`. Defense: put the hint in the CMakeLists rather than "
        "relying on the variable — `find_package(deal.II 9.0 REQUIRED HINTS "
        "${DEAL_II_DIR} $ENV{DEAL_II_DIR})` — point it at the BUILD ROOT "
        "(<root>/build works; the deeper <root>/build/lib/cmake/deal.II works "
        "too, but is not required), and then READ BACK the version line and "
        "confirm it names the version you meant. On a host with no "
        "distribution deal.II there is no silent fallback at all: REQUIRED "
        "makes CMake fail loudly instead.",

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
        "followed by `The violated condition was:` and the exception name; for "
        "an out-of-range Vector read specifically, `The violated condition "
        "was: i < size()` and `Index 7 is not in the half-open range [0,3).`, "
        "with the process aborting. Signal, Release: no message at all. "
        "READ THE RELEASE BEHAVIOUR CAREFULLY, because the obvious workaround "
        "does not work: an out-of-range read returns whatever is in the "
        "allocation padding, which is UNINITIALIZED HEAP, not zero. It often "
        "looks like 0.0 on a fresh process, and it is not: recycling the "
        "allocator first (free a larger Vector, or free a poisoned buffer of "
        "the same size) makes the same read return the old contents. So "
        "`if (v[i] == 0.0)` is NOT a usable substitute for the assertion. "
        "Whether the read even survives is also luck — a few elements past the "
        "end sit inside the same aligned block and the program exits 0, while "
        "a far-out index segfaults. Check which build you have with "
        "CONFIG_PROBES['dealii_build_type'] BEFORE acting on an "
        "assertion-based pitfall; do not try to detect the fault at runtime "
        "on a Release build.",

        "[Integration][BuildConfig] A Release deal.II cannot report a CG "
        "breakdown, because the breakdown guard is an `Assert`. In "
        "`lac/solver_cg.h` the divide-by-zero checks are "
        "`Assert(std::abs(previous_r_dot_preconditioner_dot_r) != 0., "
        "ExcDivideByZero())` and `Assert(std::abs(p_dot_A_dot_p) != 0., "
        "ExcDivideByZero())` — compiled out in Release. What a Release build "
        "emits instead is the AssertThrow at the end of the same solve(): "
        "`AssertThrow(solver_state == SolverControl::success, "
        "SolverControl::NoConvergence(it, worker.residual_norm))`. Signal: "
        "`The violated condition was: solver_state == SolverControl::success` "
        "with `Additional information: Iterative method reported convergence "
        "failure in step N. The residual in the last step was <value>.` "
        "DO NOT read that value as nan-means-singular / finite-means-too-few- "
        "iterations. That rule is FALSE and was falsified here on this build: "
        "a rank-one v*v^T matrix is exactly singular and reports a FINITE "
        "residual after exhausting the iteration cap, and a non-symmetric "
        "matrix likewise reports a finite one. Follow deal.II's own wording in "
        "the message instead, which is about MAGNITUDE: a residual that is "
        "still very small suggests too few iterations, a large one suggests a "
        "singular matrix or the wrong solver for it. Worse, CG can fail "
        "SILENTLY: on a negative-definite matrix it returned success with no "
        "diagnostic at all. So the absence of an exception is not evidence the "
        "system was suitable for CG. Check which build you have with "
        "CONFIG_PROBES['dealii_build_type'] first; and note that even a Debug "
        "build only catches the exact-breakdown case — the rank-one singular "
        "matrix gives the same NoConvergence there, and the negative-definite "
        "one still succeeds silently.",

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
        "the absence: a program using `PETScWrappers::` or "
        "`TrilinosWrappers::` fails to COMPILE — it never reaches the linker — "
        "with `error: 'dealii::PETScWrappers' has not been declared`, because "
        "the header's contents are inside an `#ifdef` that is not taken. MPI "
        "is different and quieter: the headers still declare "
        "`Utilities::MPI::`, and the calls degrade to serial stubs that return "
        "1 process and rank 0, so a parallel template built against a "
        "no-MPI deal.II compiles, runs, and silently computes the serial "
        "answer. `MPI_COMM_WORLD` also stops resolving unqualified and must be "
        "written `dealii::MPI_COMM_WORLD`. Check with "
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
        "environment, not a runtime flag:\n"
        "    conda create -n ofa-fenicsx-complex -y -c conda-forge "
        "fenics-dolfinx 'petsc=*=complex*' python=3.12\n"
        "The scalar type is a property of PETSC, so the build-string selector "
        "goes on `petsc` — `petsc=*=complex*` resolves to a petsc build named "
        "`complex_...` and the default resolves to one named `real_...`. "
        "Putting the selector on fenics-dolfinx does NOT work: its build "
        "strings do not contain the word, so the spec matches nothing. Both "
        "lines above were checked with `conda create --dry-run` before being "
        "written here.\n"
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
        "generated code, so an entry left by a DIFFERENT dolfinx build is "
        "found, trusted, and NOT regenerated. Reproduced by corrupting one "
        "cached module: three messages appear and only the middle one is the "
        "cause. Signal, in order: `FileExistsError: [Errno 17] File exists: "
        "<cache>/fenics/libffcx_forms_<hash>.c` — ffcx sees the cached source "
        "and declines to rebuild it; then `ImportError: "
        "<cache>/fenics/libffcx_forms_<hash>.cpython-<py>-<arch>.so: file too "
        "short` — the real failure; then, LAST and therefore the one people "
        "read, a misleading `AttributeError: 'LinearProblem' object has no "
        "attribute '_solver'` raised from `LinearProblem.__del__` during "
        "cleanup. That AttributeError is a consequence, not the fault — do "
        "not debug it. Any ImportError naming a path under the fenics cache "
        "is this class of problem. Defense: delete the cache after upgrading "
        "dolfinx or changing interpreter — `rm -rf \"${XDG_CACHE_HOME:-$HOME/"
        ".cache}/fenics\"` — it only costs one recompile.",

        "[Integration][Portability] dolfinx 0.10.0 made "
        "`petsc_options_prefix` a REQUIRED keyword-only argument of "
        "`dolfinx.fem.petsc.LinearProblem`. Code written against 0.9.x "
        "constructs LinearProblem without it and stops at the constructor. "
        "Signal: `TypeError: LinearProblem.__init__() missing 1 required "
        "keyword-only argument: 'petsc_options_prefix'`. Version range: the "
        "argument does not exist in 0.9.x and is mandatory in 0.10.0, so a "
        "single call cannot satisfy both — branch on "
        "`dolfinx.__version__` if you must support the pair. Confirmed from "
        "the installed signature: the parameter has no default, so it is "
        "required rather than merely accepted. IMPORTANT SCOPE: a conda-forge "
        "install performed TODAY does not get 0.10.0 — a dry-run resolve of "
        "the route above lands on 0.11.0. Every dolfinx API claim in this "
        "catalog was checked against 0.10.0, so on a fresh install the "
        "version you are running and the version the claims were written for "
        "may differ by a minor release. Check `dolfinx.__version__` first and "
        "pin it in the conda spec if you need to reproduce a claim exactly.",

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
        "pip. Install, then IMMEDIATELY prove the import works — that check is "
        "the whole install procedure, because a Kratos wheel can install "
        "cleanly and still be unloadable (below).\n"
        "    python -m pip install KratosMultiphysics "
        "KratosStructuralMechanicsApplication KratosLinearSolversApplication\n"
        "    python -c 'import KratosMultiphysics as KM; "
        "print(KM.KratosGlobals.Kernel.Version())'\n"
        "If that import fails on an older Linux, see the glibc pitfall — the "
        "fix is a version change, not an environment variable.\n"
        "The `KratosMultiphysics-all` metapackage resolves the whole "
        "application set together and is convenient, but be aware of what it "
        "currently does: it lands on 10.3.0 because the newer 10.3.1 "
        "metapackage is unsatisfiable on PyPI (it requires application wheels "
        "at 10.3.1 that were never published), so pip backtracks to the last "
        "consistent set. That is an upstream packaging accident, not a "
        "recommendation — it pins you to an older line than the newest working "
        "release. Prefer naming the applications you actually need."
    ),
    "pitfalls": [
        "[Integration][Install] SOME KRATOS WHEELS ARE MIS-TAGGED AND WILL "
        "INSTALL ON A HOST THEY CANNOT RUN ON. The affected range observed "
        "here is 10.4.0 through 10.4.2 — NOT the whole 10.4 line. "
        "`kratosmultiphysics-10.4.2-1-cp312-cp312-manylinux_2_28_x86_64.whl` "
        "declares manylinux_2_28, so pip accepts it on any host with glibc >= "
        "2.28, but its `Kratos.cpython-*.so` needs GLIBC_2.32 "
        "(`__libc_single_threaded`) and GLIBC_2.34 (`pthread_once`). Tagged "
        "honestly as manylinux_2_34 pip would have refused it. On glibc 2.31 "
        "the install reports success and the import dies. Signal: "
        "`ImportError: /lib/x86_64-linux-gnu/libc.so.6: version "
        "`GLIBC_2.32' not found (required by "
        "<site-packages>/KratosMultiphysics/.libs/Kratos.cpython-312-x86_64-"
        "linux-gnu.so)`. CRITICAL — the very next line Kratos prints is "
        "MISLEADING: `Unable to find KratosCore. Please make sure that your "
        "LD_LIBRARY_PATH or DYLD_LIBRARY_PATH environment variable includes "
        "the path to the Kratos libraries.` No value of LD_LIBRARY_PATH can "
        "fix this. The variable IS honoured — set it and the path in the error "
        "changes — which is exactly why it wastes time: the library is found, "
        "it simply cannot load against this glibc. Defense, in order: (1) "
        "check the host with `ldd --version | head -1`; (2) try the NEWEST "
        "release before assuming you must go backwards — upstream fixed this, "
        "and 10.4.3 imports on glibc 2.31 with nothing above GLIBC_2.17, "
        "verified in a clean environment; (3) only if the newest is also "
        "broken, fall back to the 10.3.x line, whose wheel is "
        "`manylinux2014_x86_64.manylinux_2_17_x86_64`. Do NOT pin to an old "
        "release as a reflex — the mis-tag is a bug that gets fixed, so the "
        "right response is to check, not to freeze. Use "
        "CONFIG_PROBES['kratos_glibc'] on the installed package.",

        "[Integration][Discovery] Kratos is IMPORTED from whichever Python is "
        "running OASiS, and there is no interpreter override — so 'installed' "
        "means 'installed in THAT interpreter'. A perfectly good Kratos in "
        "another environment on the same machine is invisible. Signal: "
        "`discover(query='list')` reports kratos `not_installed` with a "
        "fragment of the ImportError, while running the identical import by "
        "hand under a different interpreter succeeds — they are simply "
        "different environments, and neither message says so. There is a "
        "sharper trap here: KRATOS_ROOT DOES exist, but it only affects the "
        "environment used to RUN a job — it is prepended to PYTHONPATH there "
        "— and is NOT consulted by the availability check. Verified by "
        "pointing KRATOS_ROOT at a working Kratos install and watching "
        "availability still report `not_installed` while the run environment "
        "picked the path up. So the two surfaces can disagree, and the one you "
        "see first is the pessimistic one. Defense: install Kratos into the "
        "environment OASiS itself runs in, and confirm with that same "
        "interpreter: `<oasis-python> -c 'import KratosMultiphysics'`.",

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
        "not pull it, and DUNE refuses to import without it. Confirmed from "
        "the wheel metadata: neither dune-fem's nor dune-common's "
        "Requires-Dist mentions mpi4py, yet the modules were built against "
        "MPI. In a clean virtual environment `pip install dune-fem` completes "
        "successfully and the very first `import dune.fem` fails. It IS a "
        "traceback — a RuntimeError chained off a ModuleNotFoundError — so do "
        "not expect a tidy one-line message. Signal, and this FIRST line is "
        "the one worth grepping for rather than the tail: `The Dune modules "
        "were configured using MPI. For the Python bindings to work, the "
        "Python package 'mpi4py' is required.` The end of the same traceback "
        "spells out the fix: `Please run` / `    pip install mpi4py` / "
        "`before rerunning your Dune script.` Defense: install the pair "
        "together — `python -m pip install dune-fem mpi4py`. mpi4py needs an "
        "MPI implementation with a working `mpicc` to build against.",

        "[Integration][FirstRun] DUNE JIT-COMPILES AGAINST WHATEVER PYTHON ITS "
        "CMAKE FINDS, WHICH IS NOT NECESSARILY THE ONE RUNNING IT — and the "
        "precondition is one that automation hits and humans usually do not. "
        "The generated modules are built by a CMake sub-build whose FindPython3 "
        "picks an interpreter from the environment; if VIRTUAL_ENV is NOT set "
        "— i.e. you invoked `<venv>/bin/python` by absolute path WITHOUT "
        "activating the environment — and a different Python is earlier on "
        "PATH, that other Python wins. A conda base environment on PATH is the "
        "usual culprit. Note the shape of this: a normally ACTIVATED venv is "
        "safe, which is why interactive users rarely see it and a tool "
        "invoking an interpreter path routinely does. Compilation then "
        "SUCCEEDS and the failure appears at import, so it looks like a DUNE "
        "bug rather than an environment one. Signal: `ImportError: "
        "<env-prefix>/.cache/dune-py/python/dune/generated/<module>.so: "
        "undefined symbol: PyThreadState_GetUnchecked` — that symbol is "
        "CPython 3.13+, so seeing it while running 3.12 means the build used "
        "3.13 headers. Any `undefined symbol: Py*` from a file under "
        ".cache/dune-py is this same problem. Reproduced in a clean venv here "
        "with a conda 3.13 first on PATH and VIRTUAL_ENV unset. Defense, "
        "verified to fix it: set VIRTUAL_ENV to the venv prefix (or "
        "CONDA_PREFIX for a conda env, clearing the other), put its bin "
        "directory first on PATH, DELETE the poisoned cache — it is reused "
        "otherwise — and re-run. Setting `Python3_ROOT_DIR` as well is "
        "harmless but was not needed: the fix works without it.",

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

        "[Integration][BuildConfig] MKL PRESENCE CHANGES THE DEFAULT LINEAR "
        "SOLVER, and the way to detect it is that one line, not a feature "
        "list. `febio4 -info` prints `compiled on <date>` and `FEBio version "
        " = <version>`, then a banner, then `Starting without configuration "
        "file` and `Default linear solver: <name>`. It does NOT list build "
        "features, so grepping its output for MKL, MMG, HYPRE or LEVMAR "
        "returns nothing whether or not they are compiled in; any instruction "
        "to detect a feature that way is unusable. Signal of a no-MKL build: "
        "`Default linear solver: skyline`, confirmed here against a CMakeCache "
        "with `USE_MKL:BOOL=OFF` and `MKLROOT:FILEPATH=MKLROOT-NOTFOUND`, and "
        "by `ldd <binary>` showing libgomp and no MKL library. Signal of an "
        "MKL build: `Default linear solver: pardiso`. That second string was "
        "NOT observed — no MKL-enabled FEBio exists on the machine this was "
        "checked on — but it does not need to be guessed either: FEBio's "
        "NumCore.cpp selects the default with `#ifdef PARDISO` "
        "SetDefaultSolverType(\"pardiso\") `#else` "
        "SetDefaultSolverType(\"skyline\"), so those two strings are the only "
        "possibilities and the mapping is exact. Consequence to scope: any "
        "claim recommending or benchmarking the pardiso solver applies only to "
        "an MKL-enabled build, and none of its RUNTIME behaviour was checked "
        "here. Defense: run CONFIG_PROBES['febio_build'] and, for what was "
        "asked at configure time, `grep -E '^USE_|^MKLROOT' "
        "<febio-build-dir>/CMakeCache.txt`.",

        "[Integration][FirstRun] FEBio's own errors are clearly worded and "
        "worth reading literally — they are not stack traces, and the three "
        "startup failures are distinguishable. Signal, a path that cannot be "
        "opened: a boxed `ERROR` banner containing `FATAL ERROR: Failed "
        "opening input file <path>` — the problem is the path or permissions, "
        "not the model. Signal, `-i` omitted: `FATAL ERROR: no model input "
        "file was defined (use -i to define the model input file)`. Signal, a "
        "flag FEBio does not know: `FATAL ERROR: Invalid command line option "
        "'<flag>'.` Note also that FEBio with NO arguments at all does not "
        "exit — it prints its banner and drops into its own interactive "
        "`febio>` prompt, which in an automated context is indistinguishable "
        "from a hang. Always pass `-i <file>`. WORSE, AND THIS TRAPS THE "
        "OBVIOUS PROBE: `-info` and `-h` do NOT exit either. They print and "
        "then fall through to the same prompt, because FEBio returns "
        "`prompt()` whenever no input file was given. So `febio4 -info` in a "
        "script hangs forever unless stdin is already at EOF — which it "
        "happens to be in some shells, making the bug intermittent and "
        "confusing. Always redirect: `febio4 -info < /dev/null`. Note also "
        "that `-v` is a genuine error rather than a prompt — it prints "
        "`FATAL ERROR: Invalid command line option '-v'.` and exits 1.",
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

        "[Integration][BuildConfig] KOKKOS IS A SEPARATE BUILD AND IS ABSENT "
        "FROM A DEFAULT ONE — every accelerated style depends on it. FOUR "
        "distinct refusals, all captured from real runs on a build without it. "
        "Signal, command-line switch: `ERROR: Cannot use -kokkos on without "
        "KOKKOS installed (../sparta.cpp:381)`. Signal, deck command: `ERROR: "
        "Package kokkos command without KOKKOS package enabled "
        "(../input.cpp:1507)`. Signal, suffix switch: `ERROR: Using suffix kk "
        "without KOKKOS package enabled (../sparta.cpp:521)`. Signal, `suffix` "
        "as a DECK command: `ERROR: Unknown command: suffix kk "
        "(../input.cpp:244)` — and that last one deserves care, because "
        "SPARTA's own documentation contradicts it. There IS a doc/suffix.txt "
        "describing a `suffix` command with styles off/on/kk, and it is listed "
        "in the command index; the binary nevertheless has no such command, "
        "because it was documented and never implemented. Anyone checking the "
        "docs will think this entry is mistaken — it is not; the runtime "
        "refusal is what governs. The working spelling is the command-line "
        "pair `-kokkos on -sf kk`. Defense: use "
        "CONFIG_PROBES['sparta_kokkos'], NOT the package probe — `make ps` "
        "cannot answer this, because its PACKAGE list is only `fft python` and "
        "KOKKOS is not a make package. Getting KOKKOS means a separate CMake "
        "build producing a differently named binary (`spa_kokkos_omp` / "
        "`spa_kokkos_cuda`), not a rebuild of this one. And do not try to "
        "settle it by looking for Kokkos symbols: they are present precisely "
        "BECAUSE the package is absent, since accelerator_kokkos.h defines "
        "stub classes in its #else branch. Scope: only the REFUSAL messages "
        "were observed, on a build without the package. Nothing here claims "
        "what an accelerated run does or how much faster it is.",

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
            "The Debug half of the build-type claims was observed, but at a "
            "DIFFERENT VERSION from the Release half. The source build here "
            "is 9.8.0-pre Release-only; the Debug library available on the "
            "same host belongs to a distribution 9.1.1 DebugRelease install. "
            "So 'Debug aborts where Release does not' is observed, while "
            "'Debug at 9.8 behaves exactly as Debug at 9.1.1' is assumed. "
            "Where a Debug message is quoted it comes from 9.1.1."
        ),
        "dealii_optional_deps": (
            "MPI, PETSc, Trilinos, p4est, SLEPc, HDF5, SUNDIALS, SymEngine "
            "and CGAL are all OFF in the available build, so none of those "
            "code paths could be exercised at all."
        ),
        "febio_with_mkl": (
            "No MKL-enabled FEBio exists here, so nothing about how the "
            "pardiso solver BEHAVES at run time was observed. The one thing "
            "that did not need observing is which string such a build prints "
            "as its default solver: NumCore.cpp picks between exactly two "
            "literals on `#ifdef PARDISO`, so that mapping is read from the "
            "source rather than guessed."
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
