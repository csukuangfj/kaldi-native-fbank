# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this repo is

kaldi-native-fbank: kaldi-compatible feature extraction library.
Architecture contract: **C++ core** in `kaldi-native-fbank/csrc/` (Eigen-based,
semantics equal to kaldi, numerically verified against kaldi binaries) +
**Python API** in `kaldi-native-fbank/python/` (pybind11 bindings + package
`kaldi_native_fbank`).  The kaldi checkout `../../kaldi` (i.e. `../kaldi` from
repo root's parent) is the reference for everything.

## Build & test

```bash
mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE=$HOME/py312/bin/python
make -j2
ctest            # use the real ctest binary, see gotchas below
```

- Single test: `cd build && ctest -R <regexp>`.
- Python tests live in `kaldi-native-fbank/python/tests/test_*.py`; C++ gtests
  in `kaldi-native-fbank/csrc/test-*.cc` (registered via `test_srcs` in
  `kaldi-native-fbank/csrc/CMakeLists.txt`).
- Python tests are self-skipping: when a dependency (numpy/torch/the built
  module) is unavailable they print a WARNING and `exit(77)`; ctest marks
  them skipped via `SKIP_RETURN_CODE 77`.  Never exclude them with
  `ctest -E _py` — run the full suite everywhere.
- Style gate (must pass before done):
  - `./scripts/check_style_cpplint.sh 2` (cpplint; downloaded on first run)
  - `./scripts/check_style_clang_format.sh` (clang-format, whole project)
  - `make check` in the build dir (clang-tidy over the compiled sources;
    needs `CMAKE_EXPORT_COMPILE_COMMANDS=ON`, which is set at the top level
    and generates `build/compile_commands.json`).  The repo-root
    `.clang-tidy` mirrors sherpa-onnx's check set (`WarningsAsErrors: '*'`,
    `HeaderFilterRegex` limited to our code).  Where a finding is a
    deliberate kaldi-parity choice or a third-party idiom, use a bare
    `// NOLINT` with a reason comment (category-suffixed NOLINT fails
    cpplint's readability/nolint).  On macOS the `check` target passes the
    SDK path automatically; CI runs it on ubuntu.

## CI (`.github/workflows/`)

- Triggers: push/PR on `master` and `more-feat`.
- Runner preference: **never use `macos-13`** — for macOS x64 use
  `macos-15-intel`.
- Platform wheel builds follow sherpa-onnx's per-platform layout: linux
  (x64, arm64, arm32, **x86/i686**), macos (x64, arm64, **universal2**),
  windows (x86, x64, arm64), termux (4 ABIs: aarch64/arm/x86_64/i686).
- `build-test.yaml` builds AND runs the tests on all those platforms:
  native runners for linux/macos/windows (including macos universal2 via
  `CMAKE_OSX_ARCHITECTURES=arm64;x86_64` and windows-arm64 on
  `windows-11-arm`), arm32 cross-compiled on an **ubuntu arm64 runner**
  (the arm64 kernel runs armhf binaries natively via `libc6:armhf` — no
  qemu), and termux inside `termux/termux-docker:<abi>` containers.  ccache
  (`hendrikmuhs/ccache-action`) is enabled everywhere except windows.
- `style_check.yaml` runs cpplint + clang-format; `clang-tidy.yaml`
  configures with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DKALDI_NATIVE_FBANK_BUILD_TESTS=ON -DKALDI_NATIVE_FBANK_ENABLE_CHECK=ON`
  and runs `make check`.

## Documentation is the first priority — spend time on it

Three documentation layers, all must stay in sync:

1. **C++ doxygen comments** in `kaldi-native-fbank/csrc/*.h`:
   - Content sources from the corresponding kaldi files (read `../kaldi` and
     port the comments; where kaldi has none, write detailed docs anyway).
   - **Every** struct, function, method and field gets a doc; docs must be
     detailed: `@brief`, `@param [in]/[out]`, `@return`, formulas, `@code`
     examples.  One-liners are not enough.
   - Formulas in LaTeX: `@f$ ... @f$` inline, `@f[ ... @f]` display (Doxyfile
     has `USE_MATHJAX=YES`; verify equations render after `make html`).
   - Cross-refs: `@see` ("See also"), `@par Used by:` sections.
   - References to kaldi must be clickable: `@kfile{src/feat/feature-fbank.h}`
     for files, `@kbin{apply-cmvn}` for featbin binaries, markdown
     `[ApplyCmvn](https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc)`
     for functions (2-arg doxygen aliases do not work in doxygen 1.16).
   - Copyright headers of files copied from kaldi keep the original kaldi
     author lines + the local author + "This file is copied/modified from
     kaldi/src/..." inside the block.

2. **Python binding docstrings** (`kaldi-native-fbank/python/csrc/*.cc`):
   - pybind docstrings support **RST** — mirror the C++ doxygen docs in
     detail: `:param:`, `:return:`, `:class:`/`:func:`/`:meth:` cross-refs,
     `.. math::` / `:math:` for formulas, `.. seealso::`, code examples.
   - Use raw string literals `R"doc(...)doc"` for long docs.
   - **Every** class, constructor, method, property and readwrite field gets
     a docstring (a class with bare `def_readwrite` fields is not done); every
     function argument gets a `:param:` entry — **especially init arguments**.
   - References to kaldi classes/functions/binaries are RST inline links to
     the kaldi source (`` `FrameExtractionOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.h>`_ ``),
     matching the C++ `@kfile`/markdown convention; formulas use `:math:`.
   - Constructor ergonomics (pybind11 "default arguments revisited"): all
     option structs expose every field as a keyword arg with the C++ default;
     use `py::arg_v(name, value, "readable_preview")` so signatures show
     `FrameExtractionOptions()` instead of `<... object at 0x...>` and `0.00003`
     instead of `2.9999999242136255e-05`.
   - Set `cls.attr("__module__") = "kaldi_native_fbank";` right after each
     `py::class_` registration and before its `.def`s — signatures then show
     `kaldi_native_fbank.X` instead of `kaldi_native_fbank.lib._kaldi_native_fbank.X`.
   - Register classes before using them as parameter/return types (otherwise
     C++ type names leak into signatures).
   - Release the GIL (`py::call_guard<py::gil_scoped_release>()`) for
     O(T x D) compute ops, not for per-frame O(D) helpers.

3. **Sphinx docs** in `docs/` (sphinx + breathe + autodoc + doxygen):
   - The python-api pages render the binding docstrings via **autodoc**
     (`.. autoclass:: kaldi_native_fbank.X` with `:members:` /
     `:special-members: __init__`); every component gets a page with a usage
     example plus an "API reference" autodoc section.  autodoc imports the
     built module, so build it first (`cd build && make -j2`).
   - CI (`.github/workflows/docs.yaml`) builds the python module before the
     docs for this reason; `docs/requirements.txt` includes numpy (import
     requirement of the module).
   - `cd docs && make html` runs doxygen first, then sphinx; `make latexpdf`
     builds the PDF.  Both must succeed with zero warnings.
   - Every `cpp-api/*.rst` page links the standalone doxygen HTML entry:
     `` `C++ API doxygen <../cpp-api-doxygen/index.html>`_ `` (users must be
     able to find the doxygen tree from the rst pages).
   - `conf.py`: `smartquotes = False` (keep `--flag` literals intact),
     mathjax enabled, breathe points at `../build/doxygen/xml` via an
     absolute path (relative paths resolve against `docs/source/`).
   - `docs/requirements.txt` pins the verified-working set (sphinx 9.x,
     sphinx_rtd_theme 3.x, breathe 5.x — breathe 4.35 crashes on sphinx 9).

4. **Type stubs** — regenerate after ANY binding change:
   `./scripts/generate_python_stubs.sh` (pybind11-stubgen; writes
   `kaldi_native_fbank/__init__.pyi` and `kaldi_native_fbank/lib/*.pyi`).
   Never hand-edit the generated stubs.

## Code style (hard rules)

- **One declaration per statement**: `Type a; Type b;` — never `Type a, b;`
  (applies to arrays too).
- **No `#include <filesystem>`** — it is not portable.  Path handling uses
  plain `std::string` operations.
- **Use Eigen** for matrix/vector math (`colwise()`/`rowwise()`/`cwiseMax()`/
  `select()`, range-for over coefficients is fine) — no raw pointer or index
  loops in Eigen-context code (feature-matrix types).
- Keep function/file names and semantics aligned with kaldi.

## Testing discipline

- Verify each change immediately: build + `ctest` after every C++/binding
  change; `make html` after every doc change.  Never batch tests to the end.
- Parity vs kaldi is the oracle; ignore floating-point rounding and RNG
  differences, report any other behavioral difference to the user.
- Golden test data lives in `tests/data/feat_golden/` (README records the
  kaldi commands that generated it).

## Gotchas

- `ctest`: the pip `cmake` package's `ctest` shim breaks when the knf root is
  on `PYTHONPATH` (its `cmake/` dir shadows the pip module).  Use the real
  binary: `~/py312/lib/python3.12/site-packages/cmake/data/bin/ctest`, with
  `PYTHONPATH=kaldi-native-fbank/python:build/lib`.
- `PYTHON_EXECUTABLE` is only passed by setup.py; plain cmake builds fall back
  to `find_package(Python3)` (see `kaldi-native-fbank/python/CMakeLists.txt`).
- The python module is self-contained: the core lib is **STATIC**, and the
  single `_kaldi_native_fbank*.so` is built into the package dir
  `kaldi_native_fbank/lib/` (import as `kaldi_native_fbank.lib._kaldi_native_fbank`).
- The build must survive `rm -rf build/` (`docs/Makefile` creates its dirs).
- `docs/Makefile` post-processes doxygen HTML: wraps namespace-member names in
  links (stock doxygen only links the namespace label), and copies the
  doxygen tree to `build/html/cpp-api-doxygen/` (remove the old copy first —
  plain `cp -r` nests it).
