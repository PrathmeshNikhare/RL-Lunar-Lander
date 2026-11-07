# Local Setup — Windows 11 / PowerShell / Python 3.12

These steps create a project-local virtual environment and verify that the
core dependencies work. No GPU is required.

## Prerequisites

- Windows 11 (64-bit)
- Python 3.12 (64-bit) from python.org, with the `py` launcher
- Git

Check that 3.12 is installed:

```powershell
py -0
```

The list must contain `-V:3.12`. If another version (for example 3.13 or 3.14)
is the default, a bare `python` or `py` command may start the wrong
interpreter. The commands below therefore name 3.12 explicitly when creating
the environment, and afterwards call the virtual environment's own
`python.exe` directly.

## 1. Create the virtual environment

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version        # expect: Python 3.12.x
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

`.venv/` is listed in `.gitignore` and must not be committed.

## 2. Install dependencies

Choose one:

```powershell
# Reproduce the exact versions verified in Phase 00 (recommended)
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt

# Or resolve the latest compatible versions within the declared ranges
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

| File | Contents |
|---|---|
| `requirements.txt` | Runtime dependencies with compatible version ranges |
| `requirements-dev.txt` | Runtime dependencies plus pytest, ruff and mypy |
| `requirements-lock.txt` | Exact `pip freeze` output of the verified environment |

The PyTorch wheel installed from PyPI on Windows is CPU-only, which is the
intended configuration. The download is several hundred megabytes.

`gymnasium[box2d]` installs a prebuilt `Box2D` wheel for CPython 3.12 on
64-bit Windows, so SWIG and C++ build tools are not required.

## 3. Verify

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest -q
```

`check_environment.py` prints the interpreter, the version of each core
package, the result of constructing `LunarLander-v3` once, and CUDA
availability (informational only). It exits with code 0 only if every check
passes. Expected results:

- `pip check` prints `No broken requirements found.`
- `check_environment.py` prints one `[OK]` line per check and ends with
  `All checks passed.`; `$LASTEXITCODE` is then `0`. Any `[FAIL]` line names
  the failing import or call.
- `pytest` reports every test passed: `10 passed` at Phase 00, 337 at
  Phase 07, and 341 after the PPO-diagnostics
  addition. The upstream Box2D SWIG binding emits
  `DeprecationWarning`s on import; `pytest.ini` ignores exactly those two
  message patterns. Do not run the suite with a blanket
  `-W error::DeprecationWarning`: Box2D then crashes the interpreter on import
  instead of failing a test.

Both install paths in step 2 were verified in fresh virtual environments in
Phase 00 with these results.

## Optional: activating the environment

Activation is not required; every command above calls `.venv\Scripts\python.exe`
directly. If you prefer to activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses with an execution-policy error, allow scripts for the
current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Troubleshooting

- **`python --version` shows a version other than 3.12:** use
  `.\.venv\Scripts\python.exe` explicitly, or recreate the environment with
  `py -3.12 -m venv .venv`.
- **`pip` tries to build Box2D from source:** the interpreter is probably not
  64-bit CPython 3.12. Check `.\.venv\Scripts\python.exe -c "import struct; print(struct.calcsize('P') * 8)"`
  (expect `64`).
- **`pip install` fails with `OSError: [Errno 2] No such file or directory`
  under `torch-...dist-info\licenses\...`:** the project path is too long.
  - The PyTorch 2.15 wheel installs a file about 150 characters deep inside
    `.venv`. Windows limits paths to 260 characters unless long-path support
    is enabled.
  - Clone the project into a short path (the project root should be under
    about 100 characters, for example `D:\Projects\lunar-lander`), or enable
    Windows long paths (<https://pip.pypa.io/warnings/enable-long-paths>).
  - Then delete `.venv` and install again.
  - Found in Phase 07: a clone at a 111-character path failed this way.
