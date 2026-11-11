# alp4cam

`alp4cam` is a small DMD-focused toolchain for uploading binary micro-mirror patterns to a ViALUX ALP-4.2 device and projecting them in quick succession.

It is designed around a simple workflow:

1. generate binary patterns or a Hadamard pattern set,
2. store them as PNG files,
3. upload the pattern folder to DMD onboard memory,
4. project the sequence with a chosen picture time and duration.

## Commands

- `alp4cam-patterns` — generate a simple test set or a 2N binary Hadamard set
- `alp4cam-load` — upload a folder of pattern images to the DMD and project them

## Requirements

- Windows PC with the Vialux ALP drivers installed
- Python 3.9+ (64-bit, matching the SDK wheel set)
- ViALUX ALP-4.2 installed; close EasyProj before using the DMD tool
- ALP4lib available through the optional `dmd` dependency

## Install with UV

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dmd,dev]"
```

On Windows PowerShell:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dmd,dev]"
```

## Quick start

Generate a simple test pattern set in a fresh folder under the default pattern directory:

```bash
alp4cam-patterns test --mode-side 32
```

This creates a folder such as `data/patterns/test_20261104_143000`, which can then be loaded directly:

```bash
alp4cam-load --patterns data/patterns/test_20261104_143000 --picture-time-us 10000 --once
```

Generate a Hadamard pattern set for a 32x32 mode grid:

```bash
alp4cam-patterns hadamard --mode-side 32 --macro 2
```

Project the generated folder of patterns:

```bash
alp4cam-load --patterns data/patterns/hadamard_20261104_143000 --picture-time-us 1000 --once
```

## Usage examples

```bash
alp4cam-patterns test --out data/patterns/test
alp4cam-patterns hadamard --out data/patterns/hadamard_32 --mode-side 32 --macro 2
alp4cam-load --patterns data/patterns/hadamard_32 --picture-time-us 1000 --once
alp4cam-load --patterns data/patterns/test --duration 30
```

## Configuration

Defaults are defined in `config/default.yaml` and validated at load time. Unknown keys are rejected so typos fail loudly.

CLI flags override file values and are revalidated before use.

## Notes

- The project is intentionally limited to DMD pattern generation and pattern upload/projection.
- Camera capture has been intentionally removed from this repository so the code stays aligned with the DMD-only workflow.
- The ALP calls are written against the public Vialux ALP-4.2 API surface.
- Hardware validation is still pending on real hardware.

## Development

```bash
uv run --python .venv/bin/python pytest -q
```
