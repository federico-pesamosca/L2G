$ErrorActionPreference = "Stop"

# Supported pythons
$PYTHONS = @("3.10", "3.11", "3.12", "3.13", "3.14")
$ROOT = Resolve-Path "."
$DIST = Join-Path $ROOT "dist"

# Install pythons
uv python install $PYTHONS

# Remove build and dist
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

# Build the wheel for each python
foreach ($py in $PYTHONS) {
    $VENV = ".venv-cp$($py.Replace('.', ''))"

    uv venv --python $py $VENV --allow-existing

    $PY = Join-Path $VENV "Scripts\python.exe"

    # Install the packages inside the venv
    & $PY -m ensurepip
    & $PY -m pip install --upgrade pip setuptools wheel build
    & $PY -m pip install --upgrade cython numpy delvewheel

    # Build the wheel
    & $PY -m build --wheel --no-isolation

    # Repair the wheel
    & $PY -m delvewheel repair `
        --wheel-dir $DIST `
        (Get-ChildItem dist\*.whl | Sort-Object LastWriteTime | Select-Object -Last 1)
}