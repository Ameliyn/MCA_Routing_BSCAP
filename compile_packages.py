import os
import pathlib
import shutil
import subprocess
import importlib.metadata
import sys
import importlib.util
from pathlib import Path
from typing import Optional

# Nuitka is required to compile the packages
# Install it using pip if you haven't already
# pip install nuitka
# Make sure compiled_packages folder is in the same directory as main.py
# Make sure something like sys.path.insert(0, os.path.abspath("compiled_packages")) is at the top of main.py
# Or add the compiled_packages folder to the PYTHONPATH environment variable
# You might have to add the following in .vscode/settings.json to have .pyi files in compiled_packages recognized.
# "python.analysis.extraPaths": ["./Files_To_Copy/compiled_packages"]

REQUIREMENTS_FILE = "requirements.txt"

# Packages that should never be compiled
PACKAGES_TO_IGNORE = ['setuptools', 'mypy', 'mypy-extensions', 'typing-extensions', 'nuitka']

CWD = pathlib.Path().cwd()

COMPILE_TO = "compiled_packages"
COMPILE_OUTPUT = (CWD / COMPILE_TO).as_posix()

# Directory for stubgen output
STUBGEN_OUTPUT = CWD / "stubgen_output"
STUBGEN_OUTPUT.mkdir(exist_ok=True)

SUCCESSFUL_COMPILATIONS: list[str] = []


def has_c_extensions(package_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(package_name)
        if not spec or not spec.origin:
            return False

        package_path = os.path.dirname(spec.origin)

        # Look for compiled C extension ('.so' / '.pyd')
        for _, _, files in os.walk(package_path):
            for file in files:
                if file.endswith(('.so', '.pyd')):
                    return True

    except ModuleNotFoundError:
        print(f"Warning: Package {package_name} not found!")

    return False


def get_installed_package_path(package_name: str) -> Optional[Path]:
    try:
        dist = importlib.metadata.distribution(package_name)
        return Path(str(dist.locate_file('')))
    except importlib.metadata.PackageNotFoundError:
        print(f"Package {package_name} not found.")
        return None


def generate_stub_files(package_name: str) -> Optional[Path]:
    print(f"Generating .pyi stubs for {package_name}...")

    cmd = [
        sys.executable,
        "-m", "mypy.stubgen",
        "-p", package_name,
        "-o", str(STUBGEN_OUTPUT),
        "--ignore-errors"
    ]

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(f"Stubgen stderr for {package_name}:\n{process.stderr}")

        if process.returncode != 0:
            print(f"ERROR: stubgen failed for {package_name} with exit code {process.returncode}")
            return None

    except Exception as e:
        print(f"ERROR while generating stubs for {package_name}: {e}")
        return None

    stub_dir = STUBGEN_OUTPUT / package_name
    if stub_dir.exists():
        return stub_dir

    # Some packages have different naming scheme
    alt_stub = STUBGEN_OUTPUT / package_name.replace("-", "_")
    return alt_stub if alt_stub.exists() else None


def compile_package(package_name: str, package_path: Path):
    if not package_path:
        return

    if not os.path.exists(package_path / package_name):
        print(f"ERROR: Package path {package_path / package_name} does not exist, skipping...")
        return

    if has_c_extensions(package_name):
        print(f"Skipping {package_name}: has C extensions")
        return

    os.chdir(package_path)

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--module",
        f"{package_name}",
        f"--include-package={package_name}",
        f"--output-dir={COMPILE_OUTPUT}",
    ]

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        stdout, stderr = process.communicate()

        if stderr:
            print(f"> STDOUT:\n{stdout}\n")
            print(f"> STDERR:\n{stderr}\n")

        if process.returncode == 0:
            print(f"Successfully compiled {package_name}")
            SUCCESSFUL_COMPILATIONS.append(package_name)
        else:
            print(f"ERROR: Failed to compile {package_name}")

    except Exception as e:
        print(f"ERROR while compiling {package_name}: {e}")
    finally:
        os.chdir(CWD)


def copy_stub_files(stub_dir: Optional[Path], package_name: str) -> None:
    if not stub_dir or not stub_dir.exists():
        print(f"No stub files found for {package_name}, skipping...")
        return

    compiled_pkg_dir = pathlib.Path(COMPILE_OUTPUT) / package_name

    for root, _, files in os.walk(stub_dir):
        rel_path = pathlib.Path(root).relative_to(stub_dir)
        dest_dir = compiled_pkg_dir / rel_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            if file.endswith(".pyi"):
                src = pathlib.Path(root) / file
                dst = dest_dir / file
                shutil.copy2(src, dst)
                print(f"Copied stub: {dst}")


def copy_files(src_dir: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)


def main():
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"{REQUIREMENTS_FILE} not found!")
        return

    with open(REQUIREMENTS_FILE, "r") as f:
        packages = [line.strip().split("==")[0] for line in f if line.strip()]

    for package in packages:
        if package in PACKAGES_TO_IGNORE: continue

        package_path = get_installed_package_path(package)
        if not package_path: continue

        # Skip packages with C extensions
        if has_c_extensions(package):
            print(f"Skipping {package}: has C extensions")
            continue

        stub_dir = generate_stub_files(package)

        compile_package(package, package_path)

        copy_stub_files(stub_dir, package)

    copy_files(COMPILE_OUTPUT, f"Files_To_Copy/{COMPILE_TO}")

    for pkg in SUCCESSFUL_COMPILATIONS:
        print(f"Uninstall {pkg} from venv to use compiled version.")


if __name__ == "__main__":
    main()
