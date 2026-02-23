import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import argparse

PYTHON_EXEC = sys.executable
CODE_DIR = Path("Files_To_Copy")
MAIN_PY = "main.py"

def read_seeds(seed_file: Path):
    seeds = seed_file.read_text().split()
    if not seeds:
        raise ValueError(f"No seeds found in {seed_file}")
    return seeds

def get_results_folder_from_config(config_file: Path) -> Path:
    """Read RESULTS_FOLDER from the YAML config (resolved relative to Files_To_Copy)."""
    with config_file.open("r") as f:
        cfg = yaml.safe_load(f)
    try:
        return (CODE_DIR / cfg["Config"]["RESULTS_FOLDER"]).resolve()
    except KeyError:
        raise KeyError("Config YAML must contain Config -> RESULTS_FOLDER key")

def run_single(seed: str, run_index: int, config_file: Path, base_results: Path):
    """Run main.py inside Files_To_Copy with a specific seed."""

    # Need / at the end so things like '${RESULTS_FOLDER} + "NetworkGraphs/"' work
    results_dir = (base_results / f"Run{run_index + 1}").resolve().as_posix() + "/"

    cmd = [
        PYTHON_EXEC,
        str(MAIN_PY),
        seed,
        "--no-input",
        f"--config={str(config_file)}",
        f"--RESULTS_FOLDER={str(results_dir)}"
    ]

    # Let stdout go directly to terminal; redirect stderr to stdout
    process = subprocess.Popen(
        cmd,
        cwd=CODE_DIR.resolve().as_posix(),
        stderr=subprocess.STDOUT,
        text=True
    )
    process.wait()

    if process.returncode != 0:
        print(f"[ERROR] Run{run_index+1} (seed={seed}) failed with exit code {process.returncode}", file=sys.stderr)

    return f"[DONE] Run{run_index+1} (seed={seed})"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_runs",
        type=int,
        help="Number of runs to execute (must be positive, use <= total seeds)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config YAML file relative to Files_To_Copy (main.py location)"
    )
    parser.add_argument("--seeds", type=Path, default=Path("seed_numbers.txt"), help="File with seeds")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel runs")
    args = parser.parse_args()

    if args.num_runs <= 0:
        raise ValueError("num_runs must be positive")

    # Resolve config file relative to Files_To_Copy
    config_file = (CODE_DIR / args.config).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    seeds = read_seeds(args.seeds)
    if args.num_runs > len(seeds):
        raise ValueError(f"num_runs ({args.num_runs}) exceeds number of seeds ({len(seeds)})")
    seeds = seeds[:args.num_runs]

    base_results = get_results_folder_from_config(config_file)

    # Run each seed in parallel
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(run_single, seed, idx, config_file, base_results)
            for idx, seed in enumerate(seeds)
        ]
        for f in as_completed(futures):
            print(f.result())

if __name__ == "__main__":
    main()