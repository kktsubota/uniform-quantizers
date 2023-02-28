import argparse
from pathlib import Path
import subprocess
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--result_dir", help="e.g., outputs/2022-06-20/12-50-15")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result_dir: Path = Path(args.result_dir)
    checkpoint_dir = result_dir / "checkpoint"
    with (result_dir / ".hydra"/ "config.yaml").open() as f:
        config = yaml.safe_load(f)
    assert config["TRAIN"]["ITERATIONS"] == 1000000, result_dir
    assert (checkpoint_dir / "model.ckpt-1000000.meta").exists()

    cmd = [
        "python",
        "encoder.py",
        "--dataset",
        args.dataset,
        "--num_filters",
        str(config["NUM_FILTERS"]),
        "--checkpoint_dir",
        checkpoint_dir.as_posix(),
    ]
    if config["SHALLOW"]:
        cmd.append("--shallow")
    if "PRIOR" in config and config["PRIOR"] == "hyper":
        cmd += [
            "--prior",
            "hyper",
        ]
    if args.dry_run:
        print(" ".join(cmd))
    elif (checkpoint_dir / f"{args.dataset}.csv").exists():
        print(checkpoint_dir / f"{args.dataset}.csv", "exists.")
    else:
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
