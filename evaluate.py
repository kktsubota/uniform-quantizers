import argparse
from collections import defaultdict
import os
import subprocess

import pandas as pd


def parse_stdout(score_str: str):
    score_dict = dict()
    # parse standard output
    for line in score_str.strip().split("\n"):
        name, sc = line.split(": ")
        score_dict[name] = float(sc)
    return score_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="/path/to/Kodak/images")
    parser.add_argument(
        "--qua_ent",
        choices={"AUN-Q", "STE-Q", "St-Q", "U-Q", "SGA-Q"},
        default="AUN-Q",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.01,
        dest="lmbda",
        help="Lambda for rate-distortion tradeoff.",
    )
    parser.add_argument("--distortion", default="mse", choices={"mse", "msssim"})
    parser.add_argument("--checkpoint_dir", default="train")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--out", default="score.csv")
    args = parser.parse_args()

    fnames = sorted(os.listdir(args.data))
    scores_dict = defaultdict(list)

    tfci_dir: str = os.path.join(args.checkpoint_dir, "tfci")
    decomp_dir: str = os.path.join(args.checkpoint_dir, "decomp")
    os.makedirs(tfci_dir, exist_ok=True)
    os.makedirs(decomp_dir, exist_ok=True)

    for fname in fnames:
        # compress
        p = subprocess.Popen(
            "python main.py --verbose --qua_ent {} --checkpoint_dir {} compress {} {}.tfci".format(
                args.qua_ent,
                args.checkpoint_dir,
                os.path.join(args.data, fname),
                os.path.join(tfci_dir, fname),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
        print(p.args)
        output = p.communicate()[0]
        p.wait()

        # decompress
        if args.decode:
            p = subprocess.Popen(
                "python main.py --qua_ent {} --checkpoint_dir {} decompress {}.tfci {}.tfci.png".format(
                    args.qua_ent,
                    args.checkpoint_dir,
                    os.path.join(tfci_dir, fname),
                    os.path.join(decomp_dir, fname),
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True,
            )
            print(p.args)
            p.communicate()
            p.wait()

        score_str: str = str(output, encoding="utf-8", errors="replace")
        try:
            score_dict = parse_stdout(score_str)
        except Exception as e:
            print(e)
            print(fname, score_str, output)
            return

        for k, v in score_dict.items():
            scores_dict[k].append(v)

    df = pd.DataFrame.from_dict(scores_dict)
    df.index = fnames
    if args.distortion == "mse":
        df["Loss"] = (
            args.lmbda * df["Mean squared error"] + df["Information content in bpp"]
        )
    else:
        df["Loss"] = (
            args.lmbda * (1 - df["Multiscale SSIM"]) + df["Information content in bpp"]
        )
    df.to_csv(os.path.join(args.checkpoint_dir, args.out))
    print(df.mean())


if __name__ == "__main__":
    main()
