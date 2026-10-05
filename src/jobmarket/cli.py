import argparse
from .config import ensure_dirs

def main():
    parser = argparse.ArgumentParser(description="Reproduce the audited AI job market lab")
    parser.add_argument("command", choices=["download", "prepare", "analyze", "train", "report", "all"])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--search-rows", type=int, default=6000)
    parser.add_argument("--resume", action="store_true", help="Reuse completed task/split artifacts after an interrupted training run")
    args = parser.parse_args()
    if args.trials < 1 or args.search_rows < 500:
        parser.error("Use at least one trial and 500 search rows")
    ensure_dirs()
    if args.command == "download":
        from .data import download
        download()
    if args.command in ["prepare", "all"]:
        from .data import prepare
        print(prepare())
    if args.command in ["analyze", "all"]:
        from .analysis import analyze
        analyze()
    if args.command in ["train", "all"]:
        from .modeling import run_training
        run_training(args.trials, args.search_rows, args.resume)
        from .baselines import audit_domain_baselines
        audit_domain_baselines()
    if args.command in ["report", "all"]:
        from .analysis import report
        report()

if __name__ == "__main__":
    main()
