from __future__ import annotations

import sys
from typing import Callable, Dict, List


def _print_help() -> None:
    print("BIO pipeline CLI")
    print("")
    print("Usage:")
    print("  python -m bio <command> [args]")
    print("")
    print("Commands:")
    print("  prelabel         Step1: build BIO prelabels from skeletons + CSV")
    print("  synth-build      Step2: build synthetic continuous dataset (offline)")
    print("  train            Train BIO tagger on synthetic dataset")
    print("  smoke-test       Quick Step1+Step2 sanity check")
    print("  ipn-make-manifest  Build IPN manifest from train/test lists")
    print("  ipn-prelabel     Convert IPN segment npz to Step1 prelabels")
    print("")
    print("Run 'python -m bio <command> -h' for command-specific help.")


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return

    cmd = argv[0]
    rest = argv[1:]

    dispatch: Dict[str, Callable[[List[str]], None]] = {
        "prelabel": _run_prelabel,
        "synth-build": _run_synth_build,
        "train": _run_train,
        "smoke-test": _run_smoke_test,
        "ipn-make-manifest": _run_ipn_make_manifest,
        "ipn-prelabel": _run_ipn_prelabel,
    }
    handler = dispatch.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(2)
    handler(rest)


def _run_prelabel(args: List[str]) -> None:
    from bio.pipeline import prelabel

    prelabel.main(args)


def _run_synth_build(args: List[str]) -> None:
    from bio.pipeline import synth_build

    synth_build.main(args)


def _run_train(args: List[str]) -> None:
    from bio.pipeline import train

    train.main(args)


def _run_smoke_test(args: List[str]) -> None:
    from bio.pipeline import smoke_test

    smoke_test.main(args)


def _run_ipn_make_manifest(args: List[str]) -> None:
    from bio.ipn import make_manifest

    make_manifest.main(args)


def _run_ipn_prelabel(args: List[str]) -> None:
    from bio.ipn import prelabel

    prelabel.main(args)


if __name__ == "__main__":
    main()
