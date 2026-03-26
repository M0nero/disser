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
    print("  build-dataset    Canonical v2 rebuild: Step1 + overlap audit + Step2")
    print("  smoke-test       Quick Step1+Step2 sanity check")
    print("  infer-stream     Runtime BIO segmentation from live camera/video stream")
    print("  infer-video      Runtime BIO segmentation from a video via MediaPipe Hands")
    print("  infer-skeletons  Runtime BIO segmentation from canonical skeleton sequences")
    print("  export-runtime-bundle  Export a deployable BIO runtime bundle")
    print("  ipn-make-manifest  Build IPN manifest from train/test lists")
    print("  ipn-prelabel     Convert IPN segments to Step1 prelabels")
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
        "build-dataset": _run_build_dataset,
        "smoke-test": _run_smoke_test,
        "infer-stream": _run_infer_stream,
        "infer-video": _run_infer_video,
        "infer-skeletons": _run_infer_skeletons,
        "export-runtime-bundle": _run_export_runtime_bundle,
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


def _run_build_dataset(args: List[str]) -> None:
    from bio.pipeline import build_dataset

    build_dataset.main(args)


def _run_smoke_test(args: List[str]) -> None:
    from bio.pipeline import smoke_test

    smoke_test.main(args)


def _run_infer_stream(args: List[str]) -> None:
    from bio import runtime_commands

    runtime_commands.main_infer_stream(args)


def _run_infer_video(args: List[str]) -> None:
    from bio import runtime_commands

    runtime_commands.main_infer_video(args)


def _run_infer_skeletons(args: List[str]) -> None:
    from bio import runtime_commands

    runtime_commands.main_infer_skeletons(args)


def _run_export_runtime_bundle(args: List[str]) -> None:
    from bio import runtime_commands

    runtime_commands.main_export_runtime_bundle(args)


def _run_ipn_make_manifest(args: List[str]) -> None:
    from bio.ipn import make_manifest

    make_manifest.main(args)


def _run_ipn_prelabel(args: List[str]) -> None:
    from bio.ipn import prelabel

    prelabel.main(args)


if __name__ == "__main__":
    main()
