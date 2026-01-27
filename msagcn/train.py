# -*- coding: utf-8 -*-
"""CLI entrypoint for training Multi-Stream AGCN."""

from msagcn.training import parse_args, run_training


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

