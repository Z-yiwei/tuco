#!/usr/bin/env python3
"""Sim-to-real wrapper for the shared Diffusion-Policy selector."""

import sys

from tuco.cli.select_diffusion_baseline import main


if __name__ == "__main__":
    main(["--setting", "sim2real", *sys.argv[1:]])
