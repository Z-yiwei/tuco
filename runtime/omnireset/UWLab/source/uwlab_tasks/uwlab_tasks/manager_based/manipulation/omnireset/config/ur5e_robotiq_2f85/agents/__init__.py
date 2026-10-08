# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Lazy: rsl_rl_cfg is resolved by string entry point at gym.make() time.
# Eager import removed to avoid tensordict conflicts with --enable_cameras.
