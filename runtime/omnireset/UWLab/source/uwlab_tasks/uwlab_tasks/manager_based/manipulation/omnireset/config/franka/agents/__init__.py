# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Lazy: rsl_rl_cfg is imported by string entry point at gym.make() time,
# not at package-walk time. This avoids tensordict conflicts with --enable_cameras.
