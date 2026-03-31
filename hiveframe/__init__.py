# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Public import facade for distributed_pandas."""

from .core.dataframe import DFrame, read

__all__ = ["DFrame", "read"]
