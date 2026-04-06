# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Agent interfaces for hiveframe."""

from .writer import AgentWriter
from .multi_agent import MultiFrameAgent
from .result import FrameInsight, MultiFrameResult

__all__ = [
	"AgentWriter",
	"MultiFrameAgent",
	"MultiFrameResult",
	"FrameInsight",
]
