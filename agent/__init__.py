# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Agent interfaces for hiveframe."""

from .writer import AgentWriter
from .multi_agent import MultiFrameAgent
from .result import FrameInsight, MultiFrameResult, SeriesSpec
from .relational_writer import FrameRelation, RelationalAgentWriter
from .chart_generator import ChartGenerator

__all__ = [
	"AgentWriter",
	"ChartGenerator",
	"MultiFrameAgent",
	"MultiFrameResult",
	"SeriesSpec",
	"FrameInsight",
	"RelationalAgentWriter",
	"FrameRelation",
]
