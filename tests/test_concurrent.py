# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor

from agent.writer import AgentWriter
from core.coordinator import TransactionCoordinator
from core.transaction import Operation


def test_two_threads_same_cell_one_fails() -> None:
    coordinator = TransactionCoordinator()

    def worker(val: int):
        return coordinator.submit(
            [
                Operation(
                    cell_id="a_0",
                    old_value=None,
                    new_value=val,
                    author_type="human",
                    author_id="u",
                )
            ]
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        r1, r2 = list(ex.map(worker, [1, 2]))

    states = {r1.state.value, r2.state.value}
    assert "SYNCED" in states
    assert "ROLLED_BACK" in states


def test_human_then_llm_sequential_same_cell() -> None:
    coordinator = TransactionCoordinator()
    coordinator.submit(
        [
            Operation(
                cell_id="city_0",
                old_value=None,
                new_value="jakarta",
                author_type="human",
                author_id="user",
            )
        ]
    )

    async def run() -> None:
        writer = AgentWriter(coordinator, agent_id="normalizer", author_type="llm_normalization")
        await writer.normalize("city_0", "DKI Jakarta", confidence=0.97)

    asyncio.run(run())
    assert coordinator.read_fresh(["city_0"])["city_0"] == "DKI Jakarta"


def test_llm_batch_write_multiple_cells() -> None:
    coordinator = TransactionCoordinator()

    async def run() -> None:
        writer = AgentWriter(coordinator, agent_id="agent1", author_type="llm_agent")
        await writer.batch_enrich(
            [
                {"cell_id": "b_0", "value": "2024-01-15", "confidence": 0.89},
                {"cell_id": "b_1", "value": "Positive", "confidence": 0.92},
            ]
        )

    asyncio.run(run())
    result = coordinator.read_fresh(["b_0", "b_1"])
    assert result["b_0"] == "2024-01-15"
    assert result["b_1"] == "Positive"
