# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Test path bootstrap for running tests without editable install."""

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def reset_shared_state() -> None:
    """Reset all class-level shared state before each test to avoid inter-test pollution."""
    from core.registry import ClusterRegistry
    from core.quic_transport import InMemoryQuicTransport
    from core.dataframe import _RuntimeRegistry
    ClusterRegistry.reset_shared()
    InMemoryQuicTransport._registry.clear()
    _RuntimeRegistry.clear()
    yield
    ClusterRegistry.reset_shared()
    InMemoryQuicTransport._registry.clear()
    _RuntimeRegistry.clear()
