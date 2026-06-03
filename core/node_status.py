# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0
"""
Node status model: pure state machine functions for capability computation and status transitions.
No I/O, fully deterministic and testable.
"""

from __future__ import annotations


def compute_capability(
    role: str,
    status: str,
    leader_reachable: bool = True,
    wal_reachable: bool = True,
) -> str:
    """
    Compute node capability based on role, status, and reachability flags.

    Returns:
        - "rw" (read-write): node can handle both reads and writes
        - "ro" (read-only): node can only handle reads
        - "drain": node should drain and NOT accept new operations

    Args:
        role: "write" or "read"
        status: "healthy", "suspect", or "failed"
        leader_reachable: whether leader is reachable from this node
        wal_reachable: whether WAL backend is reachable from this node
    """
    # Failed nodes always drain
    if status == "failed":
        return "drain"

    # Suspect nodes downgrade to read-only (if leader is reachable, otherwise drain)
    if status == "suspect":
        if not leader_reachable:
            return "drain"
        return "ro"

    # Healthy node: check role + flags
    if status == "healthy":
        if role == "write":
            # Write nodes need both leader and WAL reachable
            if leader_reachable and wal_reachable:
                return "rw"
            # If only leader reachable, downgrade to read-only
            if leader_reachable and not wal_reachable:
                return "ro"
            # If leader not reachable, drain
            return "drain"
        elif role == "read":
            # Read nodes only need leader reachable for soft verification
            if leader_reachable:
                return "ro"
            # If leader not reachable, read nodes can still serve reads (local cache)
            return "ro"

    # Unknown status or role — default to drain (safe)
    return "drain"


def next_status(current: str, event: str) -> str:
    """
    Compute next status given current status and event.

    Status transitions:
        healthy -> suspect (on heartbeat_timeout)
        healthy -> healthy (on heartbeat_ok)
        suspect -> failed (on suspect_expired)
        suspect -> healthy (on recovered)
        failed -> healthy (on recovered)

    Args:
        current: "healthy", "suspect", or "failed"
        event: "heartbeat_ok", "heartbeat_timeout", "suspect_expired", or "recovered"

    Returns:
        Next status: "healthy", "suspect", or "failed"
    """
    if current == "healthy":
        if event == "heartbeat_ok":
            return "healthy"
        if event == "heartbeat_timeout":
            return "suspect"
        # Any other event keeps healthy
        return "healthy"

    if current == "suspect":
        if event == "recovered":
            return "healthy"
        if event == "suspect_expired":
            return "failed"
        if event == "heartbeat_ok":
            return "healthy"  # Recover if we get a heartbeat
        # Default: stay suspect
        return "suspect"

    if current == "failed":
        if event == "recovered":
            return "healthy"
        # Failed node stays failed until explicitly recovered
        return "failed"

    # Unknown status — keep as is
    return current

