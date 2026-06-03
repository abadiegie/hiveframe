# Chunked Resync Protocol — Internals

> **Audience**: infra reviewers, contributors working on `core/op_log.py` or `core/replication.py`.

---

## Overview

When a follower node lags too far behind the leader (epoch mismatch or missing ops), it triggers a **full resync** instead of incremental pull.  
The protocol transfers the leader's SQLite op-log database as a series of fixed-size binary chunks, verified by SHA-256.

```
follower                              leader
   │                                     │
   │──OPLOG_FULL_RESYNC (mode=manifest)──▶│
   │◀──── {mode, chunk_count, sha256} ───│
   │                                     │
   │──OPLOG_FULL_RESYNC (mode=chunk, 0)─▶│
   │◀──────── {mode, chunk_b64} ─────────│
   │              ...                    │
   │──OPLOG_FULL_RESYNC (mode=chunk, N)─▶│
   │◀──────── {mode, chunk_b64} ─────────│
   │                                     │
   │  [assemble buffer, verify sha256]   │
   │  [restore_sqlite_snapshot_b64()]    │
```

Fallback (if manifest path unavailable or checksum mismatch):

```
follower                              leader
   │──OPLOG_FULL_RESYNC (mode=ops) ─────▶│
   │◀──────── {ops: [...acked ops]} ─────│
   │  [DELETE op_log; apply_acked(ops)]  │
```

---

## Message Types

All messages use `MessagePack` serialization via `core/message.py`.

| Type | Direction | Purpose |
|------|-----------|---------|
| `OPLOG_FULL_RESYNC` | follower → leader | Request manifest, chunk, or ops |
| `OPLOG_FULL_RESYNC_RESPONSE` | leader → follower | Carry manifest metadata, chunk data, or op list |
| `OPLOG_PUSH` / `OPLOG_PUSH_RESPONSE` | follower → leader | Push pending local ops for ack |
| `OPLOG_PULL` / `OPLOG_PULL_RESPONSE` | follower → leader | Incremental pull of acked ops since `since_op_id` |

---

## Phase 1 — Manifest

**Request payload** (`mode=manifest`):
```json
{
  "request_id": "oplog-resync-manifest-<node_id>-<ts_ms>",
  "mode": "manifest"
}
```

**Response payload**:
```json
{
  "mode": "manifest",
  "chunk_count": 4,
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

Leader generates the manifest via:
1. `OperationLog.export_sqlite_snapshot_b64()` — consistent snapshot via `sqlite3.Connection.backup()`
2. `OperationLog.snapshot_chunks(snapshot_b64, chunk_size_bytes=256*1024)` — slices raw bytes, returns `(chunks: list[str], sha256: str)`

Default chunk size: **256 KB** per chunk.

---

## Phase 2 — Chunk Download

For each `chunk_index` in `range(chunk_count)`:

**Request payload**:
```json
{
  "request_id": "oplog-resync-chunk-<N>-<node_id>-<ts_ms>",
  "mode": "chunk",
  "chunk_index": 0
}
```

**Response payload**:
```json
{
  "mode": "chunk",
  "chunk_b64": "<base64-encoded SQLite slice>"
}
```

Follower accumulates all decoded bytes into a `bytearray`, then:
1. Computes `sha256(buffer)` and compares to `manifest.sha256`
2. On match → `restore_sqlite_snapshot_b64()`:
   - Closes current SQLite connection
   - Atomically writes bytes to `db_path`
   - Reopens connection and re-runs `_init_schema()`
3. On mismatch → logs warning, falls through to ops fallback

---

## Phase 3 — Ops Fallback

Used when:
- Leader does not respond to `manifest` mode (older node, backward compat)
- Checksum verification fails
- `chunk_count = 0`

**Request payload**:
```json
{
  "request_id": "oplog-resync-ops-<node_id>-<ts_ms>",
  "mode": "ops"
}
```

**Response payload**:
```json
{
  "ops": [
    {
      "op_id": "2-00000000000000000042",
      "entity": "registry",
      "key": "node/w1",
      "value": { ... },
      "version": 1,
      "origin_node_id": "w1",
      "status": "acked",
      "created_at": 1748400000.0,
      "updated_at": 1748400001.0
    }
  ]
}
```

Follower does a **full wipe + replay**: `DELETE FROM op_log` then `apply_acked(ops)`.

---

## Op ID Format

```
{leader_epoch}-{seq:020d}
# example: 2-00000000000000000042
```

- `leader_epoch` — increments on leader change; used to detect epoch mismatch
- `seq` — monotonically increasing local sequence, zero-padded to 20 digits for lexicographic sort

`OperationLog.estimate_gap(local_op_id, leader_op_id)` returns a rough integer gap.  
An epoch mismatch returns `1_000_000_000` as a conservative trigger for full resync.

---

## WAL Entry Structure (for reference)

WAL entries (`core/wal.py`) are separate from the op-log. They carry **data transactions**:

```json
{
  "lsn": 7,
  "tx_id": "tx-abc123",
  "state": "committed",
  "timestamp": "2026-05-28T10:00:00+00:00",
  "operations": [
    {
      "cell_id": "frame-xyz::col_0",
      "old_value": "jakarta",
      "new_value": "DKI Jakarta",
      "author_type": "llm_normalization",
      "author_id": "normalizer",
      "confidence": 0.97
    }
  ]
}
```

WAL uses a separate replication path (`DELTA` / `SYNC_REQUEST` / `SYNC_RESPONSE`).  
The op-log handles **metadata** (cluster registry, routing, membership); WAL handles **data**.

---

## Entry Points

| Symbol | File | Notes |
|--------|------|-------|
| `OperationLog.full_resync_from_leader()` | `core/op_log.py:346` | Follower entry point |
| `OperationLog.snapshot_chunks()` | `core/op_log.py:470` | Leader chunk producer (static) |
| `OperationLog.export_sqlite_snapshot_b64()` | `core/op_log.py:452` | Leader snapshot exporter |
| `OperationLog.restore_sqlite_snapshot_b64()` | `core/op_log.py:481` | Follower snapshot applier |
| `ReplicationManager._handle_oplog_full_resync()` | `core/replication.py:290` | Leader-side message handler |
| `ReplicationManager.set_oplog_handlers()` | `core/replication.py:335` | Wire leader handlers on startup |

---

## Backward Compatibility

A leader that doesn't support `mode=manifest` will return a response with `mode != "manifest"`.  
The follower detects this and checks for a `snapshot_b64` field (legacy single-message snapshot path).  
If neither is present, the follower silently returns — the next heartbeat cycle will re-trigger the resync.

