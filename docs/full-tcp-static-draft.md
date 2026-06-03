# Full TCP Static Mode (Draft)

Tujuan: jalankan cluster Hiveframe tanpa NATS, semua transport + koordinasi memakai TCP, dengan scope deployment kecil/terkontrol.

## Scope dan asumsi

- Tidak memakai NATS untuk discovery/coordination.
- Node menggunakan `tcp_transport` untuk antar-node messaging.
- Topologi static-seed (daftar node awal diketahui).
- Direkomendasikan untuk small cluster dengan node churn rendah.

## Konfigurasi yang disarankan

```yaml
cluster:
  mode: full_tcp_static
  node_id: node-a
  bind_host: 0.0.0.0
  bind_port: 7310
  advertise_host: 10.10.1.11
  advertise_port: 7310

  seed_hosts:
    - 10.10.1.11:7310
    - 10.10.1.12:7310
    - 10.10.1.13:7310

  coordinator:
    strategy: static_leader
    leader_node_id: node-a

  heartbeat:
    interval_ms: 1000
    timeout_ms: 5000
    suspect_multiplier: 2

  reconnect:
    base_backoff_ms: 200
    max_backoff_ms: 10000
    jitter: true

registry:
  backend: sqlite
  # Multi-host dengan SQLite per-node + sync periodik berbasis operation log.
  sqlite_path: /var/lib/hiveframe/registry.db
  per_node_sqlite: true

  sync:
    mode: periodic_pull_push
    interval_ms: 1000
    batch_size: 500
    full_resync_threshold_ops: 20000
    # Optimasi bottleneck leader: default steady-state pakai pull.
    # Push dipakai untuk pending/outbox di atas threshold.
    steady_state: pull_preferred
    push_threshold_ops: 100

  consistency:
    model: eventual
    conflict_policy: leader_wins
    authoritative_node_id: node-a

wal:
  backend: mysql  # mysql | redis
  mysql_dsn: mysql://user:pass@10.10.1.20:3306/hiveframe
  redis_url: redis://10.10.1.30:6379/0
  healthcheck_interval_ms: 1000

transport:
  type: tcp
  request_timeout_ms: 3000
  max_inflight: 1024
```

## Perilaku mode `full_tcp_static`

1. **Bootstrap**
   - Node start, bind TCP listener, load `seed_hosts`.
   - Node mencoba connect ke seed satu per satu (dengan backoff + jitter).

2. **Join + Membership**
   - Setelah connect, node kirim `JOIN` ke leader/coordinator.
   - Coordinator menyimpan membership table (`alive/suspect/dead`) dan menyebarkan update.

3. **Heartbeat**
   - Tiap node kirim heartbeat periodik ke coordinator.
   - Jika timeout terlewati -> status `suspect`, lalu `dead` jika tidak pulih.

4. **Routing + Writes**
   - Operasi write/read diarahkan menurut routing table dari coordinator.
   - Message antar-node wajib punya `message_id` untuk idempotency.

5. **Recovery**
   - Saat node reconnect, node sync state minimal (membership/routing/wal offset).
   - Coordinator re-advertise route agar node kembali melayani.

## Integrasi modul (mapping ke codebase sekarang)

- `core/tcp_transport.py`
  - Tambah channel kontrol ringan: `JOIN`, `HEARTBEAT`, `MEMBERSHIP_UPDATE`, `ROUTE_UPDATE`.
  - Tambah reconnect loop + exponential backoff + jitter.

- `core/heartbeat.py`
  - Jadikan sumber status liveness per `node_id`.
  - Expose callback saat transisi status (`alive -> suspect -> dead`).

- `core/registry.py` / `core/sqlite_registry.py`
  - Terapkan single-writer policy untuk metadata kritis.
  - Tambah guard agar split-brain write tidak terjadi saat leader berubah.

- `core/coordinator.py`
  - Tambah mode `static_leader` (read from config).
  - Publish snapshot membership + routing ke semua follower.

## Batasan yang perlu eksplisit

- Bukan discovery service distributed penuh.
- Jika `seed_hosts` salah/obsolete, node bisa gagal join.
- SQLite lokal per-node tidak cukup aman untuk multi-host tanpa shared state policy.
- Failover leader otomatis penuh belum tercakup di draft ini.

## Guardrails operasional minimum

- Jalankan minimal 3 seed entries (meski 1 leader static).
- Monitor metrik: heartbeat RTT, reconnect attempts, membership changes.
- Simpan audit log untuk event `JOIN`, `LEAVE`, `SUSPECT`, `DEAD`, `LEADER_CHANGE`.
- Gunakan timeout konservatif (hindari false-positive node dead).

## Checklist implementasi bertahap

- Phase 1: static leader + static seeds + heartbeat + reconnect.
- Phase 2: membership/routing broadcast yang idempotent.
- Phase 3: hardening registry write policy + recovery sync + status gate (`leader_reachable`/`wal_reachable`).
- Phase 4: simplifikasi API read non-breaking (`read_cluster*` alias) + docs migration.
- Phase 5: optional leader failover/election (kalau dibutuhkan).

## Per-node SQLite + periodic sync (leader di write node)

Asumsi mode ini:

- Tiap node menyimpan metadata cluster di SQLite lokalnya masing-masing.
- Node write (leader) adalah otoritas final untuk metadata kritis.
- Sinkronisasi antarnode dilakukan periodik via TCP dengan replikasi operation log.

### Prinsip utama

- Jangan sync file SQLite mentah antar host.
- Sync dilakukan pada level **operation log** (WAL/outbox entries).
- Semua operasi punya `op_id` unik dan urut (`leader_epoch`, `seq_no`) agar idempotent.
- Follower boleh membuat **proposal/perubahan lokal** di outbox.
- **Commit final metadata kritis hanya oleh leader** (single-writer commit authority).

### Alur sinkronisasi

1. **Local append (proposal)**
   - Node menulis event ke outbox lokal: `op_id`, `entity`, `key`, `value`, `version`, `origin_node_id`, `timestamp`.

2. **Periodic push ke leader (conditional)**
   - Tiap `interval_ms`, follower kirim batch op yang belum ter-ack saat outbox melewati `push_threshold_ops`.
   - Leader validasi, apply, lalu kembalikan ACK berisi `last_applied_op_id`.

3. **Periodic pull dari leader (steady-state)**
   - Follower meminta delta sejak checkpoint terakhir (`last_seen_leader_op_id`).
   - Leader kirim op baru agar follower catch-up.

4. **Conflict handling**
   - Jika key sama berubah di dua node, policy default: `leader_wins`.
   - Follower yang kalah conflict menandai op sebagai `rejected_by_leader` dan apply versi leader.

5. **Full resync**
   - Jika gap op terlalu besar (`full_resync_threshold_ops`) atau checksum mismatch,
     follower lakukan snapshot sync dari leader lalu lanjut incremental op-log.

### Data model minimum untuk log sync

- `op_id`: string/int monotonik global (contoh: `epoch-seq`).
- `entity`: tipe metadata (`membership`, `routing`, `lock`, dst).
- `key`: identitas record.
- `value`: payload serializable.
- `version`: versi record untuk deteksi stale write.
- `origin_node_id`: node asal.
- `status`: `pending | acked | rejected_by_leader`.
- `created_at`, `acked_at`.

### Guardrails operasional

- Leader wajib persist `leader_epoch`; naikkan epoch saat leader restart/failover.
- Terapkan dedup berbasis `op_id` di leader dan follower.
- Expose metrik: `replication_lag_ops`, `outbox_depth`, `conflict_count`, `full_resync_count`.
- Jika leader down, follower masuk mode read-only untuk metadata kritis sampai leader kembali/berganti.

## Node role/status/capability state machine

Tujuan section ini: memastikan status read/write antar node konsisten meskipun tanpa master tetap.

### Terminologi

- `role`: intent statis dari config (`write` atau `read`).
- `status`: health liveness (`healthy`, `suspect`, `failed`).
- `capability`: kemampuan efektif saat ini (`rw`, `ro`, `drain`).

### Aturan capability

- `role=write` + `status=healthy` + `leader_reachable=true` + `wal_reachable=true` -> `capability=rw`.
- `role=write` + `status=healthy` + (`leader_reachable=false` atau `wal_reachable=false`) -> `capability=ro`.
- `role=write` + `status in {suspect, failed}` -> `capability=drain`.
- `role=read` + `status=healthy` -> `capability=ro`.
- `role=read` + `status in {suspect, failed}` -> `capability=drain`.

### Event transisi

- `HEARTBEAT_OK`: set `status=healthy`.
- `HEARTBEAT_TIMEOUT`: `healthy -> suspect`.
- `SUSPECT_EXPIRED`: `suspect -> failed`.
- `LEADER_DOWN`: untuk `role=write`, turunkan `rw -> ro`.
- `LEADER_UP`: untuk `role=write` dan `status=healthy`, evaluasi `ro -> rw` bila `wal_reachable=true`.
- `WAL_DOWN`: untuk `role=write`, turunkan `rw -> ro`.
- `WAL_UP`: untuk `role=write` dan `status=healthy`, evaluasi `ro -> rw` bila `leader_reachable=true`.
- `RECOVERED`: `failed -> healthy` setelah handshake state + checkpoint valid.

### Sumber kebenaran status

- Status node dipublish sebagai event periodik (heartbeat/status-update).
- Reachability signal (`leader_reachable`, `wal_reachable`) dipublish bersama status.
- Untuk menghindari race, simpan record status dengan `status_version` monotonic.
- Update status lama tidak boleh overwrite versi terbaru.

### Routing policy

- Write hanya boleh diarahkan ke node dengan `capability=rw`.
- Read boleh diarahkan ke node `capability in {ro, rw}`.
- Node `capability=drain` tidak menerima traffic baru.

### Safety saat partition

- Node `role=write` yang kehilangan akses ke leader atau shared WAL wajib auto-downgrade ke `ro`.
- Re-upgrade ke `rw` hanya setelah `leader_reachable=true`, `wal_reachable=true`, dan status healthy.
- Saat conflict status antar peer, pilih record dengan `status_version` tertinggi.

## Simplifikasi API read (non-breaking)

Masalah saat ini: `read_fresh()` (local) vs `read_fresh_global_async()` (cluster) membingungkan naming.
Untuk mode TCP, simplifikasi dilakukan bertahap tanpa mematahkan perilaku existing.

### Prinsip kompatibilitas

- `read_fresh()` tetap local snapshot (backward-compatible).
- Hindari mengubah `read_fresh()` menjadi global karena berdampak ke performa write path dan test existing.
- Tambah nama API yang lebih eksplisit untuk akses cluster-wide.

### Target API naming

- Local:
  - `read_fresh()` (existing, dipertahankan)
  - optional alias: `read_local()`
- Global cluster:
  - async: `read_cluster_async()` -> alias ke `read_fresh_global_async()`
  - sync wrapper: `read_cluster()` -> alias ke `read_fresh_global()`

### Migration plan 2 fase

1. **Phase A (non-breaking)**
   - Tambah alias baru (`read_cluster`, `read_cluster_async`, optional `read_local`).
   - Pertahankan seluruh API lama.
   - Dokumentasi/examples baru pakai nama alias baru.

2. **Phase B (soft deprecation)**
   - Tandai `read_fresh_global*` sebagai legacy naming di docs.
   - Tambah warning ringan pada docs/release note (tanpa runtime warning keras).
   - Major release berikutnya bisa evaluasi default naming, bukan behavior.

### Alasan teknis mempertahankan `read_fresh()` sebagai local

- `__setitem__` dan beberapa write-path internal membaca snapshot melalui `read_fresh()`.
- Jika `read_fresh()` dipaksa global, update lokal bisa berubah jadi fan-out network.
- Di cluster yang degraded/partitioned, local read harus tetap tersedia untuk operasional.
