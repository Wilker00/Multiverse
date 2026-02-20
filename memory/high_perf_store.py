"""
memory/high_perf_store.py

Local high-performance memory store with:
- SQLite index for structured query
- Optional Parquet exports when polars is available
- Lightweight vector similarity search (cosine over stored embeddings)

This is designed to work with current Multiverse run artifacts:
`runs/<run_id>/events.jsonl`.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


def _to_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=str)


def _from_json(x: Optional[str]) -> Any:
    if not x:
        return None
    try:
        return json.loads(x)
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _iter_jsonl_rows(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _extract_numeric_values(obj: Any, out: Optional[List[float]] = None) -> List[float]:
    out = out if out is not None else []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _extract_numeric_values(v, out)
        return out
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            _extract_numeric_values(obj[k], out)
        return out
    return out


def _l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    da = a[:n]
    db = b[:n]
    na = _l2_norm(da)
    nb = _l2_norm(db)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    dot = sum(da[i] * db[i] for i in range(n))
    return float(dot / (na * nb))


class HighPerfMemory:
    def __init__(
        self,
        storage_dir: str = "data/memory",
        *,
        use_parquet: bool = True,
        embed_model: Optional[str] = None,
        vector_dim: int = 64,
    ):
        self.storage_dir = storage_dir
        self.use_parquet = bool(use_parquet)
        self.vector_dim = max(8, int(vector_dim))
        os.makedirs(self.storage_dir, exist_ok=True)
        self.db_path = os.path.join(self.storage_dir, "memory.sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        self.embed_model_name = str(embed_model or "").strip()
        self.embed_model = None
        if self.embed_model_name and SentenceTransformer is not None:
            try:
                self.embed_model = SentenceTransformer(self.embed_model_name)
            except Exception:
                self.embed_model = None

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              episode_id TEXT,
              step_idx INTEGER,
              verse_name TEXT,
              policy_id TEXT,
              policy_version TEXT,
              policy_age_steps REAL,
              reward REAL,
              done INTEGER,
              truncated INTEGER,
              obs_json TEXT,
              next_obs_json TEXT,
              action_json TEXT,
              info_json TEXT,
              runtime_confidence_stats_json TEXT,
              embedding_json TEXT,
              created_at_ms INTEGER
            )
            """
        )
        # Lightweight forward-compatible migration for older DBs.
        cols = {str(r[1]) for r in self.conn.execute("PRAGMA table_info(events)").fetchall()}
        if "policy_version" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN policy_version TEXT")
        if "policy_age_steps" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN policy_age_steps REAL")
        if "runtime_confidence_stats_json" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN runtime_confidence_stats_json TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_reward ON events(reward)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              source_dir TEXT,
              imported_at_ms INTEGER,
              event_count INTEGER,
              mean_reward REAL
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _embed_obs(self, obs: Any) -> List[float]:
        if self.embed_model is not None:
            try:
                vec = self.embed_model.encode(str(obs))
                if np is not None:
                    arr = np.array(vec, dtype=float).flatten()
                    return arr.tolist()
                return [float(x) for x in list(vec)]
            except Exception:
                pass

        vals = _extract_numeric_values(obs, [])
        if not vals:
            # Fallback deterministic hash vector from string.
            s = str(obs)
            vals = [float((ord(ch) % 31) / 31.0) for ch in s[: self.vector_dim]]
        if len(vals) < self.vector_dim:
            vals = vals + [0.0] * (self.vector_dim - len(vals))
        return vals[: self.vector_dim]

    def ingest_events(
        self,
        events: Iterable[Any],
        run_id: str,
        *,
        write_parquet: bool = True,
        update_run_summary: bool = True,
    ) -> int:
        run_id = str(run_id)
        now_ms = int(time.time() * 1000)
        rows: List[Tuple[Any, ...]] = []
        rewards: List[float] = []
        for i, e in enumerate(events):
            d = asdict(e) if is_dataclass(e) else dict(e)
            obs = d.get("obs", d.get("observation"))
            next_obs = d.get("next_obs", d.get("next_observation"))
            reward = float(d.get("reward", 0.0))
            rewards.append(reward)
            emb = self._embed_obs(obs)
            info = d.get("info") if isinstance(d.get("info"), dict) else {}
            se = info.get("safe_executor") if isinstance(info, dict) else {}
            se = se if isinstance(se, dict) else {}
            conf_stats = se.get("confidence_status")
            conf_stats = conf_stats if isinstance(conf_stats, dict) else {}
            policy_age_steps = d.get("policy_age_steps", None)
            if policy_age_steps is None:
                policy_age_steps = _safe_float(d.get("step_idx", d.get("step", i)), 0.0)
            event_id = str(d.get("event_id") or f"{run_id}_{i}_{uuid.uuid4().hex[:8]}")
            rows.append(
                (
                    event_id,
                    run_id,
                    str(d.get("episode_id", "")),
                    int(d.get("step_idx", d.get("step", i))),
                    str(d.get("verse_name", "")),
                    str(d.get("policy_id", "")),
                    str(d.get("policy_version", "")),
                    float(policy_age_steps),
                    reward,
                    1 if bool(d.get("done", False)) else 0,
                    1 if bool(d.get("truncated", False)) else 0,
                    _to_json(obs),
                    _to_json(next_obs),
                    _to_json(d.get("action")),
                    _to_json(d.get("info", {})),
                    _to_json(conf_stats),
                    _to_json(emb),
                    now_ms,
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO events(
              event_id, run_id, episode_id, step_idx, verse_name, policy_id, policy_version, policy_age_steps,
              reward, done, truncated, obs_json, next_obs_json, action_json, info_json,
              runtime_confidence_stats_json, embedding_json, created_at_ms
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        if update_run_summary:
            mean_reward = float(sum(rewards) / max(1, len(rewards)))
            self.conn.execute(
                """
                INSERT OR REPLACE INTO runs(run_id, source_dir, imported_at_ms, event_count, mean_reward)
                VALUES(?,?,?,?,?)
                """,
                (run_id, "", now_ms, len(rows), mean_reward),
            )
        self.conn.commit()

        if bool(write_parquet) and self.use_parquet and pl is not None and rows:
            out_path = os.path.join(self.storage_dir, f"{run_id}_events.parquet")
            pl.DataFrame(
                {
                    "event_id": [r[0] for r in rows],
                    "run_id": [r[1] for r in rows],
                    "episode_id": [r[2] for r in rows],
                    "step_idx": [r[3] for r in rows],
                    "verse_name": [r[4] for r in rows],
                    "policy_id": [r[5] for r in rows],
                    "policy_version": [r[6] for r in rows],
                    "policy_age_steps": [r[7] for r in rows],
                    "reward": [r[8] for r in rows],
                    "done": [r[9] for r in rows],
                    "truncated": [r[10] for r in rows],
                    "obs_json": [r[11] for r in rows],
                    "next_obs_json": [r[12] for r in rows],
                    "action_json": [r[13] for r in rows],
                    "info_json": [r[14] for r in rows],
                    "runtime_confidence_stats_json": [r[15] for r in rows],
                    "embedding_json": [r[16] for r in rows],
                    "created_at_ms": [r[17] for r in rows],
                }
            ).write_parquet(out_path, compression="zstd")
        return len(rows)

    def ingest_run(self, run_dir: str) -> int:
        run_dir = str(run_dir)
        run_id = os.path.basename(os.path.normpath(run_dir))
        events_path = os.path.join(run_dir, "events.jsonl")
        if not os.path.isfile(events_path):
            raise FileNotFoundError(f"events file not found: {events_path}")
        chunk_size = 2048
        chunk: List[Dict[str, Any]] = []
        count = 0
        reward_sum = 0.0
        for row in _iter_jsonl_rows(events_path):
            chunk.append(row)
            reward_sum += _safe_float(row.get("reward", 0.0), 0.0)
            if len(chunk) >= chunk_size:
                count += self.ingest_events(
                    events=chunk,
                    run_id=run_id,
                    write_parquet=False,
                    update_run_summary=False,
                )
                chunk.clear()
        if chunk:
            count += self.ingest_events(
                events=chunk,
                run_id=run_id,
                write_parquet=False,
                update_run_summary=False,
            )
            chunk.clear()

        now_ms = int(time.time() * 1000)
        mean_reward = float(reward_sum / float(max(1, count)))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs(run_id, source_dir, imported_at_ms, event_count, mean_reward)
            VALUES(?,?,?,?,?)
            """,
            (run_id, run_dir, now_ms, count, mean_reward),
        )
        self.conn.commit()
        return count

    def ingest_runs(self, runs_root: str = "runs") -> Dict[str, int]:
        out: Dict[str, int] = {}
        if not os.path.isdir(runs_root):
            return out
        for name in sorted(os.listdir(runs_root)):
            run_dir = os.path.join(runs_root, name)
            if not os.path.isdir(run_dir):
                continue
            events_path = os.path.join(run_dir, "events.jsonl")
            if not os.path.isfile(events_path):
                continue
            try:
                out[name] = int(self.ingest_run(run_dir))
            except Exception:
                out[name] = 0
        return out

    def query_similar(
        self,
        query_obs: Any,
        top_k: int = 10,
        run_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        q_emb = self._embed_obs(query_obs)
        clauses = []
        params: List[Any] = []
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            clauses.append(f"run_id IN ({placeholders})")
            params.extend([str(x) for x in run_ids])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
        SELECT event_id, run_id, episode_id, step_idx, reward, action_json, info_json, embedding_json
        FROM events
        {where}
        """
        rows = self.conn.execute(sql, params).fetchall()
        scored: List[Dict[str, Any]] = []
        for r in rows:
            emb = _from_json(r["embedding_json"]) or []
            if not isinstance(emb, list):
                continue
            try:
                emb_f = [float(x) for x in emb]
            except Exception:
                continue
            sim = _cosine(emb_f, q_emb)
            scored.append(
                {
                    "event_id": r["event_id"],
                    "run_id": r["run_id"],
                    "episode_id": r["episode_id"],
                    "step_idx": int(r["step_idx"] or 0),
                    "reward": float(r["reward"] or 0.0),
                    "action": _from_json(r["action_json"]),
                    "info": _from_json(r["info_json"]) or {},
                    "similarity": float(sim),
                }
            )
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[: max(1, int(top_k))]

    def sql_query(self, query: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        cur = self.conn.execute(query, list(params or []))
        cols = [d[0] for d in cur.description]
        return [{cols[i]: row[i] for i in range(len(cols))} for row in cur.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT
              COUNT(*) AS total_events,
              AVG(reward) AS mean_reward,
              MIN(reward) AS min_reward,
              MAX(reward) AS max_reward,
              SUM(CASE WHEN done=1 THEN 1 ELSE 0 END) AS terminal_events
            FROM events
            """
        ).fetchone()
        runs = self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        return {
            "storage_dir": self.storage_dir,
            "db_path": self.db_path,
            "total_runs": int(runs),
            "total_events": int(row[0] or 0),
            "mean_reward": float(row[1] or 0.0),
            "min_reward": float(row[2] or 0.0),
            "max_reward": float(row[3] or 0.0),
            "terminal_events": int(row[4] or 0),
        }

    def export_dataset(self, query: str, output_path: str, fmt: str = "jsonl") -> str:
        rows = self.sql_query(query)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fmt = str(fmt).strip().lower()
        if fmt == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return output_path
        if fmt == "csv":
            import csv

            with open(output_path, "w", encoding="utf-8", newline="") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            return output_path
        if fmt == "parquet":
            if pl is None:
                raise RuntimeError("parquet export requires polars")
            pl.DataFrame(rows).write_parquet(output_path, compression="zstd")
            return output_path
        raise ValueError(f"Unsupported export format: {fmt}")


def create_high_perf_store(storage_dir: str = "data/memory") -> HighPerfMemory:
    return HighPerfMemory(storage_dir=storage_dir)


def migrate_jsonl_to_high_perf(runs_root: str, memory_store: HighPerfMemory) -> Dict[str, int]:
    return memory_store.ingest_runs(runs_root=runs_root)


if __name__ == "__main__":
    store = HighPerfMemory(storage_dir="data/memory_test")
    stats = store.ingest_runs("runs")
    print("ingested_runs:", len(stats))
    print(json.dumps(store.get_statistics(), ensure_ascii=False, indent=2))
