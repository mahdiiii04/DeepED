import os
import sqlite3
import json
from contextlib import contextmanager
import torch
from omegaconf import DictConfig, OmegaConf

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    env_type    TEXT    NOT NULL,
    scenario    TEXT    NOT NULL,
    algo_name   TEXT    NOT NULL,
    seed        INTEGER NOT NULL,
    cfg_json    TEXT    NOT NULL,
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (env_type, scenario, algo_name, seed)
);

CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    global_step INTEGER NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL  NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_run  ON metrics (run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics (run_id, global_step);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (run_id, metric);

CREATE TABLE IF NOT EXISTS policy_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    global_step INTEGER NOT NULL,
    agent_idx   INTEGER NOT NULL,
    probs_json  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snap_run ON policy_snapshots (run_id);
"""

class ResultsDB:

    def __init__(
        self,
        db_path: str
    ):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._path = db_path
        self._conn = None

    def connect(self):
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(DB_SCHEMA)
        self._conn.commit()
        return self

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def upsert_run(
        self,
        env_type: str,
        scenario: str,
        algo_name: str,
        seed: str,
        cfg: DictConfig
    ) -> int:
        cfg_json = OmegaConf.to_yaml(cfg)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs (env_type, scenario, algo_name, seed, cfg_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (env_type, scenario, algo_name, seed)
                DO UPDATE SET cfg_json=excluded.cfg_json
                """,
                (env_type, scenario, algo_name, seed, cfg_json),
            )
        row = self._conn.execute(
            "SELECT run_id FROM runs WHERE env_type=? AND scenario=? AND algo_name=? AND seed=?",
            (env_type, scenario, algo_name, seed),
        ).fetchone()
        return row[0]

    def log_metrics(
        self,
        run_id: int,
        global_step: int,
        metrics: dict
    ):
        rows = [(run_id, global_step, k, float(v)) for k, v in metrics.items()]
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO metrics (run_id, global_step, metric, value) VALUES (?, ?, ?, ?)",
                rows,
            )

    def log_policy(
        self,
        run_id: int,
        global_step: int,
        avg_policy: torch.Tensor
    ):
        rows = [
            (run_id, global_step, ag, json.dumps(avg_policy[ag].tolist()))
            for ag in range(avg_policy.shape[0])
        ]
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO policy_snapshots (run_id, global_step, agent_idx, probs_json) VALUES (?, ?, ?, ?)",
                rows,
            )