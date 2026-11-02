import argparse
import glob
import os
import sqlite3
from typing import Dict, List

# Re-use the schema definition from your db.py
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

def merge_database(src_path: str, target_conn: sqlite3.Connection):
    src_conn = sqlite3.connect(src_path)
    src_conn.row_factory = sqlite3.Row
    
    # Mapping from source run_id -> target run_id
    id_map: Dict[int, int] = {}

    with target_conn:
        # 1. Merge 'runs' table
        runs = src_conn.execute(
            "SELECT run_id, env_type, scenario, algo_name, seed, cfg_json, started_at FROM runs"
        ).fetchall()

        for r in runs:
            target_conn.execute(
                """
                INSERT INTO runs (env_type, scenario, algo_name, seed, cfg_json, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (env_type, scenario, algo_name, seed)
                DO UPDATE SET cfg_json=excluded.cfg_json
                """,
                (r["env_type"], r["scenario"], r["algo_name"], r["seed"], r["cfg_json"], r["started_at"]),
            )
            
            # Fetch the assigned or updated run_id in the target DB
            new_run_id = target_conn.execute(
                "SELECT run_id FROM runs WHERE env_type=? AND scenario=? AND algo_name=? AND seed=?",
                (r["env_type"], r["scenario"], r["algo_name"], r["seed"]),
            ).fetchone()[0]
            
            id_map[r["run_id"]] = new_run_id

        # 2. Merge 'metrics' table
        metrics_cur = src_conn.execute(
            "SELECT run_id, global_step, metric, value FROM metrics"
        )
        batch_size = 50000
        while True:
            rows = metrics_cur.fetchmany(batch_size)
            if not rows:
                break
            remapped_metrics = [
                (id_map[row["run_id"]], row["global_step"], row["metric"], row["value"])
                for row in rows
                if row["run_id"] in id_map
            ]
            target_conn.executemany(
                "INSERT INTO metrics (run_id, global_step, metric, value) VALUES (?, ?, ?, ?)",
                remapped_metrics,
            )

        # 3. Merge 'policy_snapshots' table
        snaps_cur = src_conn.execute(
            "SELECT run_id, global_step, agent_idx, probs_json FROM policy_snapshots"
        )
        while True:
            rows = snaps_cur.fetchmany(batch_size)
            if not rows:
                break
            remapped_snaps = [
                (id_map[row["run_id"]], row["global_step"], row["agent_idx"], row["probs_json"])
                for row in rows
                if row["run_id"] in id_map
            ]
            target_conn.executemany(
                "INSERT INTO policy_snapshots (run_id, global_step, agent_idx, probs_json) VALUES (?, ?, ?, ?)",
                remapped_snaps,
            )

    src_conn.close()

def main():
    parser = argparse.ArgumentParser(description="Merge multiple SQLite results.db files into one.")
    parser.add_argument(
        "--src_pattern",
        type=str,
        default="run_*.db",
        help="Glob pattern or directory containing source SQLite databases (e.g., 'outputs/**/run_*.db')",
    )
    parser.add_argument(
        "--output_db",
        type=str,
        default="results.db",
        help="Path to the destination merged database",
    )
    args = parser.parse_args()

    # Find matching files
    src_files: List[str] = glob.glob(args.src_pattern, recursive=True)
    src_files = [f for f in src_files if os.path.abspath(f) != os.path.abspath(args.output_db)]

    if not src_files:
        print(f"No database files found matching: {args.src_pattern}")
        return

    print(f"Found {len(src_files)} database(s) to merge into '{args.output_db}'.")

    # Initialize destination database
    os.makedirs(os.path.dirname(args.output_db) or ".", exist_ok=True)
    target_conn = sqlite3.connect(args.output_db)
    
    # Pragmas for fast bulk ingestion
    target_conn.execute("PRAGMA synchronous = OFF;")
    target_conn.execute("PRAGMA journal_mode = MEMORY;")
    target_conn.executescript(DB_SCHEMA)

    for idx, db_file in enumerate(src_files, 1):
        print(f"[{idx}/{len(src_files)}] Merging {db_file}...")
        try:
            merge_database(db_file, target_conn)
        except Exception as e:
            print(f"  --> Error merging {db_file}: {e}")

    # Reset pragmas and optimize indices
    target_conn.execute("PRAGMA optimize;")
    target_conn.close()
    print("Done! All runs successfully merged.")

if __name__ == "__main__":
    main()