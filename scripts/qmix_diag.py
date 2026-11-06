import sqlite3
import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/TeamShaped/results.db")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output", default="qmix_diagnostics_2.csv")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    query = """
        SELECT run_id, scenario, algo_name, seed
        FROM runs
        WHERE LOWER(algo_name) LIKE '%qmix%'
    """
    params = []

    if args.scenario:
        query += " AND scenario = ?"
        params.append(args.scenario)

    query += " ORDER BY scenario, seed"

    runs = cur.execute(query, params).fetchall()

    if not runs:
        print("No QMIX runs found.")
        conn.close()
        return

    print(f"Found {len(runs)} QMIX runs.")

    rows = []

    for run_id, scenario, algo, seed in runs:

        print(f"Processing: {scenario} | {algo} | Seed {seed}")

        metrics = cur.execute("""
            SELECT metric, global_step, value
            FROM metrics
            WHERE run_id = ?
            ORDER BY metric, global_step
        """, (run_id,)).fetchall()

        for metric, step, value in metrics:
            rows.append({
                "scenario": scenario,
                "algorithm": algo,
                "seed": seed,
                "run_id": run_id,
                "metric": metric,
                "global_step": step,
                "value": value
            })

    # Export to CSV
    fieldnames = [
        "scenario",
        "algorithm",
        "seed",
        "run_id",
        "metric",
        "global_step",
        "value"
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    conn.close()

    print(f"\nExported {len(rows)} metric records.")
    print(f"CSV saved to: {args.output}")


if __name__ == "__main__":
    main()