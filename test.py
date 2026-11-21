import sqlite3, numpy as np
c = sqlite3.connect("outputs/Two/results.db")
for algo, seed, rid in c.execute("SELECT algo_name, seed, run_id FROM runs WHERE algo_name LIKE 'deep_ed%'"):
    g = lambda m: np.array([r[0] for r in c.execute(
        "SELECT value FROM metrics WHERE run_id=? AND metric=? ORDER BY global_step", (rid, m))])
    cf, sm = g("diag/clamp_frac"), g("diag/step_cond_max")
    print(algo, seed, "iters with clamp>0: %.2f" % (cf > 0).mean(), "| max clamp %.2f" % cf.max(),
          "| iters with cond>1: %.2f" % (sm > 1).mean(), "| max cond %.1f" % sm.max())