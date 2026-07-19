"""Merge chunked Phase-2 / Phase-3 result JSONs and recompute summaries."""
import glob
import json
import math
import sys

import numpy as np


def agg(vals):
    v = np.array([x for x in vals if not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1) if len(v) > 1 else 0),
            "ci95": float(1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0), "n": int(len(v))}


def merge_p2(pattern, out):
    recs = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        recs += [r for r in d["records"] if "log_e" in r]
    le = np.array([r["log_e"] for r in recs], dtype=np.float64)
    e = np.exp(np.clip(le, None, 700))
    n = len(e)
    def rate(a):
        thr = 1.0 / a
        r = float((e >= thr).mean())
        se = math.sqrt(max(r * (1 - r), 1e-12) / n)
        return {"alpha": a, "threshold": thr, "rate": r, "ci95": 1.96 * se, "nominal_ok": r <= a + 1.96 * se}
    summary = {"n_worlds": n, "mean_E": float(e.mean()), "median_E": float(np.median(e)),
               "mean_logE": float(le.mean()),
               "calibration": {f"alpha_{a}": rate(a) for a in (0.01, 0.05, 0.10)}}
    json.dump({"summary": summary, "n_chunks": len(glob.glob(pattern))}, open(out, "w"), indent=2)
    print(f"[P2] n={n}  E[E]={e.mean():.3f} (<=1?)  median={np.median(e):.3f}")
    for a in (0.01, 0.05, 0.10):
        r = rate(a)
        print(f"  alpha={a:.2f} Pr(E>={r['threshold']:.0f})={r['rate']:.4f} +/-{r['ci95']:.4f} "
              f"[{'OK' if r['nominal_ok'] else 'ANTI-CONSERVATIVE'}]")


def merge_p3(pattern, out):
    recs = []
    fams = None
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        recs += [r for r in d["records"] if "zeroshot" in r]
        fams = d.get("families", fams)
    keys = ["crowdfm", "zeroshot", "egated", "always", "oracle_latent", "fullnet_tta",
            "mv", "dawid_skene", "e_value", "gate_fired"]
    summary = {k: agg([r[k] for r in recs if k in r]) for k in keys}
    def paired(k1, k2):
        d = np.array([r[k1] - r[k2] for r in recs if k1 in r and k2 in r])
        if not len(d):
            return None
        return {"mean": float(d.mean()), "ci95": float(1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0), "n": int(len(d))}
    deltas = {"arch(zeroshot-crowdfm)": paired("zeroshot", "crowdfm"),
              "adapt(egated-zeroshot)": paired("egated", "zeroshot"),
              "egated-always": paired("egated", "always"),
              "egated-crowdfm": paired("egated", "crowdfm"),
              "oracle-egated": paired("oracle_latent", "egated"),
              "always-zeroshot": paired("always", "zeroshot")}
    json.dump({"families": fams, "n": len(recs), "summary": summary, "deltas": deltas, "records": recs}, open(out, "w"), indent=2)
    print(f"[P3] families={fams} n={len(recs)}")
    for k in keys:
        s = summary[k]
        if s:
            print(f"  {k:16s} {s['mean']:.4f} +/- {s['ci95']:.4f}")
    print("  --- paired deltas ---")
    for k, d in deltas.items():
        if d:
            print(f"  {k:26s} {d['mean']:+.4f} +/- {d['ci95']:.4f}")


if __name__ == "__main__":
    kind, pattern, out = sys.argv[1], sys.argv[2], sys.argv[3]
    (merge_p2 if kind == "p2" else merge_p3)(pattern, out)
