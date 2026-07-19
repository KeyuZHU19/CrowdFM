"""Aggregate the Go/No-Go 3-way results across 3 seeds and print the verdict."""
import json, glob, sys
import numpy as np

def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(pat))]

def synth(pat, label, thr):
    ds = load(pat)
    if not ds:
        print(f"{label}: no files"); return
    X = np.mean([d["X_official"]["mean"] for d in ds])
    Y = np.mean([d["Y_nomech"]["mean"] for d in ds])
    Z = np.mean([d["Z_full"]["mean"] for d in ds])
    ZX = np.mean([d["Z_minus_X"]["mean"] for d in ds])
    YX = np.mean([d["Y_minus_X"]["mean"] for d in ds])
    ZY = np.mean([d["Z_minus_Y"]["mean"] for d in ds])
    verdict = "PASS" if ZX >= thr else "FAIL"
    print(f"\n=== {label} (n_seeds={len(ds)}, {ds[0]['n']} worlds each) ===")
    print(f"  X official-frozen CrowdFM : {X:.4f}")
    print(f"  Y capacity-matched no-mech: {Y:.4f}   (Y-X = {YX:+.4f})")
    print(f"  Z full CrowdSI            : {Z:.4f}   (Z-X = {ZX:+.4f})")
    print(f"  mechanism effect  Z-Y     : {ZY:+.4f}")
    print(f"  >>> Z-X = {ZX*100:+.2f}pp  threshold {thr*100:.1f}pp  [{verdict}]")

def real(pat, label, thr):
    ds = load(pat)
    if not ds:
        print(f"{label}: no files"); return
    ZX = np.mean([d["mean_Z_minus_X"] for d in ds])
    YX = np.mean([d["mean_Y_minus_X"] for d in ds])
    # also per-dataset averaged over seeds
    names = list(ds[0]["rows"].keys())
    print(f"\n=== {label} (n_seeds={len(ds)}, {len(names)} datasets) ===")
    Xa=np.mean([np.mean([d['rows'][n]['X_official'] for n in names]) for d in ds])
    Ya=np.mean([np.mean([d['rows'][n]['Y_nomech'] for n in names]) for d in ds])
    Za=np.mean([np.mean([d['rows'][n]['Z_full'] for n in names]) for d in ds])
    print(f"  X official={Xa:.4f}  Y no-mech={Ya:.4f}  Z full={Za:.4f}")
    verdict = "PASS" if ZX >= thr else "FAIL"
    print(f"  mean Y-X = {YX:+.4f}   mean Z-X = {ZX:+.4f}")
    print(f"  >>> Z-X = {ZX*100:+.2f}pp  threshold {thr*100:.1f}pp  [{verdict}]")

base = sys.argv[1] if len(sys.argv) > 1 else "runs/gonogo"
print("############ GO / NO-GO 3-WAY VERDICT ############")
synth(f"{base}/inprior_s*.json", "IN-PRIOR", 0.015)
synth(f"{base}/heldout_s*.json", "HELD-OUT (coalition)", 0.015)
real(f"{base}/real_s*.json", "REAL DATASETS", 0.010)
