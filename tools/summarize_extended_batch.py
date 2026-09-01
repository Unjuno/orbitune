import json
from pathlib import Path
data = json.loads(Path("runs/cfe_extended_batch.json").read_text())
rows = data["all"]
# Mean events/sec per batch
by_bs = {}
for r in rows:
    by_bs.setdefault(r["batch_size"], []).append(r)
print(f"{'bs':>5} {'mean_ev/s':>10} {'peak_res%':>10} {'OK?':>5} {'GPU%':>6} {'power_W':>9} {'temp_C':>7}")
for bs in sorted(by_bs.keys()):
    rs = by_bs[bs]
    ok = [r for r in rs if r.get("status") == "ok"]
    if not ok:
        print(f"{bs:>5}     {'OOM':>10}")
        continue
    mean_ev = sum(r["events_per_sec"] for r in ok) / len(ok)
    peak_res = ok[0].get("peak_reserved_fraction", 0)
    util = sum(r.get("utilization", 0) for r in ok) / len(ok)
    pwr = sum(r.get("power_draw_watts", 0) for r in ok) / len(ok)
    tmp = sum(r.get("temperature_c", 0) for r in ok) / len(ok)
    safe = peak_res <= 0.92
    print(f"{bs:>5} {mean_ev:>10.0f} {peak_res*100:>9.1f}% {'OK' if safe else 'OVR':>5} {util:>5.1f}% {pwr/1000:>7.1f}kW {tmp:>6.1f}C")