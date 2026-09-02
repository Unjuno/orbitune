import json
files = [
    ('1900 (best)',  'runs/compound/full_val_best.json'),
    ('3000 (1e-4)',  'runs/compound/full_val_stage3.json'),
]
print('Per-head delta (3000 minus 1900), sorted by absolute size:')
c0 = json.load(open(files[0][1]))['per_component']
c1 = json.load(open(files[1][1]))['per_component']
deltas = []
for k in ('event_type','a1','channel','control','delta','duration','velocity'):
    d = c1[k]['mean_per_event'] - c0[k]['mean_per_event']
    deltas.append((k, d, c0[k]['mean_per_event'], c1[k]['mean_per_event']))
deltas.sort(key=lambda x: abs(x[1]), reverse=True)
for k, d, v0, v1 in deltas:
    sign = 'favors 3000' if d < 0 else 'favors 1900'
    print(f'  {k:12s} 1900={v0:>9.4f}  3000={v1:>9.4f}  delta={d:+.4f}  -> {sign}')
sum_d = sum(d for _, d, _, _ in deltas)
print(f'  sum delta = {sum_d:+.4f}  (negative = 3000 better overall)')
print(f'  /7 = {sum_d/7:+.4f}  (mean_loss delta)')
