import json
files = [
    ('1900 (best)',  'runs/compound/full_val_best.json'),
    ('2000',         'runs/compound/full_val_latest.json'),
    ('2200 A 3e-4',  'runs/compound/full_val_stage2_A.json'),
    ('2200 B 1e-4',  'runs/compound/full_val_stage2_B.json'),
    ('3000   1e-4',  'runs/compound/full_val_stage3.json'),
]
print(f'{"label":<14} {"mean_loss":>10}  {"sum7":>10}')
for name, p in files:
    d = json.load(open(p))
    c = d['per_component']
    s = sum(c[k]['mean_per_event'] for k in ('event_type','a1','channel','control','delta','duration','velocity'))
    print(f'{name:<14} {d["mean_loss_per_event"]:>10.4f}  {s:>10.4f}  (diff: {d["mean_loss_per_event"]-s:+.4f})')
