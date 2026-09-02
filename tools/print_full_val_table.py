import json
files = [
    ('1900 (best)',  'runs/compound/full_val_best.json'),
    ('2000',         'runs/compound/full_val_latest.json'),
    ('2200 A 3e-4',  'runs/compound/full_val_stage2_A.json'),
    ('2200 B 1e-4',  'runs/compound/full_val_stage2_B.json'),
    ('3000   1e-4',  'runs/compound/full_val_stage3.json'),
]
print(f'{"step":<14} {"mean_loss":>10}  {"evt_type":>9} {"a1":>9} {"channel":>9} {"control":>9} {"delta":>9} {"duration":>9} {"velocity":>9}')
for name, p in files:
    d = json.load(open(p))
    c = d['per_component']
    print(f'{name:<14} {d["mean_loss_per_event"]:>10.4f}  {c["event_type"]["mean_per_event"]:>9.4f} {c["a1"]["mean_per_event"]:>9.4f} {c["channel"]["mean_per_event"]:>9.4f} {c["control"]["mean_per_event"]:>9.4f} {c["delta"]["mean_per_event"]:>9.4f} {c["duration"]["mean_per_event"]:>9.4f} {c["velocity"]["mean_per_event"]:>9.4f}')
