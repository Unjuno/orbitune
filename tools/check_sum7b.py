import json
for n, p in [('1900','runs/compound/full_val_best.json'),('3000','runs/compound/full_val_stage3.json')]:
    d = json.load(open(p)); c = d['per_component']
    s7 = sum(c[k]['mean_per_event'] for k in ('event_type','a1','channel','control','delta','duration','velocity'))
    print(f'{n}: sum7/7={s7/7:.6f}  mean_loss={d["mean_loss_per_event"]:.6f}  diff={d["mean_loss_per_event"]-s7/7:+.6f}')
