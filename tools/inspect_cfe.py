import json
recs = json.loads(open('runs/cfe_initial.json').read())['results']
safe = [r for r in recs if r['status']=='ok' and r['peak_reserved_fraction']<=0.92]
print('seq=512 safe (sorted by events/sec):')
for r in sorted(safe, key=lambda r:-r['events_per_sec']):
    if r['seq_len']==512:
        print(f"  n_head={r['n_head']} head_dim={r['head_dim']} fast={r['causal_fastpath']} bs={r['batch_size']} ev/s={r['events_per_sec']:.1f} peak%={r['peak_reserved_fraction']*100:.1f}%")
print()
print('seq=256 safe (top 12):')
for r in sorted(safe, key=lambda r:-r['events_per_sec'])[:12]:
    if r['seq_len']==256:
        print(f"  n_head={r['n_head']} head_dim={r['head_dim']} fast={r['causal_fastpath']} bs={r['batch_size']} ev/s={r['events_per_sec']:.1f} peak%={r['peak_reserved_fraction']*100:.1f}%")