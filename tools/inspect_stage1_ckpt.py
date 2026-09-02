"""Inspect the three checkpoint files (best / healthy / latest)."""
import torch
for tag, fname in [('best', 'base-maestro2004.best.pt'), ('healthy', 'base-maestro2004.healthy.pt'), ('latest', 'base-maestro2004.pt')]:
    p = torch.load(f'runs/compound/{fname}', map_location='cpu', weights_only=False)
    h = p.get('health', {})
    print(f'[{tag}] step={p.get("step")} events_seen={p.get("events_seen")} source_commit={p.get("source_commit")[:12]}...')
    print(f'  best_validation_loss={h.get("best_validation_loss")} best_step={h.get("best_step")}')
    print(f'  last_healthy_step={h.get("last_healthy_step")} non_finite_loss={h.get("non_finite_loss_count")} non_finite_grad={h.get("non_finite_grad_count")}')
    print(f'  loss_history_len={len(h.get("loss_history", []))} grad_history_len={len(h.get("grad_norm_history", []))} spike_events={len(h.get("spike_events", []))}')
    print(f'  validation_history_len={len(p.get("validation_history", []))}')
    print()
