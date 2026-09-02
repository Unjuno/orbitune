import torch, json
ck = torch.load('runs/compound/tbptt/smoke.pt', map_location='cpu', weights_only=False)
ss = ck.get('tbptt_sampler_state', {})
print('tbptt_sampler_state:')
print(json.dumps(ss, indent=2, default=str)[:2000])
print('---')
ts = ck.get('tbptt_stream_states', {})
print('tbptt_stream_states type:', type(ts).__name__, 'len:', len(ts))
if isinstance(ts, list):
    for i, x in enumerate(ts[:2]):
        print(f'  [{i}] type={type(x).__name__} keys={list(x.keys()) if isinstance(x, dict) else "-"}')
        if isinstance(x, dict):
            for k, v in x.items():
                if hasattr(v, 'shape'):
                    print(f'    {k}: tensor shape={v.shape} dtype={v.dtype}')
                elif isinstance(v, list) and v and hasattr(v[0], 'shape'):
                    print(f'    {k}: list[{len(v)}] of tensor shape={v[0].shape}')
                else:
                    s = str(v)
                    print(f'    {k}: {s[:100]}')
