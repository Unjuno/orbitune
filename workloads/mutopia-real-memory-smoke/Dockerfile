FROM pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

WORKDIR /workspace/orbitune

ENV PYTHONPATH=/workspace/orbitune \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY orbitune/ ./orbitune/
COPY experiments/recurrent_memory_chunkwise_scan.py ./experiments/recurrent_memory_chunkwise_scan.py
COPY experiments/real_compound_memory_experiment.py ./experiments/real_compound_memory_experiment.py
COPY experiments/real_compound_memory_experiment_matched.py ./experiments/real_compound_memory_experiment_matched.py
COPY experiments/data/manifests/mutopia_public_domain_smoke_v1.json ./experiments/data/manifests/mutopia_public_domain_smoke_v1.json
COPY scripts/fetch_midi_allowlist.py ./scripts/fetch_midi_allowlist.py
COPY scripts/prepare_compound_split.py ./scripts/prepare_compound_split.py
