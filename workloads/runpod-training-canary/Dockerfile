FROM pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

ARG ORBITUNE_SOURCE_SHA=unbaked-local

WORKDIR /workspace/orbitune

ENV PYTHONPATH=/workspace/orbitune \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORBITUNE_SOURCE_SHA=${ORBITUNE_SOURCE_SHA}

LABEL org.opencontainers.image.source="https://github.com/Unjuno/orbitune" \
      org.opencontainers.image.revision="${ORBITUNE_SOURCE_SHA}" \
      io.orbitune.workload="orbitune-runpod-training-canary-v1" \
      io.orbitune.completion-signer.uid="0" \
      io.orbitune.training.uid="10001" \
      io.orbitune.training.gid="10001"

COPY orbitune/ ./orbitune/
COPY workloads/runpod-training-canary/run.py ./workloads/runpod-training-canary/run.py
COPY workloads/runpod-training-canary/entrypoint.py ./workloads/runpod-training-canary/entrypoint.py

RUN mkdir -p /outputs && chmod 0700 /outputs

# The entrypoint remains root only so it can hold the ephemeral completion key.
# It launches the training process as numeric UID/GID 10001 with no supplementary groups.
USER 0:0
ENTRYPOINT ["python", "workloads/runpod-training-canary/entrypoint.py"]
