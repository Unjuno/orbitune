FROM pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

ARG ORBITUNE_SOURCE_SHA=unbaked-local

WORKDIR /workspace/orbitune

ENV PYTHONPATH=/workspace/orbitune \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORBITUNE_SOURCE_SHA=${ORBITUNE_SOURCE_SHA}

LABEL org.opencontainers.image.source="https://github.com/Unjuno/orbitune" \
      org.opencontainers.image.revision="${ORBITUNE_SOURCE_SHA}" \
      io.orbitune.workload="orbitune-runpod-training-canary-v1"

COPY orbitune/ ./orbitune/
COPY workloads/runpod-training-canary/run.py ./workloads/runpod-training-canary/run.py

RUN mkdir -p /outputs

ENTRYPOINT ["python", "workloads/runpod-training-canary/run.py"]
