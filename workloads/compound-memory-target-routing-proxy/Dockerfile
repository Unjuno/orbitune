FROM pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

WORKDIR /workspace/orbitune

ENV PYTHONPATH=/workspace/orbitune \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY orbitune/ ./orbitune/
COPY experiments/compound_memory_target_routing_proxy.py ./experiments/compound_memory_target_routing_proxy.py

ENTRYPOINT ["python", "experiments/compound_memory_target_routing_proxy.py"]
