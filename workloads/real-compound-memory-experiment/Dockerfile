FROM pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

WORKDIR /workspace/orbitune

ENV PYTHONPATH=/workspace/orbitune \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY orbitune/ ./orbitune/
COPY experiments/recurrent_memory_chunkwise_scan.py ./experiments/recurrent_memory_chunkwise_scan.py
COPY experiments/real_compound_memory_experiment.py ./experiments/real_compound_memory_experiment.py

ENTRYPOINT ["python", "experiments/real_compound_memory_experiment.py"]
