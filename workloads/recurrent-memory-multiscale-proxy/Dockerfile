FROM pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

WORKDIR /workspace/orbitune

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY experiments/recurrent_memory_multiscale_proxy.py ./experiments/recurrent_memory_multiscale_proxy.py

ENTRYPOINT ["python", "experiments/recurrent_memory_multiscale_proxy.py"]
