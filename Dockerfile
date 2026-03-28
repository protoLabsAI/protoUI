# protoVoice — sub-200ms voice agent
# Single GPU: Whisper STT + Qwen 4B LLM + Kokoro TTS
#
# Build:  docker build -t protovoice .
# Run:    docker run --gpus all -p 7866:7866 -v /path/to/models:/models protovoice

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install deps (layer cached separately from source)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir $(python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(' '.join(d['project']['dependencies']))")

# Install spacy model for Kokoro
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY app.py ./

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV MODEL_DIR=/models
ENV PORT=7866
ENV VLLM_PORT=8100

EXPOSE 7866

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7866/')" || exit 1

CMD ["python3", "app.py"]
