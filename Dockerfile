FROM python:3.11-slim

WORKDIR /app

# Pin the Hugging Face cache to a fixed path rather than relying on $HOME, which differs
# between the root build steps below and the non-root runtime user — otherwise the model
# warmed into the image at build time would be invisible to the app at runtime.
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
# Install the CPU-only torch build explicitly first — the default PyPI wheel bundles the
# full CUDA/cuDNN toolkit (several GB) that this app never uses (EMBEDDING_DEVICE=cpu).
# The subsequent requirements.txt install is then a no-op for torch, since the CPU build
# already satisfies its version constraint there.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt

# Warm the embedding model + tokenizer into the image at build time, so a cold Space
# start doesn't also need to download them over the network before it can serve traffic.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import AutoTokenizer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')"

COPY app ./app

# Hugging Face Spaces (Docker SDK) runs the container as a non-root user.
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
