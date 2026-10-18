# Hugging Face Spaces (Docker SDK) image for the Streamlit demo.
# The Space contains only this Dockerfile and README.md; the code is pulled from GitHub at build time,
# so pushing to GitHub + "Factory rebuild" on the Space is all it takes to redeploy.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Spaces run containers as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    KC_EMBEDDER=fastembed \
    KC_RERANKER=cross-encoder \
    KC_LLM_PROVIDER=offline \
    KC_INDEX_DIR=/home/user/.index
WORKDIR /home/user/app

ARG REPO=https://github.com/Tanzil-Ali-Rizvi/enterprise-knowledge-copilot.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} . && pip install --no-cache-dir --user ".[neural,ui]"

# Download models and build the index at build time so the first visitor doesn't wait
RUN python -c "from kcopilot.pipeline import Copilot; c = Copilot(); print(len(c.index), 'chunks indexed')"

EXPOSE 7860
CMD ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
