# syntax=docker/dockerfile:1
# Always-on Kafka worker (AIS-3634). Named plain "Dockerfile" because the
# shared CI pipeline always builds whatever file is literally named
# "Dockerfile" — see Dockerfile.bulk for the one-shot bulk/backfill image.
FROM python:3.11-slim

WORKDIR /app

# Trust the corporate TLS-inspection certificate so pip can reach PyPI.
COPY zscaler-root-ca.pem /usr/local/share/ca-certificates/zscaler-root-ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Private, org-internal packages. PRIVATE_GIT_ORG is the GitHub org that
# hosts them (--build-arg PRIVATE_GIT_ORG=<org>). The token is passed via
# BuildKit secret mount (--secret id=git_token,env=GIT_TOKEN), not ARG — ARG
# values get baked into image history permanently.
ARG PRIVATE_GIT_ORG
RUN --mount=type=secret,id=git_token \
    test -n "$PRIVATE_GIT_ORG" || { echo "PRIVATE_GIT_ORG build arg is required" >&2; exit 1; } && \
    export GIT_CONFIG_GLOBAL=/tmp/.gitconfig && \
    git config --global url."https://$(cat /run/secrets/git_token)@github.com/".insteadOf "https://github.com/" && \
    pip install --no-cache-dir \
      "python-common-logger @ git+https://github.com/${PRIVATE_GIT_ORG}/python-common-logger.git" \
      "ai-infra-python-sdk-vault @ git+https://github.com/${PRIVATE_GIT_ORG}/ai-platform-infra-sdk.git#subdirectory=packages/ai_infra_python_sdk_vault" \
    && rm -f /tmp/.gitconfig

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Reads all config from environment variables — no CLI args, runs until
# the pod is terminated.
ENTRYPOINT ["python", "-m", "core.kafka_worker_entrypoint"]
