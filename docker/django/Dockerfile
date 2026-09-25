# syntax=docker/dockerfile:1
FROM python:3.12-slim AS wheels

WORKDIR /wheels

COPY requirements.txt /tmp/requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
    && useradd --no-log-init --uid "${APP_UID}" --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels -r /tmp/requirements.txt

COPY . /workspace

# 이미지에 디렉터리를 만들어 두어야 새 named volume이 app 소유로 초기화된다.
RUN mkdir -p /workspace/backend/staticfiles \
    && chown app:app /workspace/backend/staticfiles

USER app

WORKDIR /workspace/backend

EXPOSE 8001
