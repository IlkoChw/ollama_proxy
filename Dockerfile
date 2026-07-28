FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app
WORKDIR /home/app
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt
COPY app/ ./app/
COPY templates/ ./templates/
RUN mkdir -p /home/app/data /home/app/templates /home/app/app/static/dashboard \
    && chown -R app:app /home/app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]