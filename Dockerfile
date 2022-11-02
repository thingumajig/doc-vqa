FROM python:3.10.7-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install build-essential -y \
    && apt-get install curl -y \
    && curl -sSL https://install.python-poetry.org | python3 -

#ENV PATH="/root/.local/bin:$PATH"
#ENV PATH="${PATH}:/root/.poetry/bin"
ENV PATH="${PATH}:/root/.local/bin"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main
#RUN poetry config virtualenvs.create false
#RUN poetry install --no-interaction --no-ansi --only main

COPY . ./

EXPOSE 8501

CMD [ "poetry", "run", "streamlit", "run", "runner.py" ]