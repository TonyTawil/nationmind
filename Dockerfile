FROM node:lts as build-deps
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM python:3.11-slim-buster
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=assistant.py
ENV FLASK_RUN_HOST=0.0.0.0


WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ git && apt-get clean

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=build-deps /app/dist ./dist


COPY . /app

EXPOSE 8000

CMD ["python", "app.py"]
