FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .env.example ./
COPY chatbot ./chatbot
ENV PORT=3001
EXPOSE 3001
CMD ["python", "-m", "chatbot.server"]
