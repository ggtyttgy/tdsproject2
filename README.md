# Data Analyst Agent (FastAPI)

POST `/api/` with `questions.txt` and optional files. It scrapes data, runs DuckDB/Pandas analysis, and returns answers in the requested JSON format.

## Quickstart

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Test:
```bash
curl -F "questions.txt=@question.txt" http://127.0.0.1:8000/api/
```

## Docker
```bash
docker build -t data-analyst-agent .
docker run -p 8000:8000 data-analyst-agent
```
