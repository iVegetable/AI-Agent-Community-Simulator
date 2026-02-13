# Contributing

Thanks for your interest in contributing.

## Development Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Run Locally

### Backend

```bash
cd backend
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm run dev
```

## Test and Build Checks

### Backend tests

```bash
cd backend
PYTHONPATH=. python -m pytest tests -q
```

### Frontend build

```bash
cd frontend
npm run build
```

## Pull Request Guidelines

- Keep changes focused and atomic.
- Add or update tests for behavior changes.
- Do not commit secrets, local `.env` files, or generated cache/build artifacts.
- Ensure CI passes before requesting review.
