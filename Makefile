.PHONY: setup backend frontend dev test test-models inference-test data seed evaluate

setup:
	cd backend && uv sync
	cd frontend && npm install

backend:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

dev:
	$(MAKE) -j2 backend frontend

test:
	cd backend && uv run pytest -m "not models" -q

test-models:
	cd backend && uv run pytest -m models -q

inference-test:
	cd backend && uv run python ../scripts/inference_test.py

data:
	cd backend && uv run python ../scripts/download_data.py

seed:
	cd backend && uv run python ../scripts/seed_enroll.py

evaluate:
	cd backend && uv run python ../scripts/evaluate.py
