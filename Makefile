.PHONY: setup backend frontend dev test test-models inference-test data seed evaluate train-detector

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

# Fine-tune the fake detector on DeepFakeFace (~25 min on Apple Silicon; needs `make data` first)
train-detector:
	cd backend && uv run python ../scripts/download_train.py --pairs 1500
	cd backend && uv run python ../scripts/train_detector.py crops
	cd backend && uv run python ../scripts/train_detector.py train --epochs 4 --degrade strong --out detector-dff-ft-lowres
	cd backend && uv run python ../scripts/train_detector.py eval
