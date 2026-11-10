# Usage: make <target> [MODE=smoke|sample|full]
MODE ?= sample
export RECSYS_MODE=$(MODE)
export TF_USE_LEGACY_KERAS=1
PY ?= python
ENGINE ?= spark

.PHONY: setup data features train retrieval index rank coldstart export serve abtest report test lint demo smoke cache-bench load-test up down clean

setup:            ## create venv + install (run inside WSL2 Ubuntu)
	python3.10 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-dev.txt && pip install -e . --no-deps
data:             ## download (or generate, mode=smoke) + clean + time split
	$(PY) -m recsys.data.download && $(PY) -m recsys.data.prepare && $(PY) -m recsys.data.validate
features:         ## Spark feature pipeline (ENGINE=pandas for the fallback engine)
	$(PY) -m recsys.features.build --engine $(ENGINE)
retrieval:        ## baselines + two-tower
	$(PY) -m recsys.retrieval.train
index:            ## FAISS benchmark + save best index
	$(PY) -m recsys.index.build
rank:             ## DCN-V2 + Wide&Deep + ablations
	$(PY) -m recsys.ranking.train
coldstart:
	$(PY) -m recsys.coldstart.run_eval
train: retrieval index rank coldstart
export:           ## ONNX + Triton repository + serving assets
	$(PY) -m recsys.serving.export
serve:            ## whole stack in Docker (needs `make export` first)
	docker compose up --build
up: serve
down:
	docker compose down
abtest:           ## simulated A/B test (in-process); add API=http://localhost:8080 to replay through the live API
	$(PY) -m recsys.abtest.simulate $(if $(API),--api-url $(API),)
cache-bench:
	$(PY) scripts/cache_benchmark.py
load-test:
	$(PY) scripts/load_test.py --url http://localhost:8080
report:
	$(PY) scripts/make_report.py
demo:             ## Streamlit locally (API must be running)
	API_URL=http://localhost:8080 streamlit run app/streamlit_app.py
test:
	pytest -q
lint:
	ruff check src tests dags scripts app && black --check src tests dags scripts app
smoke:            ## entire pipeline on synthetic data (a few minutes) -> proves the plumbing end to end
	$(MAKE) MODE=smoke data features retrieval index rank coldstart export abtest report
clean:
	rm -rf artifacts/* data/processed model_repository/*/ mlruns logs
