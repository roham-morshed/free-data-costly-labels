PY := .venv/Scripts/python.exe

.PHONY: setup download features test smoke run ablation lfa deep seedcheck artifacts lint all clean

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

download:
	$(PY) scripts/download_datasets.py

features:
	$(PY) scripts/build_features.py

test:
	$(PY) -m pytest tests -q

smoke:
	$(PY) scripts/smoke_test.py

run:
	$(PY) scripts/run_experiments.py

ablation:
	$(PY) scripts/run_experiments.py --ablation

lfa:
	$(PY) scripts/run_lfa_study.py

deep:
	$(PY) scripts/run_deep_baseline.py --budget-seconds 60

artifacts:
	$(PY) scripts/make_paper_artifacts.py

lint:
	$(PY) -m ruff check src scripts tests

seedcheck:
	$(PY) scripts/run_experiments.py --transform-seed-check

all: download features test smoke lfa run ablation deep seedcheck artifacts

clean:
	rm -rf results/raw/* results/lfa/* results/ablation/* results/failures/*
