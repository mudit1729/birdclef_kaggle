PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: setup lint test pull-baseline download train train-local infer submit submit-code submit-kernel build-kaggle build-kaggle-submit push-kaggle push-kaggle-submit

setup:
	$(UV) sync --python /opt/homebrew/bin/python3.11 --extra dev

lint:
	$(UV) run ruff check .

test:
	$(UV) run pytest

pull-baseline:
	$(UV) run birdclef-pull-baseline

download:
	$(UV) run birdclef-download

train:
	$(UV) run birdclef-train --download-if-missing

train-local:
	$(UV) run python scripts/train_local_subset.py

infer:
	$(UV) run birdclef-infer --download-if-missing

submit:
	$(UV) run birdclef-submit

submit-code:
	$(UV) run birdclef-submit-code

submit-kernel:
	$(UV) run birdclef-submit-kernel

build-kaggle:
	$(UV) run python scripts/build_kaggle_kernel.py

build-kaggle-submit:
	$(UV) run python scripts/build_kaggle_submission_kernel.py

push-kaggle: build-kaggle
	$(UV) run kaggle kernels push -p kaggle/birdclef_2026_gpu

push-kaggle-submit: build-kaggle-submit
	$(UV) run kaggle kernels push -p kaggle/birdclef_2026_submit_cpu
