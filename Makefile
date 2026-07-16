.PHONY: install test inspect smoke evaluate train

install:
	python -m pip install -e ".[dev]"

test:
	pytest

inspect:
	voc-easy inspect

smoke:
	voc-easy evaluate --seeds 42 --output outputs/smoke

evaluate:
	voc-easy evaluate --seeds 71001:71016 --output outputs/reliable_16seed

train:
	voc-easy train --model artifacts/voc_easyensemble.joblib
