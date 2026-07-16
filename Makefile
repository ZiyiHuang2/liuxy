.PHONY: install test inspect smoke evaluate train evaluate-enhanced train-enhanced

install:
	python -m pip install -e ".[dev]"

test:
	pytest

inspect:
	voc-easy inspect

smoke:
	voc-easy evaluate --seeds 42 --output outputs/smoke

evaluate:
	voc-easy evaluate --config configs/default.json --seeds 71001:71016 --output outputs/fixed_16seed

train:
	voc-easy train --config configs/default.json --model artifacts/voc_easyensemble.joblib

evaluate-enhanced:
	voc-easy evaluate --config configs/feature_diverse.json --seeds 85001:85016,86001:86016 --output outputs/feature_diverse_32seed

train-enhanced:
	voc-easy train --config configs/feature_diverse.json --model artifacts/voc_feature_diverse.joblib
