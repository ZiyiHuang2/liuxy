.PHONY: install test inspect smoke evaluate train evaluate-enhanced train-enhanced unknown-quick unknown-full unknown-replan-quick unknown-replan-full

install:
	python -m pip install -e ".[dev]"

test:
	pytest

inspect:
	voc-easy inspect --config configs/default.json

smoke:
	voc-easy smoke --config configs/default.json

evaluate:
	voc-easy evaluate --config configs/default.json --seeds 42 --output outputs/evaluation

train:
	voc-easy train --config configs/default.json --model artifacts/voc_easyensemble.joblib

evaluate-enhanced:
	voc-easy evaluate --config configs/feature_diverse.json --seeds 85001:85016,86001:86016 --output outputs/feature_diverse_32seed

train-enhanced:
	voc-easy train --config configs/feature_diverse.json --model artifacts/voc_feature_diverse.joblib

unknown-quick:
	python experiments/run_unknown_voc_comparison.py --quick --output results/unknown_voc_quick

unknown-full:
	python experiments/run_unknown_voc_comparison.py --output results/unknown_voc

unknown-replan-quick:
	python experiments/run_unknown_voc_replanned.py --quick --output results/unknown_voc_replanned_quick

unknown-replan-full:
	python experiments/run_unknown_voc_replanned.py --output results/unknown_voc_replanned
