PYTHON = python
MAIN_SCRIPT = main.py
DEFAULT_CONFIG = configs/config.yaml

.PHONY: help install run run-opt plot clean

help:
	@echo "Available commands:"
	@echo "  make run           - Run the main evolutionary pipeline with default config"
	@echo "  make run CONFIG=x  - Run the pipeline with a specific config file"
	@echo "  make run-opt       - Run the hyperparameter trial script (ax_trial.py)"
	@echo "  make plot          - Generate plots from the current results"
	@echo "  make clean         - Remove generated results, temporary files, and __pycache__"

run:
	$(PYTHON) $(MAIN_SCRIPT) --config $(CONFIG)

# Fallback if no CONFIG is provided
ifeq ($(CONFIG),)
CONFIG := $(DEFAULT_CONFIG)
endif

run-opt:
	$(PYTHON) ax_trial.py

plot:
	$(PYTHON) plot.py

clean:
	rm -rf results/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +
	@echo "Cleaned up results directory and Python caches."
