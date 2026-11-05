.PHONY: help grade grade-all scaffold test test-00 test-01 test-02 test-solution-02 test-03 test-solution-03 clean env install-core install-classical install-tabular install-dl install-prophet install-timestral install-all

help:
	@echo "Available commands:"
	@echo "  --- Installation & Environments ---"
	@echo "  make install-core       - Install base PyData + Autograder stack"
	@echo "  make install-classical  - Install W01-W08 classical time series stack (statsmodels)"
	@echo "  make install-tabular    - Install W00 revision stack (sklearn, xgboost, umap)"
	@echo "  make install-dl         - Install W10-W11 deep learning stack (pytorch, lightning)"
	@echo "  make install-prophet    - Install W09 additive models stack (prophet, arch)"
	@echo "  make install-timestral  - Install W99 foundation model stack (timestral, hf)"
	@echo "  make install-all        - Install complete master bundle (CI & instructor)"
	@echo "  --- Autograding & Testing ---"
	@echo "  make test               - Run all autograding unit tests via pytest"
	@echo "  make test-00            - Run Workshop 00 unit tests"
	@echo "  make test-01            - Run Workshop 01 unit tests"
	@echo "  make test-02            - Run Workshop 02 unit tests"
	@echo "  make test-solution-02   - Run Workshop 02 instructor solution tests"
	@echo "  --- Maintenance ---"
	@echo "  make scaffold           - Generate student notebooks by stripping solutions"
	@echo "  make clean              - Remove build artifacts, bytecode, and cache folders"

# Installation targets
install-core:
	pip install -e .

install-classical:
	pip install -e ".[classical]"

install-tabular:
	pip install -e ".[tabular]"

install-dl:
	pip install -e ".[deeplearning]"

install-prophet:
	pip install -e ".[prophet]"

install-timestral:
	pip install -e ".[timestral]"

install-all:
	pip install -e ".[all]"

grade:
	python3 scripts/update_grades_table.py

grade-all:
	python3 scripts/update_grades_table.py --all

scaffold:
	python3 scripts/generate_scaffold.py

test:
	pytest workshop_*/tests/

test-00:
	pytest workshop_00_revising/tests/

test-01:
	pytest workshop_01_ts_graphics/tests/

test-02:
	pytest workshop_02_decomposition/tests/

test-solution-02:
	AUTOGRADE_TARGET=instructor pytest workshop_02_decomposition/tests/

test-03:
	pytest workshop_03_benchmarks/tests/

test-solution-03:
	AUTOGRADE_TARGET=instructor pytest workshop_03_benchmarks/tests/

test-04:
	pytest workshop_04_evaluation/tests/

test-solution-04:
	AUTOGRADE_TARGET=instructor pytest workshop_04_evaluation/tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf build dist *.egg-info


test-05:
	pytest workshop_05_regression/tests/
