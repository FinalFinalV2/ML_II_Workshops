.PHONY: help grade grade-all scaffold test test-00 test-01 test-02 test-solution-02 clean env

help:
	@echo "Available commands:"
	@echo "  make scaffold          - Generate student notebooks by stripping instructor solutions"
	@echo "  make test              - Run all autograding unit tests via pytest"
	@echo "  make test-00           - Run Workshop 00 unit tests"
	@echo "  make test-01           - Run Workshop 01 unit tests"
	@echo "  make test-02           - Run Workshop 02 unit tests"
	@echo "  make test-solution-02  - Run Workshop 02 instructor solution tests"
	@echo "  make clean             - Remove build artifacts, bytecode, and cache folders"

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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf build dist *.egg-info
