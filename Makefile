.PHONY: install test bench clean

install:
	pip install -e .

test:
	python -m pytest tests/ -v

bench:
	python benchmarks/run.py

chart:
	python benchmarks/plot.py

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
