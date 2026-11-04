PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
NOTEBOOKS := $(sort $(wildcard notebooks/[0-9]*.ipynb))

.PHONY: all setup data lint test notebooks clean

## Full run: install, download data, lint, run tests, execute every notebook in order
all: setup data lint test notebooks

setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -r requirements.txt
	touch $@

data: setup
	$(BIN)/python scripts/download_data.py

lint: setup
	$(BIN)/ruff check .

test: setup
	$(BIN)/pytest -q

notebooks: setup data
	@for nb in $(NOTEBOOKS); do \
		echo "executing $$nb"; \
		$(BIN)/jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=1800 "$$nb" || exit 1; \
	done

clean:
	rm -rf $(VENV) data/raw/*.csv .pytest_cache
