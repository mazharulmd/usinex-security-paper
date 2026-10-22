# Every figure and number regenerates from the raw records in a results/ run directory.
PY := .venv/bin/python

.PHONY: venv traces test analyze

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

traces:
	scripts/fetch_traces.sh

test:
	.venv/bin/pytest -q

# make analyze RUN=results/2026-11-02_tier-ladder
analyze:
	@test -n "$(RUN)" || (echo "usage: make analyze RUN=results/<date>_<experiment>" && exit 1)
	$(PY) -m harness.analyze $(RUN)
