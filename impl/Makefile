# Every figure and number regenerates from the raw records in a results/ run directory.
PY := .venv/bin/python

.PHONY: venv traces test analyze slack slack-report salt salt-report

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

# Week-3 slack simulation: full sweep into results/<date>_slack-sim/
slack:
	$(PY) -m sim.run_slack run

# Rebuild summary.csv from a run's per-request raw files
slack-report:
	@test -n "$(RUN)" || (echo "usage: make slack-report RUN=results/<date>_slack-sim" && exit 1)
	$(PY) -m sim.run_slack report $(RUN)

# Week-3 Mirage-vs-salting comparison: full sweep into results/<date>_salt-compare/
salt:
	$(PY) -m sim.run_salt run

salt-report:
	@test -n "$(RUN)" || (echo "usage: make salt-report RUN=results/<date>_salt-compare" && exit 1)
	$(PY) -m sim.run_salt report $(RUN)
