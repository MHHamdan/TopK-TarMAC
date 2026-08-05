PYTHON ?= .venv/bin/python
export PYTHONPATH := $(CURDIR)

.PHONY: help test check check-bib check-configs check-log provenance \
        figures tables paper reproduce dry-run clean

help:
	@echo "test          run the pytest suite"
	@echo "check         all gates: tests + bib + config drift + master log"
	@echo "check-bib     fail if any .bib entry is uncited or any citation dangles"
	@echo "check-configs fail if a YAML disagrees with the runs made from it"
	@echo "check-log     fail if a completed run is absent from MASTER_LOG.csv"
	@echo "provenance    regenerate results/provenance.json"
	@echo "figures       regenerate paper/figures/*.pdf"
	@echo "tables        regenerate paper/tables/*"
	@echo "paper         compile paper/main.pdf"
	@echo "dry-run       print the reproduction plan without executing it"

test:
	$(PYTHON) -m pytest -q

check-bib:
	$(PYTHON) scripts/check_bib.py

check-configs:
	$(PYTHON) scripts/check_config_drift.py

check-log:
	$(PYTHON) scripts/rebuild_master_log.py --check

check: test check-bib check-configs check-log
	@echo "all gates passed"

provenance:
	$(PYTHON) scripts/collect_provenance.py

figures:
	$(PYTHON) scripts/make_fig1_schematic.py
	$(PYTHON) scripts/make_figures.py --n 6 --n-list 3 6 12

tables:
	$(PYTHON) scripts/make_tables.py --n-list 3 6 12

paper: figures tables
	cd paper && ( command -v tectonic >/dev/null 2>&1 \
	  && tectonic --keep-logs main.tex \
	  || ( pdflatex -interaction=nonstopmode main.tex && bibtex main; \
	       pdflatex -interaction=nonstopmode main.tex; \
	       pdflatex -interaction=nonstopmode main.tex ) )

reproduce:
	bash scripts/reproduce.sh

dry-run:
	bash scripts/reproduce.sh --dry-run

clean:
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.out paper/main.log
