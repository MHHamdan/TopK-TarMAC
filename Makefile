PYTHON ?= .venv/bin/python
export PYTHONPATH := $(CURDIR)
PAPER := paper_submission_revision

.PHONY: help test check check-bib check-numbers figures tables paper \
        equivalence fixed-set supplement clean

help:
	@echo "test           run the pytest suite"
	@echo "check          tests + bibliography + every number in the manuscript"
	@echo "check-bib      fail if any .bib entry is uncited or any citation dangles"
	@echo "check-numbers  regenerate tables, re-derive prose numbers, verify the"
	@echo "               main-text number manifest (scripts/paper_numbers.py)"
	@echo "tables         regenerate $(PAPER)/tables/*.tex from results/ (and"
	@echo "               results/portability/ for the other-GPU appendix)"
	@echo "figures        regenerate $(PAPER)/figures/*.pdf from results/"
	@echo "paper          tables + figures + compile $(PAPER)/main.pdf"
	@echo "equivalence    re-check every timed configuration against the reference (GPU)"
	@echo "fixed-set      re-run the fixed-retained-set sweep (GPU, ~2 min)"
	@echo "supplement     build the anonymised supplementary archive"

test:
	$(PYTHON) -m pytest -q

check-bib:
	$(PYTHON) scripts/check_bib.py

check-numbers:
	$(PYTHON) scripts/check_revision_numbers.py --verify-paper-numbers

check: test check-bib check-numbers
	@echo "all gates passed"

tables:
	$(PYTHON) scripts/make_revision_tables.py
	$(PYTHON) scripts/summarise_portability.py

figures:
	$(PYTHON) scripts/make_revision_figures.py

paper: tables figures
	cd $(PAPER) && tectonic --keep-logs main.tex

equivalence:
	$(PYTHON) scripts/verify_all_configurations.py --device cuda:0

fixed-set:
	$(PYTHON) scripts/bench_fixed_set.py --device cuda:0

supplement:
	bash scripts/make_supplement.sh

clean:
	rm -f $(PAPER)/main.aux $(PAPER)/main.bbl $(PAPER)/main.blg \
	      $(PAPER)/main.out $(PAPER)/main.log
