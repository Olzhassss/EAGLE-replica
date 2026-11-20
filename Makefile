PYTHON ?= EAGLE/.venv/bin/python
RYU_PYTHON ?= ryu/.venv/bin/python
ARGS ?=

.PHONY: help eagle controllers network test test-ryu

help:
	@echo "make eagle ARGS='train --help'   Run any EAGLE command"
	@echo "make controllers ARGS='--network networks/v0.json'"
	@echo "make network ARGS='--network networks/v0.json --cli'"
	@echo "make test                      Run EAGLE and Mininet unit tests"
	@echo "make test-ryu                  Run tests in the Ryu Python 3.7 environment"

eagle:
	$(PYTHON) -m EAGLE $(ARGS)

controllers:
	$(RYU_PYTHON) -m ryu $(ARGS)

network:
	sudo python3 -m mininet $(ARGS)

test:
	$(PYTHON) -m unittest discover -s EAGLE/tests
	$(PYTHON) -m unittest discover -s mininet/tests

test-ryu:
	$(RYU_PYTHON) -m unittest discover -s ryu/tests
