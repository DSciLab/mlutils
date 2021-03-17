REQUIREMENTS   := requirements.txt
PIP            := pip
PYTHON         := python


.PHONY: all dep install clean


all: dep push install


dep: $(REQUIREMENTS)
	$(PIP) install -r $<


install: dep
	$(PYTHON) setup.py install


clean:
	-rm -rf .eggs .tox build MANIFEST
