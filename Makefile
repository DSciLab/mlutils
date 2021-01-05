REQUIREMENTS   := requirements.txt
PIP            := pip
PYTHON         := python


.PHONY: all dep push install clean


all: dep push install


dep: $(REQUIREMENTS)
	$(PIP) install -r $<


commit: clean
	# Not Recommended
	git add -A
	-git commit -m 'Update project'
	git pull


push: commit
	git push


install:
	$(PYTHON) setup.py install


clean:
	-rm -rf .eggs .tox build MANIFEST
