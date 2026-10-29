.PHONY: install test experiments notebook slides all
install:
	python -m pip install -e '.[lab]'
test:
	python -m pytest -q
experiments:
	python scripts/run_experiments.py
notebook:
	python scripts/build_notebook.py
slides:
	cd slides && xelatex -no-pdf -interaction=nonstopmode -halt-on-error presentation.tex
	cd slides && xelatex -no-pdf -interaction=nonstopmode -halt-on-error presentation.tex
	cd slides && xdvipdfmx -q -o presentation.pdf presentation.xdv
all: test experiments notebook slides
