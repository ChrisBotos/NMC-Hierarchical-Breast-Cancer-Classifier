# Makefile for CATS Group 9 paper.
# Usage:
#   make        - build the PDF
#   make clean  - remove build artifacts
#   make watch  - continuous rebuild (requires latexmk)

MAIN = main
TEX  = pdflatex
BIB  = bibtex

.PHONY: all clean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz \
	      $(MAIN).pdf

watch:
	latexmk -pdf -pvc $(MAIN)
