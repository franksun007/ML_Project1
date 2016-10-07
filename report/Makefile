PAPER = report
TEX = $(wildcard *.tex)
BIB = literature.bib
FIGS = $(wildcard figures/*.pdf figures/*.png graphs/*.pdf graphs/*.png)
TEXT = $(wildcard text/*.tex) 
.PHONY: all clean plots

$(PAPER).pdf: $(TEX) $(BIB) $(FIGS) IEEEtran.cls $(TEXT)
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf

