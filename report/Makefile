LATEX_OPTS="-synctex=1 -interaction=nonstopmode"

MAIN_NAME=report

all: report clean

.PHONY: report
report:
	pdflatex $(LATEX_OPTS) $(MAIN_NAME).tex

.PHONY: clean
clean:
	rm -f *.aux \
              *.log \
              *.out \
              *.synctex.* \
              *.fls \
              *_latexmk \
              *-blx.bib \
              *.bbl \
              *.blg \
              *.run.xml \
              *.vrb \
              *.toc \
              *.snm \
              *.nav