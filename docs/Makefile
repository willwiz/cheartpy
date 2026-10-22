# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile api-blueprints

# New recipe to automate the blueprint generation step across all subfolders
api-blueprints:
	@echo "Generating API blueprints for cheartpy namespace components..."
	@rm -f $(SOURCEDIR)/cheartpy.*.rst
	@for dir in ../cheartpy-*/src/cheartpy/; do \
		if [ -d "$$dir" ]; then \
			echo "-> Scanning: $$dir"; \
			sphinx-apidoc --implicit-namespaces -o "$(SOURCEDIR)" "$$dir" "**/_*"; \
		fi \
	done

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
