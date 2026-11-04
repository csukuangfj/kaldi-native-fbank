# Minimal Sphinx + Doxygen Makefile.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DOXYGEN       ?= doxygen

.PHONY: help html latexpdf clean doxygen

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

doxygen:
	@mkdir -p "$(BUILDDIR)/doxygen"
	@$(DOXYGEN) Doxyfile
	# Doxygen renders "name() : ns" entries in namespacemembers*.html with only
	# the "ns" label linked, so the function/type names themselves are not
	# clickable. Wrap each name in the same link as its "ns" label.
	@perl -pi -e 's|<li>([^<]+)&#160;:&#160;<a class="el" href="([^"]+)">|<li><a class="el" href="$$2">$$1</a>&#160;:&#160;<a class="el" href="$$2">|g' \
		"$(BUILDDIR)/doxygen/html"/namespacemembers*.html

html: doxygen
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	# Copy the doxygen HTML tree into the sphinx output. Remove any previous
	# copy first: plain `cp -r` would nest it as cpp-api-doxygen/html/ when
	# the target already exists, leaving stale files behind.
	rm -rf "$(BUILDDIR)/html/cpp-api-doxygen"
	cp -r "$(BUILDDIR)/doxygen/html" "$(BUILDDIR)/html/cpp-api-doxygen"

latexpdf: doxygen
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
