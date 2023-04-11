MAKE_HOME:=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: build check

build:
ifeq ($(VIRTUAL_ENV), )
	@echo "virtual env is not activated, not advised. "
else
	python -m pip install $(MAKE_HOME)
endif


check:
ifeq ($(VIRTUAL_ENV), )
	@echo "virtual env is not activated"
else
	@echo "virtual env is activated"
endif