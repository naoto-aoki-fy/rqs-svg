-include config.mk

CXX ?= g++
CXXFLAGS ?= -O3
CXXFLAGS += $(CFLAGS_VENDOR) -Wall -Wextra -Wformat=2 -std=c++17 -fopenmp
LDFLAGS += $(LDFLAGS_VENDOR)
LDLIBS += -fopenmp
QCS_BIN ?= bin/qcs
GENGETOPT ?= gengetopt

.PHONY: target
target: $(QCS_BIN)

cmdline.c cmdline.h &: args.ggo
	$(GENGETOPT) --input=$< --file-name=cmdline --output-dir=.

$(QCS_BIN): main.cpp cmdline.c cmdline.h
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) main.cpp cmdline.c $(LDFLAGS) $(LDLIBS) -o $@

.PHONY: clean
clean:
	$(RM) $(QCS_BIN)
