-include config.mk

CXX ?= g++
CXXFLAGS = $(CFLAGS_VENDOR) -O3 -std=c++17 -Wall -Wextra -Wformat=2 -fopenmp
LDFLAGS += $(LDFLAGS_VENDOR)
LDLIBS += -fopenmp
QCS_BIN ?= bin/qcs
TEST_BIN ?= bin/test_hadamard
GENGETOPT ?= gengetopt

.PHONY: target test
target: $(QCS_BIN)

cmdline.c cmdline.h &: args.ggo
	$(GENGETOPT) --input=$< --file-name=cmdline --output-dir=.

$(QCS_BIN): main.cpp cmdline.c cmdline.h
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) main.cpp cmdline.c $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_BIN): tests/test_hadamard.cpp main.cpp cmdline.h
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) tests/test_hadamard.cpp $(LDFLAGS) $(LDLIBS) -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

.PHONY: clean
clean:
	$(RM) $(QCS_BIN) $(TEST_BIN)
