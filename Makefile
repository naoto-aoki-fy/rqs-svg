CXX ?= g++
LDLIBS ?= -ldl
INCLUDE ?= -I./include
CXXFLAGS ?= -O2
QCS_CXXFLAGS = $(CXXFLAGS) -Wall -Wextra -Wformat=2 $(INCLUDE) -std=c++17
QCS_LDFLAGS = $(LDFLAGS) -L./lib $(LDLIBS)
QCS_BIN ?= bin/qcs
LIBQCS_SO ?= lib/libqcs.so

.PHONY: target
target: $(QCS_BIN)

.PHONY: sharedlibrary
sharedlibrary: $(LIBQCS_SO)

$(QCS_BIN): src/qcs_main.cpp src/qcs_args.c src/qcs_args.h include/qcs.h $(LIBQCS_SO)
	mkdir -p $(dir $@)
	$(CXX) $(QCS_CXXFLAGS) src/qcs_main.cpp src/qcs_args.c -lqcs $(QCS_LDFLAGS) -Wl,-rpath,$(shell realpath $(dir $(LIBQCS_SO))) -o $@

$(LIBQCS_SO): src/qcs_dummy.cpp include/qcs.h
	mkdir -p $(dir $@)
	$(CXX) $(QCS_CXXFLAGS) -fPIC -shared src/qcs_dummy.cpp $(QCS_LDFLAGS) -o $@

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: clean
clean:
	$(RM) $(QCS_BIN) $(LIBQCS_SO)
