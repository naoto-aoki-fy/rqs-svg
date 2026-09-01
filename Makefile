-include config.mk
CXX = mpicxx
INCLUDE ?= -I./include
CXXFLAGS ?= $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17
LDFLAGS ?= $(LDFLAGS_VENDOR) -L./lib
LDLIBS ?= -ldl
QCS_BIN ?= bin/qcs
LIBQCS_SO ?= lib/libqcs.so

.PHONY: target sharedlibrary clean gengetopt
target: $(QCS_BIN)
sharedlibrary: $(LIBQCS_SO)

$(QCS_BIN): src/qcs_main.cpp src/qcs_args.c src/qcs_args.h include/qcs.h $(LIBQCS_SO)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) src/qcs_main.cpp src/qcs_args.c -lqcs $(LDFLAGS) $(LDLIBS) -Wl,-rpath,$(shell realpath $(dir $(LIBQCS_SO))) -o $@

$(LIBQCS_SO): src/qcs.cpp include/qcs.h
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -fPIC -shared src/qcs.cpp $(LDLIBS) -o $@

gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

clean:
	$(RM) $(QCS_BIN) $(LIBQCS_SO)
