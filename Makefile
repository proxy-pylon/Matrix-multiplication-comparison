CXX := g++
CXXFLAGS := -O3 -march=native -DNDEBUG
SRC := src/mm_bench.cpp
BIN := mm_bench

# Default: OpenBLAS
LIBS ?= -lopenblas

# To use generic BLAS instead:
# make LIBS="-lcblas -lblas"

all: $(BIN)
$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f $(BIN)

