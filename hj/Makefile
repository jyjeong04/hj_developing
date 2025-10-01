ifndef CPPC
	CPPC=g++
endif

CCFLAGS=-O2 -ffast-math

LIBS = -lm -lOpenCL -fopenmp

COMMON_DIR = Cpp_common

INC = -I $(COMMON_DIR)

EXEC = hj

all: $(EXEC)

$(EXEC): hj.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $(EXEC)

clean:
	rm -f $(EXEC)