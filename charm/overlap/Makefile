TESTOPTS ?= ++local
CHARM_BASE ?= ../charm
CHARMC = $(CHARM_BASE)/bin/charmc $(OPTS)
OBJS = main.o

CHARM_INC = -I$(CHARM_BASE)/include
NVCC_FLAGS = -std=c++11
NVCC = nvcc $(CHARM_INC) $(NVCC_FLAGS)

# include optimization

all: main

main: $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS)

main.o: main.cu main.decl.h
	$(NVCC) -c $<

main.decl.h: main.ci
	$(CHARMC) $<

clean:
	rm -f *.decl.h *.def.h $(OBJS) main charmrun

test: all
	./charmrun ./main +p1 $(TESTOPTS)
