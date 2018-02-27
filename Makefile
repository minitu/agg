TESTOPTS ?= ++local
CHARM_BASE ?= ../charm
CHARMC = $(CHARM_BASE)/bin/charmc $(OPTS)
OBJS = main.o

all: main

main: $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS)

main.o: main.C main.decl.h
	$(CHARMC) -c $<

main.decl.h: main.ci
	$(CHARMC) $<

clean:
	rm -f *.decl.h *.def.h $(OBJS) main charmrun

test: all
	./charmrun ./main +p1 $(TESTOPTS)
