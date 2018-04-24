# For OLCF Titan
CFLAGS=-std=c++11
CC=CC $(CFLAGS) -Wall
NVCC=nvcc $(CFLAGS)
TARGET=blas
OBJS=$(TARGET).o $(TARGET)cu.o params.o

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) -lmpich -lcublas

$(TARGET).o: $(TARGET).cpp common.h params.h comp.h
	$(CC) -c $<

$(TARGET)cu.o: $(TARGET).cu common.h params.h comp.h
	$(NVCC) -o $@ -c $<

params.o: params.cpp params.h
	$(CC) -c $<

test:
	aprun -n 4 ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
