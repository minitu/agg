# For OLCF Titan
CFLAGS=-std=c++11
CC=CC $(CFLAGS)
NVCC=nvcc $(CFLAGS)
TARGET=blas
OBJS=$(TARGET).o $(TARGET)cu.o

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) -lmpich -lcublas

$(TARGET).o: $(TARGET).cpp common.h params.h
	$(CC) -c $<

$(TARGET)cu.o: $(TARGET).cu common.h params.h
	$(NVCC) -o $@ -c $<

test:
	aprun -n 4 ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
