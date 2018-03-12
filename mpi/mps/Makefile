# For OLCF Titan
CC=CC
TARGET=blas
OBJS=$(TARGET).o $(TARGET)cu.o

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) -lmpich -lcublas

$(TARGET).o: $(TARGET).cpp common.h
	$(CC) -c $<

$(TARGET)cu.o: $(TARGET).cu common.h
	nvcc -o $@ -c $<

test:
	aprun -n 4 ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
