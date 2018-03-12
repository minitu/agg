# For OLCF Titan
CC=cc
TARGET=simple

$(TARGET): $(TARGET).c
	$(CC) -o $@ $< -lmpich

test:
	aprun -n 2 ./$(TARGET)

clean:
	rm -f $(TARGET)
