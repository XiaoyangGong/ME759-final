# the compiler: gcc for C program, define as g++ for C++
CC = g++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#  `pkg-config opencv --cflags --libs` pkgconfig flags
CFLAGS  = `pkg-config opencv --cflags --libs` -g -Wall

# the build target executable:
TARGET = stitch

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) -o $(TARGET) $(TARGET).cpp $(CFLAGS)

clean:
	$(RM) $(TARGET)
