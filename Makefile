CC=nvcc
SRC=$(wildcard src/*.cu src/*.cpp)
LDLIBS=-lsfml-window -lsfml-system -lsfml-graphics -lGL
CPPFLAGS=-std=c++11 -O3 -g -dc

all: objects
	$(CC) -o main *.o $(LDLIBS)

objects: $(SRC)
	$(CC) -Isrc/include/ -Isrc/include/imgui/ $(SRC) $(CPPFLAGS)

clean:
	rm -f main
	rm -f *.o