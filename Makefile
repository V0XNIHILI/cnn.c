# Usage:
# Run 'make compile' to compile the example program
# Run 'make run' to execute the example program.
# Run 'make speed' to compile the example program with speed optimization.
# Run 'make gprof' to compile the example program with gprof analysis.
# Run 'make valgrind' to run the example program in Valgrind.
# Run 'make clean' to remove compiled files.

CC = gcc
CPPFLAGS = -g -Wall -Werror -Wextra -Wno-unused-parameter -Wshadow -Wdouble-promotion -Wformat=2 -Wno-unused-variable -Wno-unused-result -fno-common -Wconversion  -Wno-missing-field-initializers -Werror=implicit-function-declaration -pedantic -fopenmp -lm
OBJS = src/example.c src/tensor.c src/nn.c
TARGET = example.o

CPPFLAGS_FOR_SPEED = -flto -O3 -fomit-frame-pointer -march=native -lm
CPPFLAGS_FOR_GPROF = -pg -lm -fno-inline

.PHONY: compile speed gprof run valgrind clean 

default: run

compile : $(OBJS)
	$(CC) $(OBJS) $(CPPFLAGS) -o $(TARGET)

speed: $(OBJS)
	$(CC) $(OBJS) $(CPPFLAGS_FOR_SPEED) -o $(TARGET)

gprof: $(OBJS)
	$(CC) $(OBJS) $(CPPFLAGS_FOR_GPROF) -o $(TARGET)
	./example.o
	gprof ./example.o gmon.out > analysis.txt

run: compile
	./example.o

valgrind: compile
	valgrind  --tool=memcheck -s --leak-check=full --show-leak-kinds=all ./example.o

clean:
	-rm -f *.o
	-rm -f $(TARGET)
	-rm -f *.txt
	-rm -f *.out
