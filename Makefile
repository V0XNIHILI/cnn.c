# Usage:
# Run 'make compile' to compile the example program
# Run 'make run' to execute the example program.
# Run 'make valgrind' to run the example program in Valgrind.
# Run 'make clean' to remove compiled files.

CC = gcc
CPPFLAGS = -g -Wall -Werror -Wextra -Wno-unused-parameter -Wshadow -Wdouble-promotion -Wformat=2 -Wno-unused-variable -Wno-unused-result -fno-common -Wconversion  -Wno-missing-field-initializers -Werror=implicit-function-declaration -pedantic
OBJS = src/example.c src/tensor.c src/nn.c
TARGET = example.o

.PHONY: compile run valgrind clean

default: run

compile : $(OBJS)
	$(CC) $(OBJS) $(CPPFLAGS) -o $(TARGET)

run: compile
	./example.o

valgrind: compile
	valgrind  --tool=memcheck -s --leak-check=full --show-leak-kinds=all ./example.o

clean:
	-rm -f *.o
	-rm -f $(TARGET)
