SHELL = /bin/sh
CC    = gcc
FLAGS        = # -std=gnu99 -Iinclude
CFLAGS       = -fPIC -g -shared -march=native -O2 # -pedantic -Wall -Wextra -march=native -ggdb3
LDFLAGS      = -shared

TARGET  = libcec2017C.so
SOURCES = cec2017.c cec17_test_func.c
# HEADERS = cec2017.h
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJECTS)
