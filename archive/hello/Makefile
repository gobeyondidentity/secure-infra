# DOCA Hello World Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS =

# DOCA libraries and includes
DOCA_LIBS = -ldoca_common
DOCA_CFLAGS = $(shell pkg-config --cflags doca 2>/dev/null || echo "-I/opt/mellanox/doca/include")
DOCA_LDFLAGS = $(shell pkg-config --libs doca 2>/dev/null || echo "-L/opt/mellanox/doca/lib/aarch64-linux-gnu")

TARGET = hello_doca
SOURCES = hello_doca.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(DOCA_CFLAGS) -o $@ $< $(LDFLAGS) $(DOCA_LDFLAGS) $(DOCA_LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	sudo ./$(TARGET)

.PHONY: all clean run
