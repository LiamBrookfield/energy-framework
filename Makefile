CC=gcc
CFLAGS=-O3 -Wall -march=native

SRC_DIR=src
BIN_DIR=bin

# List of source files
ALGO_SRCS= \
	$(SRC_DIR)/algorithms/insertion.c \
	$(SRC_DIR)/algorithms/merge.c \
	$(SRC_DIR)/algorithms/quick.c
UTIL_SRCS=$(SRC_DIR)/utils/rapl.c
MAIN_SRC=$(SRC_DIR)/benchmark.c

# Combine all sources
SRCS=$(ALGO_SRCS) $(UTIL_SRCS) $(MAIN_SRC)

# Output binary
TARGET=$(BIN_DIR)/benchmark

all: $(TARGET)

$(TARGET): $(SRCS)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -rf $(BIN_DIR)/*
