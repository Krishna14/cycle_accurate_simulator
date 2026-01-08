.PHONY: all build run clean

BUILD_DIR ?= build

all: build

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j

run: build
	./$(BUILD_DIR)/accel_sim

clean:
	rm -rf $(BUILD_DIR)


