# Thanks to Job Vranish (https://spin.atomicobject.com/2016/08/26/makefile-c-projects/)
TARGET_EXEC := nn.out

CXX := g++

COMPILE_INF := -Wall -Wextra -fopt-info

CXXFLAGS := -O3 -fopenmp -march=native -std=c++17

DRIVER := main.cpp
BUILD_DIR := ./build
SRC_DIRS := ./src
HEADER_DIRS := ./lib

# Find all the C++ files we want to compile
SRCS := $(shell find $(SRC_DIRS)/*.cpp $(DRIVER))

# String substitution for every C++ file.
# As an example, hello.cpp turns into ./build/hello.cpp.o
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
# As an example, ./build/hello.cpp.o turns into ./build/hello.cpp.d
DEPS := $(OBJS:.o=.d)

# Every folder in ./src will need to be passed to G++ so that it can find header files
INC_DIRS := $(shell find $(HEADER_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. G++ understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS := $(INC_FLAGS) -MMD -MP

# The final build step.
# To turn on warnings and optimization information, add $(COMPILE_INF)
$(TARGET_EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C++ source
# To turn on warnings and optimization information, add $(COMPILE_INF)
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	rm -r $(BUILD_DIR)

run:
	.$(TARGET_EXEC) -i 784 -h 100 -o 10

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)