# # Compilers
# CXX := g++
# # NVCC := nvcc
# # # NVCC := /usr/local/cuda-12.6.3/bin/nvcc

# # Compiler flags
# CXXFLAGS := -std=c++17 -O2 -I.
# NVCCFLAGS := -O2 -Xcompiler -fPIC
# # CXXFLAGS := -std=c++17 -O2 -I. -I/usr/local/cuda-12.6.3/include
# # NVCCFLAGS := -O2 -Xcompiler -fPIC -I/usr/local/cuda-12.6.3/include

# # Linker flag
# # LDFLAGS := -L/usr/local/cuda-12.6.3/lib64 -lcudart

# # Executable name
# TARGET := test_runner

# # Source files
# CPP_SRCS := TestFile.cpp main.cpp lazySmith.cpp lazySmith_parallel_threads.cpp # lazySmith_parallel_futures.cpp
# # CU_SRCS := simpleGPU.cu

# # Object files
# CPP_OBJS := $(SRCS:.cpp=.o)
# CU_OBJS := $(CU_SRCS:.cu=.o)
# OBJS := $(CPP_OBJS) $(CU_OBJS)

# # Default target
# all: $(TARGET)

# # Link the executable
# $(TARGET): $(OBJS)
# 	$(CXX) $(CXXFLAGS) -o $@ $^

# # Compile source files into objects
# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@
# %.o: %.cu
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# # Clean rule
# clean:
# 	rm -f $(OBJS) $(TARGET)

# .PHONY: all clean

# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -std=c++17 -O2 -I.

# Executable name
TARGET := test_runner

# Source files
SRCS := TestFile.cpp main.cpp lazySmith.cpp lazySmith_parallel_threads.cpp lazySmith_parallel_futures.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files into objects
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean 