CUDA_HOME    := /usr/local/cuda-12.6.3
CC_BIN       := /usr/bin/g++

LINKER       := /usr/bin/g++
CUDA_LIB     := /usr/local/cuda-12.6.3/targets/x86_64-linux/lib

# settings
DIALECT      = -std=c++17
#OPTIMIZATION = -O0 -g
OPTIMIZATION = -O3 -g
WARNINGS     = -Xcompiler="-Wall -Wextra"
# NVCC_FLAGS   = -arch=sm_61 -lineinfo --expt-relaxed-constexpr -rdc=true
NVCC_FLAGS   = -arch=sm_86 -lineinfo --expt-relaxed-constexpr -rdc=true --extended-lambda -lnvToolsExt -Xcompiler="-fopenmp" -I$(CUDA_HOME)/targets/x86_64-linux/include  -ccbin=$(CC_BIN) #-res-usage #-Xptxas "-v"
#NVCC_FLAGS   = -DCUDASW_DEBUG_CHECK_CORRECTNESS -arch=native -lineinfo --expt-relaxed-constexpr -rdc=true --extended-lambda -lnvToolsExt -Xcompiler="-fopenmp" #-res-usage #-Xptxas "-v"
LDFLAGS      = -Xcompiler="-pthread"  -lz
LDFLAGS      = -Xcompiler="-pthread"  $(NVCC_FLAGS) -lz
COMPILER     = nvcc
ARTIFACT     = align

BUILDDIR = build

MAKEDB = makedb
MODIFYDB = modifydb
GRIDSEARCH = gridsearch

$(shell mkdir -p $(BUILDDIR))

# make targets
.PHONY: clean

release: $(ARTIFACT) $(MAKEDB)


clean :
	rm -f $(BUILDDIR)/*
	rm -f $(ARTIFACT)
	rm -f $(MAKEDB)

# compiler call
COMPILE = $(COMPILER) $(NVCC_FLAGS) $(DIALECT) $(OPTIMIZATION) $(WARNINGS) -c $< -o $@


# link object files into executable
$(ARTIFACT): $(BUILDDIR)/main.o $(BUILDDIR)/sequence_io.o $(BUILDDIR)/dbdata.o $(BUILDDIR)/options.o $(BUILDDIR)/blosum.o $(BUILDDIR)/half2_kernel_instantiations.o $(BUILDDIR)/float_kernel_instantiations.o $(BUILDDIR)/dpx_s32_kernel_instantiations.o $(BUILDDIR)/dpx_s16_kernel_instantiations.o
	$(COMPILER) $^ -o $(ARTIFACT) $(LDFLAGS)

 $(MAKEDB): $(BUILDDIR)/makedb.o $(BUILDDIR)/sequence_io.o $(BUILDDIR)/dbdata.o
	$(LINKER) \
	  -std=c++17 $(OPTIMIZATION) \
	  -Wall -Wextra \
	  -fopenmp \
 	  -I$(CUDA_HOME)/targets/x86_64-linux/include \
 	  $^ \
 	  -L$(CUDA_LIB) -lcudart -lnvToolsExt -lz \
	  -lgomp \
 	  -o $(MAKEDB)



$(MODIFYDB): $(BUILDDIR)/modifydb.o $(BUILDDIR)/sequence_io.o $(BUILDDIR)/dbdata.o
	$(COMPILER) $^ -o $(MODIFYDB) $(LDFLAGS)

$(GRIDSEARCH): $(BUILDDIR)/gridsearch.o $(BUILDDIR)/sequence_io.o $(BUILDDIR)/dbdata.o $(BUILDDIR)/blosum.o $(BUILDDIR)/half2_kernel_instantiations.o $(BUILDDIR)/float_kernel_instantiations.o $(BUILDDIR)/dpx_s32_kernel_instantiations.o $(BUILDDIR)/dpx_s16_kernel_instantiations.o
	$(COMPILER) $^ -o $(GRIDSEARCH) $(LDFLAGS)


$(BUILDDIR)/main.o : src/main.cu src/sequence_io.h src/length_partitions.hpp src/dbdata.hpp src/cudasw4.cuh src/kernels.cuh src/convert.cuh src/float_kernels.cuh src/half2_kernels.cuh src/dpx_s16_kernels.cuh src/blosum.hpp src/types.hpp
	$(COMPILE)

$(BUILDDIR)/sequence_io.o : src/sequence_io.cpp src/sequence_io.h
	$(COMPILE)

$(BUILDDIR)/dbdata.o : src/dbdata.cpp src/dbdata.hpp src/mapped_file.hpp src/sequence_io.h src/length_partitions.hpp
	$(COMPILE)

$(BUILDDIR)/options.o : src/options.cpp src/options.hpp src/types.hpp
	$(COMPILE)

$(BUILDDIR)/blosum.o : src/blosum.cu src/blosum.hpp
	$(COMPILE)

$(BUILDDIR)/half2_kernel_instantiations.o: src/half2_kernel_instantiations.cu src/half2_kernels.cuh src/kernels.cuh
	$(COMPILE)

$(BUILDDIR)/float_kernel_instantiations.o: src/float_kernel_instantiations.cu src/float_kernels.cuh src/kernels.cuh
	$(COMPILE)

$(BUILDDIR)/dpx_s16_kernel_instantiations.o: src/dpx_s16_kernel_instantiations.cu src/dpx_s16_kernels.cuh src/kernels.cuh
	$(COMPILE)

$(BUILDDIR)/dpx_s32_kernel_instantiations.o: src/dpx_s32_kernel_instantiations.cu src/dpx_s32_kernels.cuh src/kernels.cuh
	$(COMPILE)

$(BUILDDIR)/makedb.o : src/makedb.cpp src/dbdata.hpp src/sequence_io.h
	$(COMPILE)

$(BUILDDIR)/modifydb.o : src/modifydb.cpp src/dbdata.hpp src/sequence_io.h
	$(COMPILE)

$(BUILDDIR)/gridsearch.o : src/gridsearch.cu src/length_partitions.hpp src/dbdata.hpp
	$(COMPILE)

