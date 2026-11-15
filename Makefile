NVCC ?= nvcc
NVCCFLAGS ?= -O3

.PHONY: all clean

all: final_project

final_project: final_project.cu
	$(NVCC) $(NVCCFLAGS) final_project.cu -o final_project

clean:
	rm -f final_project final_project.exe
