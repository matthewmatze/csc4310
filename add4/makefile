#
# make FN=<filename without the .cu>
# make clean FN=<filename without the .cu
#
#
#
FN=vec_add
NVCC=nvcc

NVCCFLAGS=-arch sm_35

$(FN): $(FN).cu
	$(NVCC) $(FN).cu $(NVCCFLAGS) -o $(FN)
clean:
	rm *.o *~ $(FN)
