SERIAL_DEPS = serial.c include/util.c
MPI_DEPS = mpi.c include/util.c
FLAGS = -lpng -lm

IMAGE=experiment1_1000.png
RADIUS=100
PROCS=16

compileserial: serial.c
	gcc -o serial $(SERIAL_DEPS) $(FLAGS)

serial: compileserial
	./serial ${RADIUS} ${IMAGE}

compilecuda:
	nvcc -x c include/util.c -x cu cuda.cu -o cuda $(FLAGS)
	
cuda: compilecuda
	./cuda ${RADIUS} ${IMAGE}

compilempi: mpi.c
	mpicc -o mpi $(MPI_DEPS) $(FLAGS)
	
mpi: compilempi
	mpirun -np $(PROCS) ./mpi ${RADIUS} ${IMAGE}

clean:
	rm -rf serial
	rm -rf out_serial.png
	rm -rf cuda
	rm -rf out_cuda.png
	rm -rf mpi
	rm -rf out_mpi.png

.PHONY: clean serial