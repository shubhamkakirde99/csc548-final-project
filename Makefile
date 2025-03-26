
SERIAL_DEPS = serial.c include/util.c
FLAGS = -lpng -lm

IMAGE=spidey.png
RADIUS=50

compileserial: serial.c
	gcc -o serial $(SERIAL_DEPS) $(FLAGS)

serial: compileserial
	./serial ${RADIUS} ${IMAGE}

compilecuda:
	nvcc -x c include/util.c -x cu cuda.cu -o cuda $(FLAGS)
	
cuda: compilecuda
	./cuda ${RADIUS} ${IMAGE}



clean:
	rm -rf serial
	rm -rf out_serial.png
	rm -rf cuda
	rm -rf out_cuda.png

.PHONY: clean serial