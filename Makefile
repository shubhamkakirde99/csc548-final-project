
SERIAL_DEPS = serial.c include/util.c
FLAGS = -lpng -lm

serial: runserial

compileserial: serial.c
	gcc -o serial $(SERIAL_DEPS) $(FLAGS)

runserial: compileserial
	./serial 5



clean:
	rm serial
	rm out.png
