
SERIAL_DEPS = serial.c include/util.c
FLAGS = -lpng -lm


compileserial: serial.c
	gcc -o serial $(SERIAL_DEPS) $(FLAGS)

serial: compileserial
	./serial 5



clean:
	rm serial
	rm out.png

.PHONY: clean serial