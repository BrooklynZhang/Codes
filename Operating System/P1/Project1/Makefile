dash : dash.o modes.o mode_funcs.o
	gcc -o dash dash.o modes.o mode_funcs.o

dash.o : dash.c dash.h
	gcc -c dash.c

modes.o : modes.c dash.h
	gcc -c modes.c

mode_funcs.o : mode_funcs.c dash.h
	gcc -c mode_funcs.c

clean :
	rm dash dash.o modes.o mode_funcs.o

