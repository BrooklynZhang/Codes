#ifndef DASH_H
#define DASH_H


#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>

#define bufsize 512


void interactive_mode();
void batch_mode();
void print_error_msg();
void check_current_process_normal(char** line);
void check_current_process_redirect(char** line, char *filename);
char** separate_line(char* line, char* deli);

#endif
