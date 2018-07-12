#ifndef CMD_LINE_PARSER_H
#define CMD_LINE_PARSER_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

int get_int_arg(int argc, char **argv, char *arg, int def);
float get_float_arg(int argc, char **argv, char *arg, float def);
char *get_string_arg(int argc, char **argv, char *arg, char *def);

#endif
