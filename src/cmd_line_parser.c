#include "cmd_line_parser.h"
/*Command line parsing implementation adapted from Darknet*/
/*def is served as default*/
int get_int_arg(int argc, char **argv, char *arg, int def)
{
   int i;
   for(i = 0; i < argc-1; ++i){
      if(!argv[i]) continue;
      if(0==strcmp(argv[i], arg)){
         def = atoi(argv[i+1]);
         break;
      }
   }
   return def;
}

float get_float_arg(int argc, char **argv, char *arg, float def)
{
   int i;
   for(i = 0; i < argc-1; ++i){
      if(!argv[i]) continue;
      if(0==strcmp(argv[i], arg)){
         def = atof(argv[i+1]);
         break;
      }
   }
   return def;
}

char *get_string_arg(int argc, char **argv, char *arg, char *def)
{
   int i;
   for(i = 0; i < argc-1; ++i){
      if(!argv[i]) continue;
      if(0==strcmp(argv[i], arg)){
         def = argv[i+1];
         break;
      }
   }
   return def;
}
