VPATH=./examples
OBJDIR=./
EXEC=top

CC=gcc
CXX=g++
AR=ar
ARFLAGS=rcs

LDFLAGS= -lm -pthread
CFLAGS=-Wall -fPIC
DARKIOT=..
COMMON=-I$(DARKIOT)/include/ -I$(DARKIOT)/src/  -Iinclude/ -Isrc/ 
LDLIB=-L$(DARKIOT) -l:libdistriot.a

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS+=-O0 -g
else
OPTS+=-Ofast
endif

ifeq ($(ARM_NEON), 1)
COMMON+= -DARM_NEON
CFLAGS+= -DARM_NEON -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize
endif

CFLAGS+=$(OPTS)
EXECOBJ = $(EXEC).o
DEPS = $(wildcard */*.h) Makefile

all: $(EXEC)

$(EXEC): $(EXECOBJ)
	$(MAKE) -C ../ clean
	$(MAKE) -C ../
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@  $(LDLIB) $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

test:
	./$(EXEC) ${ARGS}

.PHONY: clean

clean:
	rm -rf $(EXEC) $(EXECOBJ) *.log *.png *.dat out*.jpg
