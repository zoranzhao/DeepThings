OPENMP=1
VPATH=./src
DARKIOT=../DarkIoT
DARKNET=../nnpack_darknet/darknet-nnpack
OBJDIR=./obj/
EXEC=deep
DARKNETLIB=libdarknet.a
DARKIOTLIB=libdarkiot.a

CC=gcc
LDFLAGS= -lm -pthread
CFLAGS=-Wall -fPIC
COMMON=-I$(DARKIOT)/include/ -I$(DARKIOT)/src/ -I$(DARKNET)/include/ -I$(DARKNET)/src/ -Iinclude/ -Isrc/ 
LDLIB=-L$(DARKIOT) -l:$(DARKIOTLIB) -L$(DARKNET) -l:$(DARKNETLIB)

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
OBJS = configure.o top.o ftp.o inference_engine_helper.o
EXECOBJ = $(addprefix $(OBJDIR), $(OBJS))
DEPS = $(wildcard */*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(EXECOBJ) $(DARKNETLIB) $(DARKIOTLIB)
	$(CC) $(COMMON) $(CFLAGS) $(EXECOBJ) -o $@  $(LDLIB) $(LDFLAGS)

$(DARKNETLIB):
	$(MAKE) -C $(DARKNET)

$(DARKIOTLIB):
	$(MAKE) -C $(DARKIOT)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

test:
	./deep ${ARGS}

.PHONY: clean

clean:
	rm -rf $(EXEC) $(EXECOBJ) *.log $(OBJDIR) 
