OPENMP=1
NNPACK=0
ARM_NEON=0
DEBUG=1

VPATH=./src
DARKIOT=../DarkIoT
DARKNET=../nnpack_darknet/darknet-nnpack
OBJDIR=./obj/
EXEC=deepthings
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

ifeq ($(NNPACK), 1)
COMMON+= -DNNPACK
CFLAGS+= -DNNPACK
LDFLAGS+= -lnnpack -lpthreadpool
endif

ifeq ($(ARM_NEON), 1)
COMMON+= -DARM_NEON
CFLAGS+= -DARM_NEON -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize
endif

CFLAGS+=$(OPTS)
OBJS = configure.o top.o ftp.o inference_engine_helper.o frame_partitioner.o reuse_data_serialization.o
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
	./deepthings ${ARGS}

.PHONY: clean

clean:
	rm -rf $(EXEC) $(EXECOBJ) *.log $(OBJDIR) *.png 
