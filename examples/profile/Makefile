OPENMP=0
NNPACK=1
ARM_NEON=1
DEBUG=0
DEEPTHINGS=../..
DARKNET=../../darknet-nnpack
DISTRIOT=../../distriot

VPATH=$(DEEPTHINGS)/src:./
OBJDIR=./obj/
EXEC=deepthings
DARKNETLIB=libdarknet.a
DISTRIOTLIB=libdistriot.a
SLIB=lib$(EXEC).so
ALIB=lib$(EXEC).a

CC=gcc
AR=ar
ARFLAGS=rcs
LDFLAGS= -lm -pthread
CFLAGS=-Wall -fPIC
COMMON=-I$(DEEPTHINGS)/src/ -I$(DISTRIOT)/include/ -I$(DISTRIOT)/src/ -I$(DARKNET)/include/ -I$(DARKNET)/src/ -I. 
LDLIB=-L$(DISTRIOT) -l:$(DISTRIOTLIB) -L$(DARKNET) -l:$(DARKNETLIB)

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
OBJS = test.o test_utils.o deepthings_profile.o deepthings_annotation.o ftp.o inference_engine_helper.o frame_partitioner.o adjacent_reuse_data_serialization.o self_reuse_data_serialization.o deepthings_edge.o deepthings_gateway.o cmd_line_parser.o
EXECOBJ = $(addprefix $(OBJDIR), $(OBJS))

DEPS = $(wildcard */*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(EXECOBJ) $(DARKNETLIB) $(DISTRIOTLIB)
	$(CC) $(COMMON) $(CFLAGS) $(EXECOBJ) -o $@  $(LDLIB) $(LDFLAGS)

$(DARKNETLIB):
	$(MAKE) -C $(DARKNET) NNPACK=$(NNPACK) ARM_NEON=$(ARM_NEON) DEBUG=$(DEBUG) OPENMP=$(OPENMP) all

$(DISTRIOTLIB):
	$(MAKE) -C $(DISTRIOT) DEBUG=$(DEBUG) all

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

test:
	./deepthings ${ARGS}

.PHONY: clean clean_all

clean_all:
	make -C $(DISTRIOT) clean
	make -C $(DARKNET) clean
	rm -rf $(EXEC) $(EXECOBJ) *.log $(OBJDIR) *.png *.prof 

clean:
	rm -rf $(EXEC) $(EXECOBJ) *.log $(OBJDIR) *.png *.prof
