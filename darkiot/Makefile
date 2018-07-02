VPATH=./src
SLIB=libdarkiot.so
ALIB=libdarkiot.a
OBJDIR=./obj/

CC=gcc
CXX=g++
AR=ar
ARFLAGS=rcs
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -fPIC

ifeq ($(DEBUG), 1) 
OPTS+=-O0 -g
else
OPTS+=-Ofast
endif

CFLAGS+=$(OPTS)
OBJ = thread_safe_queue.o thread_util.o data_blob.o network_util.o exec_ctrl.o config.o gateway.o client.o global_queues.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard */*.h) Makefile

all: obj $(SLIB) $(ALIB) 

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) obj *.log
