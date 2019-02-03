#ifndef THREAD_UTIL_H
#define THREAD_UTIL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*thread_fn)(void *arg);
struct sys_thread;
typedef struct sys_thread* sys_thread_t;
/*multithreading APIs*/
sys_thread_t sys_thread_new(const char *name, thread_fn function, void *arg, int stacksize, int prio);
void sys_thread_join(sys_thread_t thread);

/*Semaphore APIs*/
struct sys_sem;
typedef struct sys_sem* sys_sem_t;
int8_t sys_sem_new(struct sys_sem **sem, uint8_t count);
void sys_sem_signal(struct sys_sem **s);
uint32_t sys_arch_sem_wait(struct sys_sem **s, uint32_t timeout);
void sys_sem_free(struct sys_sem **sem);
void sys_sleep(uint32_t milliseconds);
uint32_t sys_now(void);
double sys_now_in_sec(void);
#ifdef __cplusplus
}
#endif

#endif /*THREAD_UTIL_H*/
