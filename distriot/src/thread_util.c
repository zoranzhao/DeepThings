#include "thread_util.h"
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>

#ifndef SYS_ARCH_TIMEOUT
#define SYS_ARCH_TIMEOUT 1000000
#endif

#ifndef ERR_OK
#define ERR_OK 0
#endif

#ifndef ERR_MEM
#define ERR_MEM -1
#endif
/*
   A retargetable multithread API implementation adopted from lwIP library.
   Current implementation utilizes pthread.
*/

struct sys_thread {
  struct sys_thread *next;
  pthread_t pthread;
};
static sys_thread_t threads = NULL;
static pthread_mutex_t threads_mutex = PTHREAD_MUTEX_INITIALIZER;
static sys_thread_t introduce_thread(pthread_t id)
{
  struct sys_thread *thread;
  thread = (struct sys_thread *)malloc(sizeof(struct sys_thread));
  if (thread != NULL) {
    pthread_mutex_lock(&threads_mutex);
    thread->next = threads;
    thread->pthread = id;
    threads = thread;
    pthread_mutex_unlock(&threads_mutex);
  }

  return thread;
}

sys_thread_t sys_thread_new(const char *name, thread_fn function, void *arg, int stacksize, int prio)
{
  int code;
  pthread_t tmp;
  struct sys_thread *st = NULL;
  /*name stacksize and prio is not used*/
  code = pthread_create(&tmp,
                        NULL, 
                        (void *(*)(void *)) 
                        function, 
                        arg);

  if (0 == code) {
    st = introduce_thread(tmp);
  }
  
  if (NULL == st) {
    printf("sys_thread_new: pthread_create %d, st = 0x%lx", code, ((unsigned long)st));
    abort();
  }
  return st;
}

void sys_thread_join(sys_thread_t thread){
    pthread_join(thread->pthread, NULL);
}

/*
   A retargetable Semaphore implementation adopted from lwIP library.
   Current implementation utilizes pthread.
*/
struct sys_sem {
  unsigned int c;
  pthread_condattr_t condattr;
  pthread_cond_t cond;
  pthread_mutex_t mutex;
};

static void
get_monotonic_time(struct timespec *ts)
{
  clock_gettime(CLOCK_MONOTONIC, ts);
}

static struct sys_sem * sys_sem_new_internal(uint8_t count)
{
  struct sys_sem *sem;
  sem = (struct sys_sem *)malloc(sizeof(struct sys_sem));
  if (sem != NULL) {
    sem->c = count;
    pthread_condattr_init(&(sem->condattr));
    pthread_cond_init(&(sem->cond), &(sem->condattr));
    pthread_mutex_init(&(sem->mutex), NULL);
  }
  return sem;
}

int8_t sys_sem_new(struct sys_sem **sem, uint8_t count)
{
  *sem = sys_sem_new_internal(count);
  if (*sem == NULL) {
    return ERR_MEM;
  }
  return ERR_OK;
}

static uint32_t cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex, uint32_t timeout)
{
  struct timespec rtime1, rtime2, ts;
  int ret;
  if (timeout == 0) {
    pthread_cond_wait(cond, mutex);
    return 0;
  }
  get_monotonic_time(&rtime1);
  ts.tv_sec = rtime1.tv_sec + timeout / 1000L;
  ts.tv_nsec = rtime1.tv_nsec + (timeout % 1000L) * 1000000L;
  if (ts.tv_nsec >= 1000000000L) {
    ts.tv_sec++;
    ts.tv_nsec -= 1000000000L;
  }
  ret = pthread_cond_timedwait(cond, mutex, &ts);
  if (ret == ETIMEDOUT) {
    return SYS_ARCH_TIMEOUT;
  }
  get_monotonic_time(&rtime2);
  ts.tv_sec = rtime2.tv_sec - rtime1.tv_sec;
  ts.tv_nsec = rtime2.tv_nsec - rtime1.tv_nsec;
  if (ts.tv_nsec < 0) {
    ts.tv_sec--;
    ts.tv_nsec += 1000000000L;
  }
  return (uint32_t)(ts.tv_sec * 1000L + ts.tv_nsec / 1000000L);
}

void
sys_sem_signal(struct sys_sem **s)
{
  struct sys_sem *sem;
  sem = *s;
  pthread_mutex_lock(&(sem->mutex));
  sem->c++;
  if (sem->c > 1) {
    sem->c = 1;
  }
  pthread_cond_broadcast(&(sem->cond));
  pthread_mutex_unlock(&(sem->mutex));
}

uint32_t
sys_arch_sem_wait(struct sys_sem **s, uint32_t timeout)
{
  uint32_t time_needed = 0;
  struct sys_sem *sem;
  sem = *s;
  pthread_mutex_lock(&(sem->mutex));
  while (sem->c <= 0) {
    if (timeout > 0) {
      time_needed = cond_wait(&(sem->cond), &(sem->mutex), timeout);
      if (time_needed == SYS_ARCH_TIMEOUT) {
        pthread_mutex_unlock(&(sem->mutex));
        return SYS_ARCH_TIMEOUT;
      }
    } else {
      cond_wait(&(sem->cond), &(sem->mutex), 0);
    }
  }
  sem->c--;
  pthread_mutex_unlock(&(sem->mutex));
  return (uint32_t)time_needed;
}

static void sys_sem_free_internal(struct sys_sem *sem)
{
  pthread_cond_destroy(&(sem->cond));
  pthread_condattr_destroy(&(sem->condattr));
  pthread_mutex_destroy(&(sem->mutex));
  free(sem);
}

void sys_sem_free(struct sys_sem **sem)
{
  if (sem != NULL) {
    sys_sem_free_internal(*sem);
  }
}

void sys_sleep(uint32_t milliseconds){
   struct timespec ts;
   ts.tv_sec = milliseconds / 1000;
   ts.tv_nsec = (milliseconds % 1000) * 1000000;
   nanosleep(&ts, NULL);
}

uint32_t sys_now(void)
{
  struct timespec ts;
  get_monotonic_time(&ts);
  return (uint32_t)(ts.tv_sec * 1000L + ts.tv_nsec / 1000000L);
}

double sys_now_in_sec(void)
{
  struct timespec ts;
  get_monotonic_time(&ts);
  return ts.tv_sec + ts.tv_nsec*1e-9;
}
