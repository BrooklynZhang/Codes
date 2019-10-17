#include "types.h"
#include "x86.h"
#include "defs.h"
#include "param.h"
#include "mmu.h"
#include "proc.h"
#include "sysfunc.h"
#include "pstat.h"

int
sys_fork(void)
{
  return fork();
}

int
sys_exit(void)
{
  exit();
  return 0;  // not reached
}

int
sys_wait(void)
{
  return wait();
}

int
sys_kill(void)
{
  int pid;

  if(argint(0, &pid) < 0)
    return -1;
  return kill(pid);
}

int
sys_getpid(void)
{
  return proc->pid;
}

int
sys_sbrk(void)
{
  int addr;
  int n;

  if(argint(0, &n) < 0)
    return -1;
  addr = proc->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

int
sys_sleep(void)
{
  int n;
  uint ticks0;
  
  if(argint(0, &n) < 0)
    return -1;
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(proc->killed){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

// return how many clock tick interrupts have occurred
// since boot.
int
sys_uptime(void)
{
  uint xticks;
  
  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

/* The following code is added/modified by your Ningjie Zhang nxz190006 
kernel fun for set tickets, it will check the ticknum is avaliable or if it is <=0
and all the fun in proc.c
*/
int
sys_settickets(void)
{
  int ticket_num;
  if (argint(0, &ticket_num) < 0)
  {
    return -1;
  }
  else
  {
    if (ticket_num <= 0)
    {
      return -1;
    }
    setticket(ticket_num);
  }
  return 0;
}
/* End of code added/modified */ 
/* The following code is added/modified by your Ningjie Zhang nxz190006 
it will call the getpinfo in proc.c to get the infos
*/
int
sys_getpinfo(void)
{
  struct pstat *cur_pstat;
  if(argptr(0, (void*)&cur_pstat, sizeof(struct pstat *)) < 0) 
  {
    return -1;
  }
  if(cur_pstat == NULL) 
  {
    return -1;
  }
  
  //for(struct proc* p=ptable.proc;p != &(ptable.proc[NPROC]); p++)
  //{
  //  const int index = p - ptable.proc;

  getpinfo(cur_pstat);
  return 0;
}
/* End of code added/modified */ 

