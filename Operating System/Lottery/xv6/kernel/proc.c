#include "types.h"
#include "defs.h"
#include "param.h"
#include "mmu.h"
#include "x86.h"
#include "proc.h"
#include "spinlock.h"
//#include <stdlib.h>
#include "pstat.h"
#define GET_TICKETS_SWITCH

/* The following code is added/modified by Ningjie Zhang and nxz190005
this is a rand number generateor, A is any 32 bit numbers with out 0
it use ^ to decide each bit is 1 or 0 to get a random number
*/

unsigned int A = 1324234;

unsigned int rand()
{
  unsigned int x = A;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return A = x;

}

/* End of code added/modified */ 

int total_tickets;
void saveticketstoproccess(struct proc* pp, int n);

struct {
  struct spinlock lock;
  struct proc proc[NPROC];
} ptable;

static struct proc *initproc;

int nextpid = 1;
extern void forkret(void);
extern void trapret(void);

static void wakeup1(void *chan);

void
pinit(void)
{
  initlock(&ptable.lock, "ptable");
}

// Look in the process table for an UNUSED proc.
// If found, change state to EMBRYO and initialize
// state required to run in the kernel.
// Otherwise return 0.
static struct proc*
allocproc(void)
{
  struct proc *p;
  char *sp;

  acquire(&ptable.lock);
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
    if(p->state == UNUSED)
      goto found;
  release(&ptable.lock);
  return 0;

found:
  p->state = EMBRYO;
  p->pid = nextpid++;
  //cprintf("pid: %d \n", nextpid);
  //p->tickets = 1;
  release(&ptable.lock);

  // Allocate kernel stack if possible.
  if((p->kstack = kalloc()) == 0){
    p->state = UNUSED;
    return 0;
  }
  sp = p->kstack + KSTACKSIZE;
  // Leave room for trap frame.
  sp -= sizeof *p->tf;
  p->tf = (struct trapframe*)sp;
  
  // Set up new context to start executing at forkret,
  // which returns to trapret.
  sp -= 4;
  *(uint*)sp = (uint)trapret;

  sp -= sizeof *p->context;
  p->context = (struct context*)sp;
  memset(p->context, 0, sizeof *p->context);
  p->context->eip = (uint)forkret;
 // p->credits = 0;
  return p;
}

// Set up first user process.
void
userinit(void)
{
  struct proc *p;
  extern char _binary_initcode_start[], _binary_initcode_size[];
  
  p = allocproc();
  acquire(&ptable.lock);
  initproc = p;
  if((p->pgdir = setupkvm()) == 0)
    panic("userinit: out of memory?");
  inituvm(p->pgdir, _binary_initcode_start, (int)_binary_initcode_size);
  p->sz = PGSIZE;
  memset(p->tf, 0, sizeof(*p->tf));
  p->tf->cs = (SEG_UCODE << 3) | DPL_USER;
  p->tf->ds = (SEG_UDATA << 3) | DPL_USER;
  p->tf->es = p->tf->ds;
  p->tf->ss = p->tf->ds;
  p->tf->eflags = FL_IF;
  p->tf->esp = PGSIZE;
  p->tf->eip = 0;  // beginning of initcode.S

  safestrcpy(p->name, "initcode", sizeof(p->name));
  p->cwd = namei("/");

  p->state = RUNNABLE;
  release(&ptable.lock);
}

// Grow current process's memory by n bytes.
// Return 0 on success, -1 on failure.
int
growproc(int n)
{
  uint sz;
  
  sz = proc->sz;
  if(n > 0){
    if((sz = allocuvm(proc->pgdir, sz, sz + n)) == 0)
      return -1;
  } else if(n < 0){
    if((sz = deallocuvm(proc->pgdir, sz, sz + n)) == 0)
      return -1;
  }
  proc->sz = sz;
  switchuvm(proc);
  return 0;
}

// Create a new process copying p as the parent.
// Sets up stack to return as if from system call.
// Caller must set state of returned proc to RUNNABLE.
int
fork(void)
{
  int i, pid;
  struct proc *np;

  // Allocate process.
  if((np = allocproc()) == 0)
    return -1;

  // Copy process state from p.
  if((np->pgdir = copyuvm(proc->pgdir, proc->sz)) == 0){
    kfree(np->kstack);
    np->kstack = 0;
    np->state = UNUSED;
    return -1;
  }
  np->sz = proc->sz;
  np->parent = proc;
  *np->tf = *proc->tf;
 // cprintf("num_tickets: %d \n", proc->tickets);
  saveticketstoproccess(np, proc->tickets); //Brooklyn
 // cprintf("child_tickets: %d \n", np->tickets);
  // Clear %eax so that fork returns 0 in the child.
  np->tf->eax = 0;

  for(i = 0; i < NOFILE; i++)
    if(proc->ofile[i])
      np->ofile[i] = filedup(proc->ofile[i]);
  np->cwd = idup(proc->cwd);
 
  pid = np->pid;
  np->state = RUNNABLE;
  safestrcpy(np->name, proc->name, sizeof(proc->name));
  return pid;
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait() to find out it exited.
void
exit(void)
{
  struct proc *p;
  int fd;

  if(proc == initproc)
    panic("init exiting");

  // Close all open files.
  for(fd = 0; fd < NOFILE; fd++){
    if(proc->ofile[fd]){
      fileclose(proc->ofile[fd]);
      proc->ofile[fd] = 0;
    }
  }

  iput(proc->cwd);
  proc->cwd = 0;

  acquire(&ptable.lock);

  // Parent might be sleeping in wait().
  wakeup1(proc->parent);

  // Pass abandoned children to init.
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->parent == proc){
      p->parent = initproc;
      if(p->state == ZOMBIE)
        wakeup1(initproc);
    }
  }
  saveticketstoproccess(proc, 0);
  // Jump into the scheduler, never to return.
  proc->state = ZOMBIE;
  sched();
  panic("zombie exit");
}

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int
wait(void)
{
  struct proc *p;
  int havekids, pid;

  acquire(&ptable.lock);
  for(;;){
    // Scan through table looking for zombie children.
    havekids = 0;
    for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
      if(p->parent != proc)
        continue;
      havekids = 1;
      if(p->state == ZOMBIE){
        // Found one.
        pid = p->pid;
        kfree(p->kstack);
        p->kstack = 0;
        freevm(p->pgdir);
        p->state = UNUSED;
        p->pid = 0;
        p->parent = 0;
        p->name[0] = 0;
        p->killed = 0;
        p->ticks = 0;
        saveticketstoproccess(p, 0);
        release(&ptable.lock);
        return pid;
      }
    }

    // No point waiting if we don't have any children.
    if(!havekids || proc->killed){
      release(&ptable.lock);
      return -1;
    }

    // Wait for children to exit.  (See wakeup1 call in proc_exit.)
    sleep(proc, &ptable.lock);  //DOC: wait-sleep
  }
}

// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run
//  - swtch to start running that process
//  - eventually that process transfers control
//      via swtch back to the sched

/* The following code is added/modified by Ningjie Zhang and nxz190005
settickets function is used to save the tickets assiged from user to process
*/
void setticket(int num)
{
  acquire(&ptable.lock);
  saveticketstoproccess(proc, num);
  release(&ptable.lock);
}
/* End of code added/modified */ 

/* The following code is added/modified by Ningjie Zhang and nxz190005
the total tickets the to calculate the total tickets, it will be used for random values
*/
void saveticketstoproccess(struct proc* p, int n)
{
  total_tickets -= p->tickets;
  p->tickets = n;
  p->used_tickets = n;
  total_tickets += p->tickets;
}
/* End of code added/modified */ 

/* The following code is added/modified by Ningjie Zhang and nxz190005
print out the error msg
*/
void storeticketscheck(struct proc* pp)
{
  if(pp->state != SLEEPING)
    panic("Not sleeping");
#ifdef GET_TICKETS_SWITCH
	  total_tickets -= pp->tickets;
#endif
}
/* End of code added/modified */ 

/* The following code is added/modified by Ningjie Zhang and nxz190005
print out the error msg
*/
void restorecheck(struct proc* pp)
{
  if(pp->state != SLEEPING)
    panic("Not sleeping");
#ifdef GET_TICKETS_SWITCH
	  total_tickets += pp->tickets;
#endif
}
/* End of code added/modified */ 

/* The following code is added/modified by Ningjie Zhang and nxz190005
it is used to calculate the total tickets based on if it is runnable
*/
int get_current_total(void)
{
  struct proc *p;
  int tickets=0;
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
  {
    if(p->state!=RUNNABLE)
    {
      continue;
    }
    tickets+=p->tickets;
  }
  return tickets;
}
/* End of code added/modified */ 


/* The following code is added/modified by Ningjie Zhang and nxz190005
this is the other way to run the project by using credits instead of rand time
*/

/*
void scheduler(void)
{
  struct proc *p;
  acquire(&ptable.lock);
  saveticketstoproccess(ptable.proc, 1);
  release(&ptable.lock);
  
  for(;;){
    sti();
    acquire(&ptable.lock);

    for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
    {
      if(p->state != RUNNABLE)
      {
        continue;
      }
      p->credits += 100000;
    }
    
    for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
    {
      if(p->state != RUNNABLE)
      {
        continue;
      }
      while (500000/(p->tickets)  <=  p->credits)
      {
        proc = p;
	p->inuse = 1;
	p->state = RUNNING;
        const int tick_starttime = ticks;
	int ptotalticks = p->ticks;
	switchuvm(p);
        swtch(&cpu->scheduler, proc->context);	
	int spendticks = ticks - tick_starttime;
	p->ticks = ptotalticks + spendticks;
        cprintf("pid is %d, ticks: %d\n", p->pid, spendticks);
	p->inuse = 0;
        switchkvm();
        proc = 0;
	p->credits -= 500000/(p->tickets);
      }
    }
    release(&ptable.lock);

  }
}
void setallusedtickets(void)
{
  struct proc *p;
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
  { 
    if (p->state != RUNNABLE)
    {
      continue;
    }
    p->used_tickets = p->tickets;
    cprintf("p->tickets is %d\n", p->tickets);
  }
}
*/
/* End of code added/modified */ 



/* The following code is added/modified by Ningjie Zhang and nxz190005
it the core part of this project

basiclly, we use generate a 32 bit random number and mod total tickets to 
get a random golden tickets which is for the winner

and start accumulate the tickets for each process, when the ticket numebr accumulated
is large than the golden tickets, then it is the winner

so we assign this proc as the process going to run
and get the ticks before running and after running

through the ticks before and after running, we can calcuate the total ticks 
used in the running of this process

Our x86 system has two cpus. For high tickets test case, we have two process with tickets of 5 and 100000.
If we pick the large tickets winner to run, another cpu is idle so it will take the lower process. Then low ticket process may run. it will be unfair based on their tickets.
So for the high tickets one, high tickets will have 1200 ticks meanwhile low tickets have 200 ticks I think that is not fail.
 
 for the high ticks case and other fairness case
 
 we must shut the number of cpus down to 1


*/
void
scheduler(void)
{
  struct proc *p;
 
  acquire(&ptable.lock);
  saveticketstoproccess(ptable.proc, 1);
  release(&ptable.lock);
  long h_tickets;

  for(;;){
  
    sti();
    int num_tickets = 0;
    //int num_used_tickets = 0;
    h_tickets = 0;   
    /*
    for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
    {
      if(p->state != RUNNABLE)
      {
        continue;
      }
      num_used_tickets+= p->used_tickets;

    }
    //cprintf("totalusedtickets is : %d, totaltickets is %d\n",num_used_tickets, total_tickets); 
    if (num_used_tickets  <= 0 )
    {
      for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
      {
        if(p->state != RUNNABLE)
        {
          continue;
        }
	p->used_tickets = p->tickets;
        num_used_tickets+=p->used_tickets;
      }
    }*/
   // int act_total_tickets = get_current_total();
   // if (act_total_tickets != total_tickets)
   // {
      
   //   continue;
   // }
    
    unsigned int R = rand();
    h_tickets = (R)%(total_tickets + 1);
    acquire(&ptable.lock);
   // cprintf("R is %d, Total is %d, h_tickets is %d\n", R, total_tickets, h_tickets );
    for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
    {
      if(p->state != RUNNABLE && p->tickets > 0)
      {
        //cprintf("pid is %d, has tickets of %d, num_tickets: %d, total tickets is %d, the h_tickets is %d\n", p->pid, p->tickets, num_tickets, total_tickets, h_tickets);
      }
      
      if(p->state != RUNNABLE)// || p->state != RUNNING)
      {
#ifndef GET_TICKETS_SWITCH
				ticket_count += p->tickets;
#endif
        continue;
      }
      num_tickets += p->tickets;
      if (num_tickets < h_tickets)
      {
        continue;
      }
      //cprintf("pid is %d, has tickets of %d, num_tickets: %d, total tickets is %d, the h_tickets is %d\n", p->pid, p->tickets, num_tickets, total_tickets, h_tickets);
      proc = p;
      switchuvm(p);
      p->state = RUNNING;
      p->inuse = 0;
      int tick_starttime = ticks;//start time of the tick before run
      swtch(&cpu->scheduler, proc->context);
      int ptotalticks = p->ticks; //total ticks accumualted
      int spendticks = ticks - tick_starttime;//current time - star time
      p->ticks = ptotalticks + spendticks;
      p->inuse = 0;
      //p->used_tickets -= 1;
     
      switchkvm();
      proc = 0;
      break;
    }
    release(&ptable.lock);
  }
}
/* End of code added/modified */ 
/* The following code is added/modified by Ningjie Zhang and nxz190005
it is used to get the info from proc and add it to pstat
*/
int getpinfo(struct pstat *cur_pro) {
  struct proc *p;
  int i = 0;
  acquire(&ptable.lock);
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++) 
  {
    if(p->state == UNUSED)
    {
      cur_pro->inuse[i] = 0;
    }
    else
    {
      cur_pro->inuse[i] = 1;
    }
    cur_pro->pid[i] = p->pid;
    cur_pro->tickets[i] = p->tickets;
    cur_pro->ticks[i] = p->ticks;
    ++i;
  }
  release(&ptable.lock);
  return 0;
}
/* End of code added/modified */ 

// Enter scheduler.  Must hold only ptable.lock
// and have changed proc->state.
void
sched(void)
{
  int intena;

  if(!holding(&ptable.lock))
    panic("sched ptable.lock");
  if(cpu->ncli != 1)
    panic("sched locks");
  if(proc->state == RUNNING)
    panic("sched running");
  if(readeflags()&FL_IF)
    panic("sched interruptible");
  intena = cpu->intena;
  swtch(&proc->context, cpu->scheduler);
  cpu->intena = intena;
}

// Give up the CPU for one scheduling round.
void
yield(void)
{
  acquire(&ptable.lock);  //DOC: yieldlock
  if(proc->state == SLEEPING)
  {
    panic("Sleep happens here, apparently. We need to restore tickets.");
  } 
  
  proc->state = RUNNABLE;
  sched();
  release(&ptable.lock);
}

// A fork child's very first scheduling by scheduler()
// will swtch here.  "Return" to user space.
void
forkret(void)
{
  // Still holding ptable.lock from scheduler.
  release(&ptable.lock);
  
  // Return to "caller", actually trapret (see allocproc).
}

// Atomically release lock and sleep on chan.
// Reacquires lock when awakened.
void
sleep(void *chan, struct spinlock *lk)
{
  if(proc == 0)
    panic("sleep");

  if(lk == 0)
    panic("sleep without lk");

  // Must acquire ptable.lock in order to
  // change p->state and then call sched.
  // Once we hold ptable.lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup runs with ptable.lock locked),
  // so it's okay to release lk.
  if(lk != &ptable.lock){  //DOC: sleeplock0
    acquire(&ptable.lock);  //DOC: sleeplock1
    release(lk);
  }

  // Go to sleep.
  proc->chan = chan;
  proc->state = SLEEPING;
  storeticketscheck(proc); 
  sched();

  // Tidy up.
  proc->chan = 0;

  // Reacquire original lock.
  if(lk != &ptable.lock){  //DOC: sleeplock2
    release(&ptable.lock);
    acquire(lk);
  }
}

// Wake up all processes sleeping on chan.
// The ptable lock must be held.
static void
wakeup1(void *chan)
{
  struct proc *p;

  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
    if(p->state == SLEEPING && p->chan == chan)
    {
      restorecheck(p);
      p->state = RUNNABLE;
    }
}

// Wake up all processes sleeping on chan.
void
wakeup(void *chan)
{
  acquire(&ptable.lock);
  wakeup1(chan);
  release(&ptable.lock);
}

// Kill the process with the given pid.
// Process won't exit until it returns
// to user space (see trap in trap.c).
int
kill(int pid)
{
  struct proc *p;

  acquire(&ptable.lock);
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->pid == pid){
      p->killed = 1;
      // Wake process from sleep if necessary.
      if(p->state == SLEEPING)
      {
        restorecheck(p);
        p->state = RUNNABLE;
      }
      release(&ptable.lock);
      return 0;
    }
  }
  release(&ptable.lock);
  return -1;
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [EMBRYO]    "embryo",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  int i;
  struct proc *p;
  char *state;
  uint pc[10];
  
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    cprintf("%d %s %s", p->pid, state, p->name);
    if(p->state == SLEEPING){
      getcallerpcs((uint*)p->context->ebp+2, pc);
      for(i=0; i<10 && pc[i] != 0; i++)
        cprintf(" %p", pc[i]);
    }
    cprintf("\n");
  }
}

