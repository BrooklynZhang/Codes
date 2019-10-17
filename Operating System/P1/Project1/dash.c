#include "dash.h"

// user input (interactive)                batch mode
// 						
// ------->   commands store in stdin       <-------
//
//            loop to execute each commands
//
//                   execute done
//
//                   go back and ask for input again

char error_msg[128] = "An error has occurred\n";
int flag = 0;
bool batchflag = false;
bool checkpath = false;
bool Flag1 = false;
bool pathchange = false;
char* path;
char paths[bufsize][bufsize];
int numpath;



int main(int argc, char* argv[])
{
  
  if (argc == 1)  //interactive mode, loop go infinity til exit
  {
    interactive_mode(); //dash
  }
  else if (argc == 2) //batch mode, reads input from a batch file
  {
    char* batchname = strdup(argv[1]);
    batch_mode(batchname);
  }
  else
  {
    print_error_msg();
  }

}

void testmsg()
{
  char msg[128] = "checkpt \n";
  write(STDERR_FILENO, msg, strlen(msg));
}


void print_error_msg()
{
    write(STDERR_FILENO, error_msg, strlen(error_msg));
    exit(1);
}
