#include "dash.h"

extern int flag;
extern bool batchflag;
extern bool checkpath;
extern bool Flag1;
extern bool pathchange;
extern char* path;
extern char paths[bufsize][bufsize];
extern int numpath;


void check_current_process_redirect(char** line, char *filename)
{
  char *path;
  int rc = fork();
  int status = 0;
  if (rc < 0)
  {
    print_error_msg();
  }
  else if (rc == 0)
  {
    char syspath[bufsize] = "/bin /usr/bin";
    //printf("%s","child process\n");
    int pathflag = 0;
    char** cmd = separate_line(syspath, " \t\r\n\a");
    char* pathc = NULL;
    int i = 0;
    while (cmd[i] != NULL)
    {
      // printf("%s","in the llpp\n");
      pathc = (char*)malloc(bufsize);
      strcpy(pathc, cmd[i]);
      strcat(pathc, "/");
      strcat(pathc, line[0]);
      if (access(pathc, X_OK) == 0)
      {
        pathflag =1;
	if(checkpath == false)
	{
	  if (execv(pathc,line) == -1)
	  {
	    print_error_msg();
	    exit(1);
	  }
	  else
	  {
	    print_error_msg();
	    //printf("%s","run the command successfully\n");
	  }
	}
	else
	{
	  print_error_msg();
	 // printf("%s","check path is not false\n");
	}
      }
      else
      {
         print_error_msg();
        //printf("%s","child access not work\n");

      }
      free(pathc);
      i += 1;
    }
    if (checkpath == true){
    int out = open(filename,O_WRONLY|O_CREAT|O_TRUNC,0666);
    int error = open(filename,O_WRONLY|O_CREAT|O_TRUNC,0666);
    fflush(stdout);
    dup2(out,STDOUT_FILENO);
    dup2(out,STDERR_FILENO);
    close(out);
    if (out == -1 || error == -1 || pathflag == 0){
      print_error_msg();
    }
    
    }
    if (pathflag == 1)
    {
      execv(pathc,line);
    }
  }
  else
  {
      waitpid(rc, &status, 0);
      if (status == 1)
      {
        print_error_msg();
      }
  }
}

void check_current_process_normal(char** line) //commands
{

  if (line[0][0] != '\0' && line[0][0] != '\n')//check whether line is empty or not
  {
    if(strcasecmp(line[0],"cd") == 0)//cd
    {  
      // printf("%s","in the cd func\n");
       if (line[1] == NULL)
       {
        // printf("%s","in the cd 1 NULL func\n");
         print_error_msg();
       }
       else if (line[2] != NULL)
       {
         print_error_msg();
       }
       else
       {
         int cds = chdir(line[1]);
	// printf("%s","in the cd func success\n");
	 if (cds != 0)
	 {
	   print_error_msg();
	 }
       }
    }
    else if (strcasecmp(line[0],"exit") == 0)
    {
      exit(0);
    }
    else if (strcasecmp(line[0],"path") == 0) 
    {
      char syspath[bufsize] = "/bin /usr/bin";
      strcpy(syspath, "");
      int i = 0;
      while (line[i] != NULL)
      {
        //printf("%s","in the path loop func\n");
        strcat(syspath, line[i]);
	strcat(syspath, " ");
	i += 1;
      }
    }
    else
    { 
      checkpath = false;
      check_current_process_redirect(line, NULL);
    }
  }
  else
  {
    print_error_msg();
  }
}
