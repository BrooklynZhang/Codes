#include "dash.h"

extern int flag;
extern bool batchflag;
extern bool checkpath;
extern bool Flag1;
extern bool pathchange;
extern char* path;
extern char paths[bufsize][bufsize];
extern int numpath;

int separate_line_num(char* line, char* deli);
char** separate_line(char* line, char* deli);

void checkparacommands(char* line)
{
  char** spearatecmd = NULL;
  char** parallelcmd = NULL;
  parallelcmd = separate_line(line, "&");
  int loc = 0;
  int rc;
  while (parallelcmd[loc] != NULL)
  {  
    char** recmd = separate_line(parallelcmd[loc], ">");
    if (strstr(parallelcmd[loc],">") != NULL) //redirection
    {
      if (recmd[1] != NULL)
      {
        char** redictfname = separate_line(recmd[1], " \t\r\n\a");
	spearatecmd = separate_line(recmd[0], " \t\r\n\a");
	if (spearatecmd[0] != NULL)
	{
	  checkpath = true;
	  //printf("%s","redirect model\n");
          check_current_process_redirect(spearatecmd, redictfname[0]);
	}
      }
    }
    else
    {
      char** nonredcmd = NULL;
      nonredcmd = separate_line(recmd[0], " \t\r\n\a");
      if (nonredcmd[0] != NULL)
      {
        //printf("%s","in the normal model func\n");
        check_current_process_normal(nonredcmd);
      }

    }
    loc += 1;
  }
}


void interactive_mode()
{
  write(STDOUT_FILENO, "dash> ", strlen("dash> ")); //dash
  char* line = NULL;
  size_t len = 0;
  ssize_t read;

  while ((read = getline(&line, &len, stdin)) != -1)
  {
    Flag1 = false;
    if (read > 0 && line[read -1] == '\n')
    {
      if (strstr(line, "&") == NULL) //when inputs has no parallel commands
      {
        checkparacommands(line);
      }
      else // has parallel commands
      {
        checkparacommands(line);
      }
    }
    else
    {
      print_error_msg();
    }
    write(STDOUT_FILENO, "dash> ", strlen("dash> ")); 
  }
}

char** separate_line(char* line, char* deli)
{
  char** elements = malloc(bufsize*sizeof(char*));
  char* element = NULL;
  int n = 0;
  element = strtok(line, deli);
  while (element != NULL)
  {
    elements[n] = element;
    n += 1;
    element = strtok(NULL, deli);
  }
  elements[n] = NULL;
  return elements;
}

int separate_line_num(char* line, char* deli)
{
  char** elements = malloc(bufsize*sizeof(char*));
  char* element = NULL;
  int n = 0;
  element = strtok(line, deli);
  while (element != NULL)
  {
    elements[n] = element;
    n += 1;
    element = strtok(NULL, deli);
  }
  return n;


}


void batch_mode(char* filename)
{
  char* line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE* batchfile = fopen(filename, "r");
  if (batchfile == NULL)
  {
    print_error_msg();
  }


  while ((read = getline(&line, &len, batchfile)) != -1)
  {
    Flag1 = false;
    if (read > 0 && line[read -1] == '\n')
    {
      if (strstr(line, "&") == NULL) //when inputs has no parallel commands
      {
        checkparacommands(line);
      }
      else // has parallel commands
      {
        checkparacommands(line);
      }
    }
  }
  fclose(batchfile);
  exit(0);
}

