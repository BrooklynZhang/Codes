#include "sorter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mergesort.c"

FILE *file;

void printData(linkedlist* dlist, data cdata){ //cmov = currentdata, prints the contents of a data struct
     int n=0;
     for(n = 0;n < dlist->numfields;n++){
       printf("%s: %s\n",dlist->fields[n], cdata.fielddata[n]);
     }
} P

void printTypes(linkedlist* dlist){
int n = 0;
for(n = 0; n < dlist->numfields; n++){
   printf("%s type:", dlist->fields[n]);
   if(dlist->types[n] == 0){
   printf("String\n");
   }
   else if(dlist->types[n] == 1){
    printf("Long\n");
   }
   else{
    printf("Float\n");
   }
}
}


void addTail(linkedlist* dlist, data cdata){ //takes a linked list and a data, adds the data to a node and puts it at the end of the linked list
   Node* datanode = (Node*) malloc(sizeof(Node)); //allocate space for a node
   datanode->next = NULL;
   datanode->ndata = cdata; //set the new node's ndata to cdata
if((dlist->head == NULL) && (dlist->tail == NULL)){ //empty list case
   dlist->head = datanode;
   dlist->tail = datanode;
   dlist->count++;
}
else if((dlist->head == dlist->tail)){ //single node case
   dlist->tail = datanode;
   dlist->head->next = dlist->tail;
   dlist->count++;
}
else{// every other case
   dlist->tail->next = datanode;
   dlist->tail = datanode;
   dlist->count++;
}
}

void Finish(linkedlist* dlist){ //frees everything in a LL
Node* temp = dlist->head;
int n = 0;
while(temp!= NULL){
   char** ftemp = temp->ndata.fielddata;
   for(n = 0;n < dlist->numfields;n++){
    free(ftemp[n]);
   }
   free(ftemp);
Node* oldtemp = temp;
temp = temp->next;
free(oldtemp);
}
for(n = 0;n < dlist->numfields;n++){
   free(dlist->fields[n]);
}

free(dlist->fields);
free(dlist->types);
free(dlist);
}

void initializeList(linkedlist* dlist, char* fields){ //initializes the numfields, types, and fields in the linked list
char* temp = strdup(fields);
char* currvar = (char*) strtok(temp, ",");
int totalfields = 0; //counts total fields, determined by the number of tokens found
while(currvar != 0){ //discovers total number of fields
    totalfields++;
    currvar = (char*) strtok(NULL,",");
  
    }
dlist->numfields = totalfields;
dlist->types = (int*) malloc(sizeof(int)*dlist->numfields); //allocates fields for keeping track of what type each field is, to be used once all data is received
dlist->fields = (char**) malloc(sizeof(char*)*dlist->numfields); //keeps track of the name of the fields, to be used later when sorting
free(temp);
temp = strdup(fields);
currvar = (char*) strtok(temp, ",");
    int n;
for(n = 0;n < dlist->numfields;n++)
{
      if(n == dlist->numfields-1)
	  { //strips the newline character for the last field
         int specchar = strcspn(currvar, "\r\n"); //find the location of the first \r or the first \n
         if(specchar != 0)
		 { //if the location isn't the start
                currvar[specchar] = '\0'; //remove the special characters
          
		  }
      }
     dlist->fields[n] = strdup(currvar);
     currvar = (char*) strtok(NULL,",");
    }
free(temp);
free(currvar);
}

void initializeListTypes(linkedlist* dlist){ //determines what type of data belongs to what field
if(dlist->head == NULL){
return;
}
Node* temp = dlist->head;
int n = 0;
for(n;n < dlist->numfields;n++){
   char* tempstr = temp->ndata.fielddata[n];
   char* endptr;
   long lg = 0;
   float fl = 0.0;
   if(n == dlist->numfields-1){ //if this field is the last field on the line, remove \r and \n
       int specchar = strcspn(tempstr, "\r\n"); //find the location of the first \r or the first \n
   if(specchar != 0){ //if the location isn't the start
   tempstr[specchar] = '\0'; //remove the special characters
   }
   }
   if(strchr(tempstr, '.')){ //if it has a '.', it's a string or a float/double
       fl = strtof(tempstr, &endptr);
       if((endptr == tempstr) || (endptr != (tempstr+strlen(tempstr)))){ //if endptr == tempstr, no floats at the start, the other conditional checks to make sure that the end of the read in float is the end of the string inputted
   dlist->types[n] = 0;
       }
       else{
   dlist->types[n] = 2;
       }
   }
   else{ //if it has no '.', it's either an int or a string
     lg = strtol(tempstr, &endptr, 10);
     if((endptr == tempstr) || (endptr != (tempstr+strlen(tempstr)))){ //if endptr == tempstr, no floats at the start, the other conditional checks to make sure that the end of the read in float is the end of the string inputted
   dlist->types[n] = 0;
       }
       else{
   dlist->types[n] = 1;
       }
   
}
}

}


void export(linkedlist* dlist){
int n = 0;
for(n = 0;n < dlist->numfields;n++){
   if(n != dlist->numfields-1){
   printf("%s,",dlist->fields[n]);
   }
   else{
    printf("%s\n",dlist->fields[n]);
   }
}
Node* temp = dlist->head;
while(temp != NULL){
   if(temp->ndata.comma == NULL){ //if there's no commas in the fields

    for(n = 0;n < dlist->numfields;n++){
      if(n != dlist->numfields-1){
   printf("%s,",temp->ndata.fielddata[n]);
   }
      else{
   printf("%s\n",temp->ndata.fielddata[n]);
   }
      }
   }
   else{
   
    for(n = 0;n < dlist->numfields;n++){
      if(temp->ndata.comma[n] == 0){
    
      if(n != dlist->numfields-1){
   printf("%s,",temp->ndata.fielddata[n]);
   }
      else{
   printf("%s\n",temp->ndata.fielddata[n]);
   }
      }
      else{
   if(n != dlist->numfields-1){
   printf("\"%s\",",temp->ndata.fielddata[n]);
   }
      else{
   printf("\"%s\"\n",temp->ndata.fielddata[n]);
   }
  
      }
      }
   
   
   }


temp = temp->next;
}
}

int main(int argc, char *argv[]){
//Constructs an array containing all of the data data into cdata
file = stdin;
LL* dlist = (LL*) malloc(sizeof(LL)); //malloc a data linked list to which data nodes will be added
char* test = (char*) malloc(sizeof(char)*1000); //for getting data from the file
memset(test, 0 , sizeof(char)*1000);
//i should make sure there's no problem when there's more than 1000 chars
fgets(test,1000,file); //skips the column line
initializeList(dlist, test);

char c;
int currentvar=0;
int count = 0;
int place = 0;
int quote = 0;
data cdatanode = {};
cdatanode.fielddata = (char**) malloc(sizeof(char*)*dlist->numfields);
cdatanode.comma = NULL;
memset(test, 0 , sizeof(char)*1000);
while((c = fgetc(file)) != EOF){
    /*
    if(c < 0 || c > 128){
      continue;
    }
    */
    if((c == ',' || c == '\n') && (quote == 0)){
      cdatanode.fielddata[currentvar] = strdup(test); //copy test to cdatanode STRDUP IS MALLOC, REMEMBER TO FREE
      if(c == '\n'){
      addTail(dlist, cdatanode);
      count++;
      currentvar = 0;
      place = 0;
      memset(test, 0 , sizeof(char)*1000);
      cdatanode.fielddata = (char**) malloc(sizeof(char*)*dlist->numfields); //get new char array for cdatanode
      cdatanode.comma = NULL;
    }
    else{
      currentvar++;
      place = 0;
      memset(test, 0 , sizeof(char)*1000);
    }
    
    }
    else if(c == '"'){
      if(quote == 0){
   //insert comma handling here, somehow
   cdatanode.comma = (int*) malloc(sizeof(int)*dlist->numfields);
   memset(cdatanode.comma,0,sizeof(sizeof(int)*dlist->numfields));
   cdatanode.comma[currentvar] = 1;
   quote = 1;
      }
      else{
   quote = 0;
      }
    }
    else{
      test[place] = c;
      place++;
    }
}
free(cdatanode.fielddata);
free(test);


//printData(dlist,dlist->head->ndata);
//printData(dlist,dlist->head->next->ndata);
//printData(dlist,dlist->tail->ndata);
initializeListTypes(dlist);
//printTypes(dlist);

//mergeSort(dlist, dlist->fields[25]);

int n = 0;
Node* temp = dlist->head;


if(argc >= 3){
    if(strcmp("-c", argv[1]) == 0){
mergeSortBegin(dlist, argv[2]);
    }
}

export(dlist); //exports the dlist into csv
// printData(dlist, split(dlist,dlist->head)->ndata);
// Finish(dlist); //frees everything in dlist
}