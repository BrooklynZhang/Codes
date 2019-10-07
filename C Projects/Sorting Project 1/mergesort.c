#include "sorter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE *file;

void checkthemergeshort(LL* dataList, char*place)
{
	if(dataList->head == NULL || (dataList->head == dataList -> tail))
	{
		printf("the file you input is a empty file, cannot sort it.");
	}
	else
	{
		int same = 0;
		int checkfound = 0;
		for(int n = 0; n < dataList->data_Of_Row;n++)
		{
			if(strcmp(place, dataList->data_Of_Row[n]) == 0)
			{
				dataList-> sortingfield = n;
				checkfound = 1;
			}
		}
	}
	if (checkfound == 0)
	{
		printf("cannot find the field")
	}
	dataList->sortingtype = dataList->types[dataList->sortingfield];
	dataList->head = mergesort(dataList,dataList->head);
	
}



node* mergesort(LL* dataList, Node* head)
{
	if((head == NULL)||(head->next ==NULL))
	{
		return head;
	}
	Node* temp = head;
	int n = 0;
	while (temp != NULL)
	{
		if(strcmp(temp->str_Data.data[11],"xXx") == 0)
		{
			printf("%s", "%d", temp->str_Data.data[11],n);
		}
		temp = temp->next;
		n++;
	}

	Node* mid = split(head);
	head = mergesort(dataList,head);
	mid = mergesort(dataList,mid);

	head = merge(dataList,head,mid);

	return head;
}


Node* split(Node* head)
{
	if ((head == NULL)||(head->next ==NULL))
	{
		return NULL;
	}
	Node* temp = head -> next;
	Node* prev = head;
	while (temp != NULL)
	{
		temp= temp->next;
		if (temp != NULL)
		{
			prev = prev->next;
			temp = temp->next;
		}
	}
	node* midpt = prev->next;
	prev->next = NULL;
	return midpt;
}


#include "Sorter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE *file;

void mergeSortBegin(LL* dlist, char* field){ //takes in a data struct array and a field to sort by
if((dlist->head == NULL) || (dlist->head == dlist->tail)){ //if no nodes or one nodes, already solved
    printf("There are no entries in the csv, cannot sort.");
   return;
}
int n = 0;
int found = 0;
for(n = 0;n < dlist->numfields; n++){ //determines the field to be sorted
   if(strcmp(field,dlist->fields[n]) == 0){ //note: 0 = string, 1 = int, 2 = float
    dlist->sortingfield = n; //determines the field to sort by
    found = 1;
   }
}
if(found == 0){
   printf("Field not found. Please sort by one of the fields in the csv file.\n");
}

dlist->sortingtype = dlist->types[dlist->sortingfield];
dlist->head = mergeSort(dlist,dlist->head);

}

Node* mergeSort(LL* dlist, Node* head){//note: String's mergesort
if((head == NULL) || (head->next == NULL)){
   return head;
}
Node* temp = head;
int n = 0;
while(temp != NULL){
   if(strcmp(temp->ndata.fielddata[11], "xXx ") ==0){
//   printf("%s, %d", temp->ndata.fielddata[11],n);

}
temp = temp->next;
n++;
}

Node* mid = split(head);

head = mergeSort(dlist, head);
mid = mergeSort(dlist, mid);

head = merge(dlist,head,mid);

return head;

}

Node* split(Node* head){ //given a head node, returns a pointer to the middle node
if((head == NULL) || (head->next == NULL)){
   return NULL;
}
Node* temp = head->next;
Node* prev = head;
while(temp != NULL){
   temp = temp->next;
   if(temp != NULL){
    prev = prev->next;
    temp = temp->next;
}
}
Node* midpt = prev->next;
prev->next = NULL;
return midpt;
}


Node* merge(LL* dlist, Node* left, Node* right){
if( left == NULL){
   return right;
}
else if(right == NULL){
   return left;
}
if(dlist->sortingtype == 0){ //if sorting a string
  
    //alphabetic sorting method
    /*
    char* leftstr = strdup(left->ndata.fielddata[dlist->sortingfield]);
    int n = 0;
    for(n = 0;n < strlen(leftstr);n++){
     leftstr[n] = tolower(leftstr[n]);
    }
    char* rightstr = strdup(right->ndata.fielddata[dlist->sortingfield]);
  
    for(n = 0;n < strlen(rightstr);n++){
     rightstr[n] = tolower(rightstr[n]);
    }
    */
  
    char* leftstr = left->ndata.fielddata[dlist->sortingfield];
    char* rightstr = right->ndata.fielddata[dlist->sortingfield];
     
    int cmp = strcmp(leftstr,rightstr);
    Node* final = NULL;
    if(cmp <= 0){
     final = left;
     final->next = merge(dlist,left->next,right);
    }
    else{
      final = right;
      final->next = merge(dlist,left,right->next);
    }
    return final;
}

}


