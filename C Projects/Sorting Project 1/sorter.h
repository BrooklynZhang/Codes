/*****
*
*	Define structures and function prototypes for your sorter
*
*
*
******/

//Suggestion: define a struct that mirrors a record (row) of the data set


//Suggestion: prototype a mergesort function



#ifndef _SORTER_H_
#define _SORTER_H_


typedef struct_data{
	char **data2;
	int *comma;
}data;

typedef stuct_node{
	data nodedata;
	stuct _node*next;
}node;

typedef struct_linkedlist{
	node *head;
	node *tail;
	int count;
	int numfields;
	int sortingfiled;
	int sortingtype;
	int *types;
	char **fields;
}linkedlist;

linkedlist* mergesortBegin(linkedlist* dlist, char* field);
node* mergesort(linkedlist* dlist, node* head);
node* split(node* head);
node* merge(linkedlist* dlist, node* left, node* right);

#endif


