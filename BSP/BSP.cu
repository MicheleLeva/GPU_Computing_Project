#include <stdlib.h>
#include <stdio.h>
#include "../utils/common.h"

#define N 100
#define M 100

typedef struct Node_t{
    int topRightx;
    int topRighty;
    int bottomLeftx;
    int bottomLefty;
    Node_t *leftChild;
    Node_t *rightChild;
}Node;

typedef struct Tree_t{
  Node root;
	Node * leftChild;
  Node * rightChild;
} Tree;

void print(Tree * tree){}

int main() {

Tree * BSPTree = (Tree *)malloc(sizeof(Tree));
Node * head = NULL;
head -> topRightx = N;
head -> topRighty = M;
head -> bottomLeftx = 0;
head -> bottomLefty = 0;
BSPTree -> root = *head;


return 0;
}