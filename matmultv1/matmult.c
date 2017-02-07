#include<stdio.h>
#include<stdlib.h>

void load(FILE *file1, int **matrix1, int n);
//void multiply(**matrix1,**matrix2,**output);
//void output(**output);

int main(int argc, char *argv[]){

FILE* file1;
//FILE* file2;
int **matrix1;
//int** matrix2;

int n;

file1=fopen("matrix1","r");
//file2=fopen("matrix2","r");

//fscanf(file2, "%d", &n);
fscanf(file1, "%d", &n);

printf("lol");
matrix1 = (int**) malloc(n*sizeof(int *));
for(int i=0; i<n; i++){

	matrix1[i] = (int *) malloc(n*sizeof(int));	

}
//matrix2 = malloc(n*sizeof(int));
//for(int i=0; i<n; i++){

//matrix2[i] = malloc(n*sizeof(int));	

//}

printf("lol");
matrix1[0][0] = 22;
printf("%d", matrix1[0][0]);
load(file1, matrix1, n);
//printf("%d\n", n);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%d", matrix1[j][i]);
		}
	}
return 0;
}
void load(FILE *file1,int **matrix1,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			fscanf(file1,"%d", matrix1[j][i]);
			printf("%d", matrix1[j][i]);
		}
	}
}
