/*
	Name: Matthew Matze
	Date: 9/3/2016
	Class: csc4310
	Location: ~/csc4310/matmultv1
*/


#include<stdio.h>
#include<stdlib.h>
void load(FILE *file1, int **matrix, int n);
/*
 * The load function puts the matrix from the file into a 2-d array of ints.
*/
void multiply(int **matrix1,int **matrix2,int **output, int n);
/*
 * The multiply function does the dot product of the two matrices and puts
 * the result in the output 2-d array.
*/
void outfunc(FILE *outfile, int **output, int n);
/*
 * The output function takes the output matrix in the form of a 2-d array
 * and puts it into an output file.
*/
int main(int argc, char *argv[]){

FILE *file1;
FILE *file2;
FILE *outfile;
int **matrix1;
int **matrix2;
int **output;
int n;
//Intialize Variables

file1=fopen(argv[1],"r");
file2=fopen(argv[2],"r");
outfile=fopen(argv[3],"w");
//Open input and output files

fscanf(file2, "%d", &n);
fscanf(file1, "%d", &n);
//Scan in the size of each so both are properly incremented in read file

matrix1 = (int**) malloc(n*sizeof(int *));
for(int i=0; i<n; i++){
	matrix1[i] = (int *) malloc(n*sizeof(int));	
}

matrix2 = (int**) malloc(n*sizeof(int *));
for(int i=0; i<n; i++){
	matrix2[i] = (int *) malloc(n*sizeof(int));	
}

output = (int**) malloc(n*sizeof(int *));
for(int i=0; i<n; i++){
	output[i] = (int *) malloc(n*sizeof(int));	
}
//Allocate memory for matrix1,matrix2, and the output

load(file1, matrix1, n);
load(file2, matrix2, n);
//Load the 2-d arrays from the input files

fclose(file1);
fclose(file2);
//Close the input files

multiply(matrix1, matrix2, output, n);
//Multiply the Matrices

outfunc(outfile, output, n);
//Output the matrix to the file

fclose(outfile);
//Close the output file

return 0;
}
void load(FILE *file1,int **matrix,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			fscanf(file1,"%d", &matrix[j][i]);
		}
	}
}
void multiply(int **matrix1,int **matrix2, int **output, int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			for(int k=0;k<n;k++){
				output[j][i] += (matrix1[k][i] * matrix2[j][k]);	
			}
		}
	}
}
void outfunc(FILE *outfile, int **output, int n){
	fprintf(outfile,"%d\n", n);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			fprintf(outfile,"%d", output[j][i]);
			if(j!=n-1)
				fprintf(outfile," ");
			else
				fprintf(outfile,"\n");
		}
	}
}
