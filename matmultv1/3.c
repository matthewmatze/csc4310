/*
	Name: Matthew Matze
	Date: 9/4/2016
	Class: csc4310
	Location: ~/csc4310/matmultv1
*/

#include<stdio.h>
#include<stdlib.h>

void load(FILE *file1, int *matrix, int n);
/*
 * The load function puts the matrix from the file into a 1-d array of ints.
*/
void multiply(int *matrix1,int *matrix2,int *output, int n);
/*
 * The multiply function does the dot product of the two matrices and puts
 * the result in the output 1-d array.
*/
void outfunc(FILE *outfile, int *output, int n);
/*
 * The output function takes the output matrix in the form of a 1-d array
 * and puts it into an output file.
*/
int main(int argc, char *argv[]){

FILE *file1;
FILE *file2;
FILE *outfile;
int *matrix1;
int *matrix2;
int *output;
int n;
//Intialize Variables

file1=fopen(argv[1],"r");
file2=fopen(argv[2],"r");
outfile=fopen(argv[3],"w");
//Open input and output files

fscanf(file2, "%d", &n);
fscanf(file1, "%d", &n);
//Scan in the size of each so both are properly incremented in read file

matrix1 = (int*) malloc(n*n*sizeof(int *));
matrix2 = (int*) malloc(n*n*sizeof(int *));
output = (int*) malloc(n*n*sizeof(int *));
//Allocate memory for matrix1,matrix2, and the output

load(file1, matrix1, n);
load(file2, matrix2, n);
//Load the 1-d arrays from the input files

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
void load(FILE *file1,int *matrix,int n){
	for(int i=0;i<n*n;i++){
		fscanf(file1,"%d", &matrix[i]);
	}
}
void multiply(int *matrix1,int *matrix2, int *output, int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			for(int k=0;k<n;k++){
			output[i*n+j] += (matrix1[k+(i*n)] * matrix2[(k*n)+j]);	
			}
		}
	}
}
void outfunc(FILE *outfile, int *output, int n){
	fprintf(outfile,"%d\n", n);
	for(int i=0;i<n*n;i++){
			fprintf(outfile,"%d", output[i]);
			if(0!=(i+1)%n)
				fprintf(outfile," ");
			else
				fprintf(outfile,"\n");
	}
}
