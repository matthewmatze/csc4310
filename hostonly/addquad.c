/*
	Name: Matthew Matze
	Date: 9/28/2016
	Class: csc4310
	Location: ~/csc4310/hostonly

	General Summary of Program

	The program is designed to take one input file holding a matrix of
	integers and add the second, third, and fourth quardrant to the first.
	After the addition the first quadrant is outputed to the output file.

	To Compile:

	gcc -Wall addquad.c

	To Execute:

	a.out inputfile outputfile


*/

#include<stdio.h>
#include<stdlib.h>

void load(FILE *file, int *matrix, int n);
/*
 * The load function puts the matrix from the file into a 1-d array of ints.
 *
 * Precondition: The file has the row/column dimension on the first line
 * which has already been read in. On the following lines it will have that
 * number of rows and columns of integers to be read in. The next parameter
 * is an empty array of integers large enough to hold the contents of the 
 * input file. Lastly we have the row/column value in the final in parameter
 *
 * Postcondition: After Execution the file has been completely read through 
 * and the integer array is now fully loaded from the provided input file
*/
void add4(int *matrix, int n);
/*
 * The add4 function adds the contents of quadrant 2, 3, and 4 into the 
 * contents of quadrant 1. 
 *
 * Precondition: The matrix array is filled with the values of the matrix
 * ready to be processed. The integer holds the size of the rows/columns
 *
 * Postcondition: After execution the contents of the first quadrant of the
 * matrix will be equal to the all four quadrants added together.
*/
void outfunc(FILE *outfile, int *output, int n);
/*
 * The output function outputs the  matrix in the form of a 1-d array to the
 * output file.
 *
 * Precondition: The first parameter is the outputfile and it starts out
 * completely empty. Secondly we have the array of the integers we have
 * already processed and lastly we have the row/column dimension
 *
 * Postcondition: After Execution the output file is loaded with the first
 * quadrant of the output array.
*/
int main(int argc, char *argv[]){

	FILE *file;
	FILE *outfile;
	int *matrix;
	int n;
	//Intialize Variables

	file=fopen(argv[1],"r");
	outfile=fopen(argv[2],"w");
	//Open input and output files

	fscanf(file, "%d", &n);
	//Scan in the size of each so both are properly incremented in read file

	matrix = (int*) malloc(n*n*sizeof(int *));
	//Allocate memory for matrix1,matrix2, and the output

	load(file, matrix, n);
	fclose(file);
	//Load the 1-d array from the input file and close file


	add4(matrix, n);
	//add all quadrants of the matrix into the first

	outfunc(outfile, matrix, n);
	fclose(outfile);
	//Output the matrix to the file and close the file

	return 0;
}
void load(FILE *file,int *matrix,int n){
	for(int i=0;i<n*n;i++){
		fscanf(file,"%d", &matrix[i]);
	}
}
void add4(int *matrix, int n){
	int loc=0;
	for(int j=0;j<n/2;j++){
		for(int i=0;i<n/2;i++){
			loc=i+(j*n);
			matrix[loc]+=matrix[loc+(n/2)]
			+matrix[loc+(n*n/2)]+matrix[loc+(n*n/2)+(n/2)];
		}
	}
}
void outfunc(FILE *outfile, int *output, int n){
	int loc=0;
	fprintf(outfile,"%d\n", n);
	for(int j=0;j<n/2;j++){
		for(int i=0;i<n/2;i++){
			loc=i+(j*n);
			fprintf(outfile,"%d ", output[loc]);
		}
		fprintf(outfile,"\n");
	}
}
