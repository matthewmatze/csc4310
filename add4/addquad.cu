/*
	Name: Matthew Matze
	Date: 10/4/2016
	Class: csc4310
	Location: ~/csc4310/add4

	General Summary of Program

	The program is designed to take one input file holding a matrix of
	integers and add the second, third, and fourth quardrant to the first.
	After the addition the first quadrant is outputed to the output file.

	To Compile:

	nvcc addquad.cu

	To Execute:

	a.out <inputfile >outputfile


*/

#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>

void load(int *matrix, int n);
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

__global__ void kerneladd4(int *matrix, int n);
/*
 * The kerneladd4 function adds the contents of quadrant 2, 3, and 4 into the 
 * contents of quadrant 1. 
 *
 * Precondition: The matrix array is filled with the values of the matrix
 * ready to be processed. The integer holds the size of the rows/columns
 *
 * Postcondition: After execution the contents of the first quadrant of the
 * matrix will be equal to the all four quadrants added together.
*/

void add4(int *matrix, int n);
/*
 * The add4 function sets the size of the grid and block and calls the kerneladd4
 * function. 
 *
 * Precondition: The matrix array is filled with the values of the matrix
 * ready to be processed. The integer holds the size of the rows/columns
 *
 * Postcondition: After execution the contents of the first quadrant of the
 * matrix will be equal to the all four quadrants added together.
*/

void outfunc(int *output, int n);
/*
 * The output function outputs the  matrix in the form of a 1-d array to the
 * output file.
 *
 * Precondition: The first parameter is the array of the integers we have
 * already processed and the second is the row/column dimension
 *
 * Postcondition: After Execution the output file is loaded with the first
 * quadrant of the output array.
*/
int main(int argc, char *argv[]){

	int *matrix;
	int n;
	//Intialize Variables

	scanf("%d", &n);
	//Scan in the size
	
   matrix = (int*) malloc(n*n*sizeof(int *));
   //Allocate memory for matrix
	
	load(matrix, n);
	//Load the 1-d array from the input file
	
	add4(matrix, n);
	//add all quadrants of the matrix into the first

	outfunc(matrix, n);
	//Output the matrix to the file and close the file	

	return 0;
}
void load(int *matrix,int n){
	for(int i=0;i<n*n;i++){
		scanf("%d", &matrix[i]);
	}
}
__global__ void kerneladd4(int *matrix, int n){

	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	if((row<n/2)&&col<(n/2)){
		int loc=row+(col*n);
		matrix[loc]+=matrix[loc+(n/2)]
		+matrix[loc+(n*n/2)]+matrix[loc+(n*n/2)+(n/2)];
	}

}
void add4(int *matrix, int n){
	
	int size=n*n*sizeof(int *);
	int *output;
	cudaMalloc((void **) &output, size);
	cudaMemcpy(output, matrix, size, cudaMemcpyHostToDevice);
	
	dim3 DimGrid((int)ceil((double)n/8.0),(int)ceil((double)n/8.0),1);
	if(size%8) DimGrid.x++;
	dim3 DimBlock(8,8,1);
	
	kerneladd4<<<DimGrid,DimBlock>>>(output, n);
	cudaMemcpy(matrix, output, size, cudaMemcpyDeviceToHost);

	cudaFree(output);
}
void outfunc(int *output, int n){
	int loc=0;
	printf("%d\n", n);
	for(int j=0;j<n/2;j++){
		for(int i=0;i<n/2;i++){
			loc=i+(j*n);
			printf("%d ", output[loc]);
		}
		printf("\n");
	}
}
