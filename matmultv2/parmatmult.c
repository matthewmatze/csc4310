/*
	Name: Matthew Matze
	Date: 9/14/2016
	Class: csc4310
	Location: ~/csc4310/matmultv2

	General Summary of Program
	
	The program is designed to take in two input files of two square matrices
	with the first line of the file being the row/column length. The rest of
	the input file is a document which has each row of the matrix separated by
	new line characters and the columns are separated by white space characters.
	The output matrix is outputed in the exact same format as the input.
	
	To Compile:

	mpicc 21.c -o execFile

	To Execute:
	
	mpiexec --mca btl_tcp_if_include enp5s0 -n 9 -hostfile hostFile execFile matrix1 matrix2 output

*/

#include "openmpi-x86_64/mpi.h"
#include<sys/time.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#define MASTER 0
#define TAG1 1
#define TAG2 2
#define TAG3 3
#define TAG4 4

void load(FILE *file, int *sub1mat, int *sub2mat, int *sub3mat, int *sub4mat, int n);
/*
 * The load function puts the matrix from the file into 4 different 1-d arrays 
 * of ints of the four different quadrants.
 *
 * Precondition: The parameters are a pointer to the input file, pointers to the 
 * empty 1-d int arrays for each of the individual quadrants of the  matrix, and 
 * lastly the final int is the length of the rows/column. 
 *
 * Postcondition: The after the program is run the entire matrix is loaded into
 * the four different 1-d int arrays and the entire input file is read.
*/
void multiply(int *matrix1,int *matrix2,int *output, int n);
/*
 * The multiply function does the dot product of the two matrices and puts
 * the result in the output 1-d array.
 *
 * Precondition: The parameters of the function are the two matrices to be
 * multiplied stored as integer arrays, an output matrix stored into an array,
 * and the rows/column size.
 *
 * Postcondition: After the function is executed the output array will be filled
 * with the result of the multiplication of matrix1 and matrix2.
*/
void outfunc(FILE *outfile, int *output1, int *output2, int *output3, int *output4, int n);
/*
 * The output function takes the output matrices of the four quadrants in the 
 * form of a 1-d array and puts them into an output file in the correct format.
 *
 * Precondition: The parameters are an empty output file, the four arrays are
 * the four quadrants of the matrix post multiplication, and the last element
 * is an int which is the size of the rows/column.
 *
 * Postcondition: After execution the output file is loaded in the same format
 * of the input file with the result arrays of each quadrant
 *
*/

void addition(int *matrix1, int *matrix2, int n);
/*
 * The addition function adds two matrices together and puts the result into the output array
 *
*/
void master_node_process(char *argv[], int numtasks, int rank);
/*
 * Executes the master node's process
 *
*/
void worker_node_process(int numtasks, int rank);
/*
 * Executes worker thread process
 *
*/
void openfiles(char *argv[], FILE **file1,FILE **file2,FILE **outfile,int *n);
/*
 *
 * The function opens the files and scans in the rows/column size
 *
 * Precondition: argv is loaded with the names of the files,file1 and file2 are the file pointers
 * to the input file, outfile is the file pointer to the output file, and the integer point is the
 * value at the address of n
 *
 * Postcondition: file1 and file2 are loaded with the input names from argv, outfile is loaded 
 * with the output name, and n is loaded with the number of rows/column size of the matrix
 *
*/
void addall(int *out1,int *out2,int *out3,int *out4,int *out5,int *out6,int *out7,int *out8,int n);
/*
 *The function adds all of the correctly Multiplied arrays together 
 *
 *Precondition: All of the output int arrays are used as parameters with the size of the
 *rows/column
 *
 *Postcondition: The ouput arrays are added together. out1 has the value of out1+out2,out3 has the
 *value of out3+out4 and so on.
 *
*/
void send2arrays(int *matrix1,int *matrix2,int rank,int n);
/*
 *The function sends two matrices in the form of two int arrays to the worker thread
 *
 *Precondition:matrix1 and matrix2 are loaded from the input files with the matrices,the rank of
 *of the thread is a parameter, and the row/column size is n
 *
 *Postcondition: The two martrices are sent to be multiplied to the worker threaded assigned by rank
 *
*/ 
void sendallarrays(int *sub1mat1,int *sub2mat1,int *sub3mat1,int *sub4mat1,int *sub1mat2,
int *sub2mat2, int *sub3mat2, int *sub4mat2,int n);
/*
 *This function executes all of the send2array functions
 *
 *Precondition: the function has all of the sub matrices and the row/column size for parameters
 *
 *Postcondition: The submatrices are sent by executing the send2arrays functions 
 *
*/
void recvall(int *out1,int *out2,int *out3,int *out4,int *out5,int *out6,int *out7,
int *out8, int n);
/*
 *The function is designed to recieve all of the submatrices which had been multiplied 
 *in the worker threads
 *
 *Precondition: The empty allocated output int arrays and row/column size are the parameters 
 *
 *Postcondition: The output arrays are filled with the contents recieved from the worker threads
 *
*/
void allocatemem(int **sub1mat1,int **sub2mat1,int **sub3mat1,int **sub4mat1, int **sub1mat2,
int **sub2mat2,int **sub3mat2,int **sub4mat2,int **out1,int **out2,int **out3,int **out4,
int **out5,int **out6,int **out7,int **out8,int n);
/*
 * This function allocates memory for all of the sub and output matrices
 *
 *Precondition: All sub and output arrays are referenced by double pointers in order to call
 *them by reference inside the function also the row/column size is a parameter too
 *
 *Postcondition: The sub and output arrays are called by reference and malloced so the arrays
 *have allocated the appropriate size ammount of memory 
 *
*/
void work(int *matrix1,int *matrix2,int *output);
/*
 *The function recieves the size of the row/column,matrix1, and matrix2. Then it multiplies
 *matrix1 times matrix2 and saves it in the output array then sends the output to the master 
 *
 *Precondition: Three integer arrays are inputed as parameters and all are empty
 *
 *Postcondition: matrix1 and matrix2 recieve memory from the master and are multiplied 
 *together and the answer is stored in output and is sent back to the master
*/ 

int main(int argc, char *argv[]){
	int numtasks, rank;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	double start, stop, diff;
	//Intialize Variables
	

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("MPI task %d of %d has started...\n",rank+1, numtasks);

	if(rank == MASTER){
		start = MPI_Wtime();
		master_node_process(argv, numtasks, rank);
		stop = MPI_Wtime();
		diff = stop - start;
		printf("MPI_Wtime %f\n",diff);	
	} else
		worker_node_process(numtasks, rank);

	MPI_Finalize();

	return 0;

}
void master_node_process(char *argv[],int numtasks, int rank){
	FILE *file1, *file2, *outfile;
	int *sub1mat1, *sub2mat1, *sub3mat1, *sub4mat1,*sub1mat2; 
	int *sub2mat2, *sub3mat2, *sub4mat2, *out1, *out2, *out3; 
	int *out4, *out5, *out6, *out7, *out8, n, chunksize;
	MPI_Status status;
	//Initialize Variables
	
	openfiles(argv, &file1,&file2,&outfile,&n);
	//Open files and scan in n
	
	allocatemem(&sub1mat1,&sub2mat1,&sub3mat1,&sub4mat1,&sub1mat2,&sub2mat2,
	&sub3mat2,&sub4mat2,&out1,&out2,&out3,&out4,&out5,&out6,&out7,&out8,n);
	//Allocate Memory

	load(file1, sub1mat1, sub2mat1, sub3mat1, sub4mat1, n);
	load(file2, sub1mat2, sub2mat2, sub3mat2, sub4mat2, n);
	//Load the 1-d arrays from the input files
		
	sendallarrays(sub1mat1,sub2mat1,sub3mat1,sub4mat1,sub1mat2,sub2mat2,
	sub3mat2,sub4mat2,n);

	recvall(out1,out2,out3,out4,out5,out6,out7,out8,n);	
	addall(out1,out2,out3,out4,out5,out6,out7,out8,n);	
	//Recieve and add all output matrices
	
	outfunc(outfile, out1, out3, out5, out7, n);
	//Output the matrix to the output file	
}
void openfiles(char *argv[], FILE **file1,FILE **file2,FILE **outfile,int *n){

	*file1=fopen(argv[1],"r");
	*file2=fopen(argv[2],"r");
	*outfile=fopen(argv[3],"w");
	fscanf(*file2, "%d", n);
	fscanf(*file1, "%d", n);

}
void addall(int *out1,int *out2,int *out3,int *out4,int *out5,int *out6,
int *out7,int *out8,int n){
	addition(out1, out2, n/2);
	addition(out3, out4, n/2);
	addition(out5, out6, n/2);
	addition(out7, out8, n/2);
}
void recvall(int *out1,int *out2,int *out3,int *out4,int *out5,int *out6,int *out7,int *out8,int n){
	int chunksize=n*n/4;
   MPI_Status status;
	MPI_Recv(out1, chunksize, MPI_INT, 1, TAG3, MPI_COMM_WORLD,&status);
   MPI_Recv(out2, chunksize, MPI_INT, 2, TAG3, MPI_COMM_WORLD,&status);
   MPI_Recv(out3, chunksize, MPI_INT, 3, TAG3, MPI_COMM_WORLD,&status);
   MPI_Recv(out4, chunksize, MPI_INT, 4, TAG3, MPI_COMM_WORLD,&status);
   MPI_Recv(out5, chunksize, MPI_INT, 5, TAG3, MPI_COMM_WORLD,&status);
   MPI_Recv(out6, chunksize, MPI_INT, 6, TAG3, MPI_COMM_WORLD,&status);
   MPI_Recv(out7, chunksize, MPI_INT, 7, TAG3, MPI_COMM_WORLD,&status);
   MPI_Recv(out8, chunksize, MPI_INT, 8, TAG3, MPI_COMM_WORLD,&status);

}
void sendallarrays(int *sub1mat1,int *sub2mat1,int *sub3mat1,int *sub4mat1,
int *sub1mat2,int *sub2mat2, int *sub3mat2, int *sub4mat2, int n){
	
	send2arrays(sub1mat1,sub1mat2,1,n);//Worker 1 m11*m12
	send2arrays(sub2mat1,sub3mat2,2,n);//Worker 2 m21*m32
	send2arrays(sub1mat1,sub2mat2,3,n);//Worker 3 m11*m22
	send2arrays(sub2mat1,sub4mat2,4,n);//Worker 4 m21*m42
	send2arrays(sub3mat1,sub1mat2,5,n);//Worker 5 m31*m12
	send2arrays(sub4mat1,sub3mat2,6,n);//Worker 6 m41*m32
	send2arrays(sub3mat1,sub2mat2,7,n);//Worker 7 m31*m22
	send2arrays(sub4mat1,sub4mat2,8,n);//Worker 8 m41*m42

}
void send2arrays(int *matrix1,int *matrix2,int rank,int n){
	int chunksize=n*n/4;
   MPI_Send(&n, 1, MPI_INT, rank, TAG4, MPI_COMM_WORLD);
   MPI_Send(matrix1, chunksize, MPI_INT, rank, TAG1, MPI_COMM_WORLD);
   MPI_Send(matrix2, chunksize, MPI_INT, rank, TAG2, MPI_COMM_WORLD);

}

void allocatemem(int **sub1mat1,int **sub2mat1,int **sub3mat1,int **sub4mat1, int **sub1mat2,int **sub2mat2,int **sub3mat2,int **sub4mat2,int **out1,int **out2,int **out3,int **out4,int **out5,int **out6,int **out7,int **out8,int n){
	int chunksize=n*n/4;
	*sub1mat1 = (int*) malloc(chunksize*sizeof(int *));
	*sub2mat1 = (int*) malloc(chunksize*sizeof(int *));
	*sub3mat1 = (int*) malloc(chunksize*sizeof(int *));
	*sub4mat1 = (int*) malloc(chunksize*sizeof(int *));
	*sub1mat2 = (int*) malloc(chunksize*sizeof(int *));
	*sub2mat2 = (int*) malloc(chunksize*sizeof(int *));
	*sub3mat2 = (int*) malloc(chunksize*sizeof(int *));
	*sub4mat2 = (int*) malloc(chunksize*sizeof(int *));
	*out1 = (int*) malloc(chunksize*sizeof(int *));
	*out2 = (int*) malloc(chunksize*sizeof(int *));
	*out3 = (int*) malloc(chunksize*sizeof(int *));
	*out4 = (int*) malloc(chunksize*sizeof(int *));
	*out5 = (int*) malloc(chunksize*sizeof(int *));
	*out6 = (int*) malloc(chunksize*sizeof(int *));
	*out7 = (int*) malloc(chunksize*sizeof(int *));
	*out8 = (int*) malloc(chunksize*sizeof(int *));
	//Allocate memory for all matrices and output arrays 
}
void work(int *matrix1,int *matrix2,int *output){
	int chunksize=0;
	int n=0;
	MPI_Status status;
  	MPI_Recv(&n, 1, MPI_INT, 0, TAG4, MPI_COMM_WORLD, &status);
	chunksize=n*n/4;
	matrix1 = (int*) malloc(chunksize*sizeof(int *));
	matrix2 = (int*) malloc(chunksize*sizeof(int *));
	output = (int*) malloc(chunksize*sizeof(int *));
  	MPI_Recv(matrix1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
  	MPI_Recv(matrix2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
	multiply(matrix1, matrix2, output, n/2);
  	MPI_Send(output, chunksize, MPI_INT, 0, TAG3, MPI_COMM_WORLD);
}
void worker_node_process(int numtasks, int rank){
	int *sub1mat1, *sub2mat1, *sub3mat1, *sub4mat1;
	int *sub1mat2, *sub2mat2, *sub3mat2, *sub4mat2;
	int *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8;
	
	if(rank==1)//Worker 1 m11*m12
		work(sub1mat1,sub1mat2,out1);	
	if(rank==2)//Worker 2 m21*m32
		work(sub2mat1, sub3mat2, out2);
	if(rank==3)//Worker 3 m11*m22
		work(sub1mat1, sub2mat2, out3);
	if(rank==4)//Worker 4 m21*m42
		work(sub2mat1, sub4mat2, out4);
	if(rank==5)//Worker 5 m31*m12
		work(sub3mat1, sub1mat2, out5);
	if(rank==6)//Worker 6 m41*m32
		work(sub4mat1, sub3mat2, out6);
	if(rank==7)//Worker 7 m31*m22
		work(sub3mat1, sub2mat2, out7);
	if(rank==8)//Worker 8 m41*m42
		work(sub4mat1, sub4mat2, out8);
}

void load(FILE *file, int *sub1mat, int *sub2mat, int *sub3mat, int *sub4mat, int n){
	for(int j=0;j<n/2;j++){
		for(int i=0;i<n/2;i++){
			fscanf(file,"%d", &sub1mat[i+(j*(n/2))]);
		}
		for(int i=0;i<n/2;i++){
			fscanf(file,"%d", &sub2mat[i+(j*(n/2))]);
		}
	}
	for(int j=0;j<n/2;j++){
		for(int i=0;i<n/2;i++){
			fscanf(file,"%d", &sub3mat[i+(j*(n/2))]);
		}
		for(int i=0;i<n/2;i++){
			fscanf(file,"%d", &sub4mat[i+(j*(n/2))]);
		}
	}
	fclose(file);
}

void multiply(int *matrix1,int *matrix2, int *output, int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			output[i*n+j]=0;
			for(int k=0;k<n;k++){
				output[i*n+j] += (matrix1[k+(i*n)] * matrix2[(k*n)+j]);	
			}
		}
	}
}

void addition(int *matrix1, int *matrix2, int n){
	for(int i=0;i<n*n;i++)
		matrix1[i]+=matrix2[i];
}

void outfunc(FILE *outfile, int *output1, int *output2, int *output3, int *output4, int n){
	fprintf(outfile,"%d\n", n);
	for(int j=0;j<n/2;j++){
		for(int i=0;i<n/2;i++){
			fprintf(outfile,"%d ", output1[i+(j*n/2)]);
		}
		for(int i=0;i<n/2;i++){
			fprintf(outfile,"%d ", output2[i+(j*n/2)]);
		}
		fprintf(outfile,"\n");
	}
	for(int j=0;j<n/2;j++){
		for(int i=0;i<n/2;i++){
			fprintf(outfile,"%d ", output3[i+(j*n/2)]);
		}
		for(int i=0;i<n/2;i++){
			fprintf(outfile,"%d ", output4[i+(j*n/2)]);
		}
		fprintf(outfile,"\n");
	}
	fclose(outfile);
}
