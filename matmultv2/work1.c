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

	mpicc matmultv2.c -o execFile

	To Execute:
	
	mpiexec --mca btl_tcp_if_include enp5s0 -n 4 -hostfile hostFile execFile matrix1 matrix2 output

*/

#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#define MASTER 0
#define TAG1 1
#define TAG2 2
#define TAG3 3
#define TAG4 4
#define TAG5 5
#define TAG6 6
#define TAG7 7
#define TAG8 8
#define TAG9 9
#define TAG10 10
#define TAG11 11
#define TAG12 12
#define TAG13 13
#define TAG14 14
#define TAG15 15
#define TAG16 16
#define TAG17 17
#define TAG18 18
#define TAG19 19
#define TAG20 20
#define TAG21 21
#define TAG22 22
#define TAG23 23
#define TAG24 24
#define TAG25 25

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
void allocatemem(int **sub1mat1,int **sub2mat1,int **sub3mat1,int **sub4mat1, int **sub1mat2,
int **sub2mat2,int **sub3mat2,int **sub4mat2,int **out1,int **out2,int **out3,int **out4,
int **out5,int **out6,int **out7,int **out8,int chunksize);

int main(int argc, char *argv[]){

	int numtasks, rank;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	//Intialize Variables

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("MPI task %d of %d has started...\n",rank+1, numtasks);

	if(rank == MASTER)
		master_node_process(argv, numtasks, rank);
	else
		worker_node_process(numtasks, rank);

	MPI_Finalize();

	return 0;

}
void master_node_process(char *argv[],int numtasks, int rank){

	FILE *file1, *file2, *outfile;
	int *sub1mat1, *sub2mat1, *sub3mat1, *sub4mat1;
	int *sub1mat2, *sub2mat2, *sub3mat2, *sub4mat2;
	int *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8;
	int n;
	int chunksize;
	MPI_Status status;
	//Initialize Variables

	file1=fopen(argv[1],"r");
	file2=fopen(argv[2],"r");
	outfile=fopen(argv[3],"w");
	//Open input and output files

	fscanf(file2, "%d", &n);
	fscanf(file1, "%d", &n);
	//Scan in the size of each so both are properly incremented in read file
	
	chunksize=n*n/4;

	allocatemem(&sub1mat1,&sub2mat1,&sub3mat1,&sub4mat1,&sub1mat2,&sub2mat2,
	&sub3mat2,&sub4mat2,&out1,&out2,&out3,&out4,&out5,&out6,&out7,&out8,
	chunksize);
	//Allocate Memory

	load(file1, sub1mat1, sub2mat1, sub3mat1, sub4mat1, n);
	load(file2, sub1mat2, sub2mat2, sub3mat2, sub4mat2, n);
	fclose(file1);
	fclose(file2);

	
	//Load the 1-d arrays from the input files and Close the input files
    MPI_Send(&n, 1, MPI_INT, 1, TAG17, MPI_COMM_WORLD);
    MPI_Send(sub1mat1, chunksize, MPI_INT, 1, TAG1, MPI_COMM_WORLD);
    MPI_Send(sub1mat2, chunksize, MPI_INT, 1, TAG2, MPI_COMM_WORLD);
	//Worker 1 m11*m12

   MPI_Send(&n, 1, MPI_INT, 2, TAG17, MPI_COMM_WORLD);
	MPI_Send(sub2mat1, chunksize, MPI_INT, 2, TAG1, MPI_COMM_WORLD);
   MPI_Send(sub3mat2, chunksize, MPI_INT, 2, TAG2, MPI_COMM_WORLD);
	//Worker 2 m21*m32

   MPI_Send(&n, 1, MPI_INT, 3, TAG17, MPI_COMM_WORLD);
   MPI_Send(sub1mat1, chunksize, MPI_INT, 3, TAG1, MPI_COMM_WORLD);
   MPI_Send(sub2mat2, chunksize, MPI_INT, 3, TAG2, MPI_COMM_WORLD);
	//Worker 3 m11*m22

   MPI_Send(&n, 1, MPI_INT, 4, TAG17, MPI_COMM_WORLD);
   MPI_Send(sub2mat1, chunksize, MPI_INT, 4, TAG1, MPI_COMM_WORLD);
   MPI_Send(sub4mat2, chunksize, MPI_INT, 4, TAG2, MPI_COMM_WORLD);
	//Worker 4 m21*m42

   MPI_Send(&n, 1, MPI_INT, 5, TAG17, MPI_COMM_WORLD);
   MPI_Send(sub3mat1, chunksize, MPI_INT, 5, TAG1, MPI_COMM_WORLD);
   MPI_Send(sub1mat2, chunksize, MPI_INT, 5, TAG2, MPI_COMM_WORLD);
	//Worker 5 m31*m12
		
   MPI_Send(&n, 1, MPI_INT, 6, TAG17, MPI_COMM_WORLD);
   MPI_Send(sub4mat1, chunksize, MPI_INT, 6, TAG1, MPI_COMM_WORLD);
   MPI_Send(sub3mat2, chunksize, MPI_INT, 6, TAG2, MPI_COMM_WORLD);
	//Worker 6 m41*m32
		
   MPI_Send(&n, 1, MPI_INT, 7, TAG17, MPI_COMM_WORLD);
   MPI_Send(sub3mat1, chunksize, MPI_INT, 7, TAG1, MPI_COMM_WORLD);
   MPI_Send(sub2mat2, chunksize, MPI_INT, 7, TAG2, MPI_COMM_WORLD);
	//Worker 7 m31*m22
		
   MPI_Send(&n, 1, MPI_INT, 8, TAG17, MPI_COMM_WORLD);
   MPI_Send(sub4mat1, chunksize, MPI_INT, 8, TAG1, MPI_COMM_WORLD);
   MPI_Send(sub4mat2, chunksize, MPI_INT, 8, TAG2, MPI_COMM_WORLD);
	//Worker 8 m41*m42

   MPI_Recv(out1, chunksize, MPI_INT, 1, TAG18, MPI_COMM_WORLD,&status);
   MPI_Recv(out2, chunksize, MPI_INT, 2, TAG19, MPI_COMM_WORLD,&status);
   MPI_Recv(out3, chunksize, MPI_INT, 3, TAG20, MPI_COMM_WORLD,&status);
   MPI_Recv(out4, chunksize, MPI_INT, 4, TAG21, MPI_COMM_WORLD,&status);
   MPI_Recv(out5, chunksize, MPI_INT, 5, TAG22, MPI_COMM_WORLD,&status);
   MPI_Recv(out6, chunksize, MPI_INT, 6, TAG23, MPI_COMM_WORLD,&status);
   MPI_Recv(out7, chunksize, MPI_INT, 7, TAG24, MPI_COMM_WORLD,&status);
   MPI_Recv(out8, chunksize, MPI_INT, 8, TAG25, MPI_COMM_WORLD,&status);
	//Recieve multiplied Matrices

	addition(out1, out2, n/2);
	addition(out3, out4, n/2);
	addition(out5, out6, n/2);
	addition(out7, out8, n/2);
	//Add the matrices together

	outfunc(outfile, out1, out3, out5, out7, n);
	fclose(outfile);
	//Output the matrix to and close the output file
	
}

void allocatemem(int **sub1mat1,int **sub2mat1,int **sub3mat1,int **sub4mat1, int **sub1mat2,int **sub2mat2,int **sub3mat2,int **sub4mat2,int **out1,int **out2,int **out3,int **out4,int **out5,int **out6,int **out7,int **out8,int chunksize){
	
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
void worker_node_process(int numtasks, int rank){
	int *sub1mat1, *sub2mat1, *sub3mat1, *sub4mat1;
	int *sub1mat2, *sub2mat2, *sub3mat2, *sub4mat2;
	int *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8;
	int chunksize=0;
	int n=0;
	MPI_Status status;
	
	if(rank==1){
		
     	MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub1mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub1mat2 = (int*) malloc(chunksize*sizeof(int *));
		out1 = (int*) malloc(chunksize*sizeof(int *));
     	MPI_Recv(sub1mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
     	MPI_Recv(sub1mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub1mat1, sub1mat2, out1, n/2);
     	MPI_Send(out1, chunksize, MPI_INT, 0, TAG18, MPI_COMM_WORLD);
	}
	//Worker 1 m11*m12
	
	if(rank==2){
     	MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub2mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub3mat2 = (int*) malloc(chunksize*sizeof(int *));
		out2 = (int*) malloc(chunksize*sizeof(int *));
     	MPI_Recv(sub2mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
     	MPI_Recv(sub3mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub2mat1, sub3mat2, out2, n/2);
     	MPI_Send(out2, chunksize, MPI_INT, 0, TAG19, MPI_COMM_WORLD);
	}
	//Worker 2 m21*m32

	if(rank==3){
     	MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub1mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub2mat2 = (int*) malloc(chunksize*sizeof(int *));
		out3 = (int*) malloc(chunksize*sizeof(int *));
  	 	MPI_Recv(sub1mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
    	MPI_Recv(sub2mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub1mat1, sub2mat2, out3, n/2);
     	MPI_Send(out3, chunksize, MPI_INT, 0, TAG20, MPI_COMM_WORLD);
	}
	//Worker 3 m11*m22

	if(rank==4){
     	MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub2mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub4mat2 = (int*) malloc(chunksize*sizeof(int *));
		out4 = (int*) malloc(chunksize*sizeof(int *));
     	MPI_Recv(sub2mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
     	MPI_Recv(sub4mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub2mat1, sub4mat2, out4, n/2);
     	MPI_Send(out4, chunksize, MPI_INT, 0, TAG21, MPI_COMM_WORLD);
	}
	//Worker 4 m21*m42

	if(rank==5){
      MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub3mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub1mat2 = (int*) malloc(chunksize*sizeof(int *));
		out5 = (int*) malloc(chunksize*sizeof(int *));
   	MPI_Recv(sub3mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
    	MPI_Recv(sub1mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub3mat1, sub1mat2, out5, n/2);
      MPI_Send(out5, chunksize, MPI_INT, 0, TAG22, MPI_COMM_WORLD);
	}
	//Worker 5 m31*m12
		
	if(rank==6){
     	MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub4mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub3mat2 = (int*) malloc(chunksize*sizeof(int *));
		out6 = (int*) malloc(chunksize*sizeof(int *));
     	MPI_Recv(sub4mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
     	MPI_Recv(sub3mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub4mat1, sub3mat2, out6, n/2);
     	MPI_Send(out6, chunksize, MPI_INT, 0, TAG23, MPI_COMM_WORLD);
	}
	//Worker 6 m41*m32
		
	if(rank==7){
     	MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub3mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub2mat2 = (int*) malloc(chunksize*sizeof(int *));
		out7 = (int*) malloc(chunksize*sizeof(int *));
     	MPI_Recv(sub3mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
     	MPI_Recv(sub2mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub3mat1, sub2mat2, out7, n/2);
     	MPI_Send(out7, chunksize, MPI_INT, 0, TAG24, MPI_COMM_WORLD);
	}
	//Worker 7 m31*m22
		
	if(rank==8){
     	MPI_Recv(&n, 1, MPI_INT, 0, TAG17, MPI_COMM_WORLD, &status);
		chunksize=n*n/4;
		sub4mat1 = (int*) malloc(chunksize*sizeof(int *));
		sub4mat2 = (int*) malloc(chunksize*sizeof(int *));
		out8 = (int*) malloc(chunksize*sizeof(int *));
     	MPI_Recv(sub4mat1, chunksize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, &status);
     	MPI_Recv(sub4mat2, chunksize, MPI_INT, 0, TAG2, MPI_COMM_WORLD, &status);
		multiply(sub4mat1, sub4mat2, out8, n/2);
     	MPI_Send(out8, chunksize, MPI_INT, 0, TAG25, MPI_COMM_WORLD);
	}
	//Worker 8 m41*m42

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
}
