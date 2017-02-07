/*
	Name: Matthew Matze
	Date: 11/1/2016
	Class: csc4310
	Location: ~/csc4310/cuda_mult3

   General Summary of Program

   The program is designed to take two matrices via input files and output
   the result into the resultant file. 

   To Compile: 

   nvcc cudamultv3.cu -o cudamultv3

   To Execute:

   cudamultv3 Device_Number Tile_Width File_One File_Two Output_File

   To Script:

   nohup cudaRun.sh Device_Number &
*/

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>

void load(FILE *file1, float *matrix, int n);
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
//__global__ void kernelmult(int *matrix1, int *matrix2, int *output, int n);

__global__ void kernelmult(float *Md, float *Nd, float *Pd, int Width,
      int TILE_WIDTH);
/*
 * The Kernel Multiply function multiplies the the matrices together
 *
 * Precondition: The first two parameters are integer arrays to be multiplied, 
 * third is a array to put the output in, and the last if for the size
 *
 * Postcondtion: The multiplier function multiplys the first two matrices and
 * puts the output in the third
 *
*/
void multiply(int tilewidth, float *matrix1,float *matrix2, float *output, int n,
FILE *kerneltime);
/*
 * The multiply function sets up the kernel and executes the kernel multiply
 * function.
 *
 * Precondition: The tilewidth is the user inputed size of the tile, matrix1 
 * and matrix2 are the inputed matrices, the output matrix is the processed
 * matrix and the size of the dimensions
 *
 * Postcondition: The output matrix is completely filled with the multiplied
 * first two matrices.
*/
void outfunc(FILE *outfile, float *output, int n);
/*
 * The output function takes the output matrix in the form of a 1-d array
 * and puts it into an output file.
 * output function outputs the  matrix in the form of a 1-d array to the
 * output file.
 *
 * Precondition: The first parameter is the array of the integers we have
 * already processed and the second is the row/column dimension
 *
 * Postcondition: After Execution the output file is loaded with the first
 * quadrant of the output array.
 */

int main(int argc, char *argv[]){
	struct timeval startTime1, stopTime1, startTime2, stopTime2;
	struct timeval startTime3, stopTime3, startTime4, stopTime4;
	struct timeval startTime5, stopTime5, startTime6, stopTime6;
	double start, stop, diff1 = 0, diff2 = 0, diff3 =0, diff4 = 0, diff5 = 0;
   double diff6 = 0;
	gettimeofday(&startTime1,NULL);

	int device = atoi(argv[1]);
   cudaSetDevice(device);
	int tilewidth = atoi(argv[2]);
	FILE *file1;
	FILE *file2;
	FILE *outfile;
	FILE *timing;
	FILE *cudatime1;
	FILE *cudatime2;
	FILE *cudatime3;
	FILE *cudatime4;
	FILE *cudatime5;
	FILE *kerneltime;
	float *matrix1;
	float *matrix2;
	float *output;
	int n;
	//Intialize Variables

	file1=fopen(argv[3],"r");
	file2=fopen(argv[4],"r");
	outfile=fopen(argv[5],"w");
	timing=fopen("tottime.csv","a");
	cudatime1=fopen("cudatime1.csv","a");
	cudatime2=fopen("cudatime2.csv","a");
	cudatime3=fopen("cudatime3.csv","a");
	cudatime4=fopen("cudatime4.csv","a");
	cudatime5=fopen("cudatime5.csv","a");
	kerneltime=fopen("kerneltime.csv","a");
	//Open input and output files

	fscanf(file2, "%d", &n);
	fscanf(file1, "%d", &n);
	//Scan in the size of each so both are properly incremented in read file
	//fprintf(timing, "%d,%d,", n, tilewidth);

	matrix1 = (float*) malloc(n*n*sizeof(float *));
	matrix2 = (float*) malloc(n*n*sizeof(float *));
	output = (float*) malloc(n*n*sizeof(float *));
	//Allocate memory for matrix1,matrix2, and the output

	load(file1, matrix1, n);
	load(file2, matrix2, n);
	//Load the 1-d arrays from the input files

	fclose(file1);
	fclose(file2);
	//Close the input files
	gettimeofday(&startTime2,NULL);

	multiply(tilewidth, matrix1, matrix2, output, n, kerneltime);
	
   gettimeofday(&stopTime2,NULL);
	start = startTime2.tv_sec + (startTime2.tv_usec/1000000.0);
	stop = stopTime2.tv_sec + (stopTime2.tv_usec/1000000.0);
	diff2 = stop - start;
	fprintf(cudatime1, ",%lf", diff2);
	gettimeofday(&startTime3,NULL);
	
   multiply(tilewidth, matrix1, matrix2, output, n, kerneltime);
	
   gettimeofday(&stopTime3,NULL);
	start = startTime3.tv_sec + (startTime2.tv_usec/1000000.0);
	stop = stopTime3.tv_sec + (stopTime2.tv_usec/1000000.0);
	diff3 = stop - start;
	fprintf(cudatime2, ",%lf", diff3);
	gettimeofday(&startTime4,NULL);
	
   multiply(tilewidth, matrix1, matrix2, output, n, kerneltime);
	
   gettimeofday(&stopTime4,NULL);
	start = startTime4.tv_sec + (startTime4.tv_usec/1000000.0);
	stop = stopTime4.tv_sec + (stopTime4.tv_usec/1000000.0);
	diff4 = stop - start;
	fprintf(cudatime3, ",%lf", diff4);
   gettimeofday(&startTime5,NULL);
	
   multiply(tilewidth, matrix1, matrix2, output, n, kerneltime);
	gettimeofday(&stopTime5,NULL);
	start = startTime5.tv_sec + (startTime5.tv_usec/1000000.0);
	stop = stopTime5.tv_sec + (stopTime5.tv_usec/1000000.0);
	diff5 = stop - start;
	fprintf(cudatime4, ",%lf", diff5);
   gettimeofday(&startTime6,NULL);
	
   multiply(tilewidth, matrix1, matrix2, output, n, kerneltime);
	gettimeofday(&stopTime6,NULL);
	start = startTime6.tv_sec + (startTime6.tv_usec/1000000.0);
	stop = stopTime6.tv_sec + (stopTime6.tv_usec/1000000.0);
	diff6 = stop - start;
	fprintf(cudatime5, ",%lf", diff6);
   //Multiply the Matrices

	outfunc(outfile, output, n);
	//Output the matrix to the file
	
	fclose(outfile);
	//Close the output file
	
	gettimeofday(&stopTime1,NULL);
	start = startTime1.tv_sec + (startTime1.tv_usec/1000000.0);
	stop = stopTime1.tv_sec + (stopTime1.tv_usec/1000000.0);
	diff1 = stop - start;
	fprintf(timing, ",%lf", diff1);
	return 0;
}
void load(FILE *file1,float *matrix,int n){
	for(int i=0;i<n*n;i++){
		fscanf(file1,"%f", &matrix[i]);
	}
}
__global__ void kernelmult(float *Md, float *Nd, float *Pd, int Width,
      int TILE_WIDTH){
   
   extern __shared__ float smem[];

   float *Mds = smem;
   float *Nds = smem + (TILE_WIDTH*TILE_WIDTH);

   int bx = blockIdx.x; int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

   float Pvalue1 = 0;
   float Pvalue2 = 0;
   float Pvalue3 = 0;
   float Pvalue4 = 0;

   int s_offset1 = TILE_WIDTH/2;
   int g_offset1 = TILE_WIDTH/2;

   int s_offset2 = s_offset1*TILE_WIDTH;
   int g_offset2 = g_offset1*Width;

   int s_offset3 = s_offset1+s_offset2;
   int g_offset3 = g_offset1+g_offset2;

   int s_bp = ty*TILE_WIDTH+tx;

   for(int m=0;m<Width/TILE_WIDTH;++m){

      Mds[s_bp] = Md[Row*Width+(m*TILE_WIDTH+tx)];
      Nds[s_bp] = Nd[Col*Width+(m*TILE_WIDTH+ty)];

      Mds[s_bp+s_offset1] = Md[Row*Width+(m*TILE_WIDTH+tx+g_offset1)];
      Nds[s_bp+s_offset1] = Nd[Col*Width+(m*TILE_WIDTH+ty+g_offset1)];

      Mds[s_bp+s_offset2] = Md[Row*Width+(m*TILE_WIDTH+tx+g_offset2)];
      Nds[s_bp+s_offset2] = Nd[Col*Width+(m*TILE_WIDTH+ty+g_offset2)];

      Mds[s_bp+s_offset3] = Md[Row*Width+(m*TILE_WIDTH+tx+g_offset3)];
      Nds[s_bp+s_offset3] = Nd[Col*Width+(m*TILE_WIDTH+ty+g_offset3)];

      __syncthreads();

      for(int k=0; k<TILE_WIDTH;k++){
   
         Pvalue1 += Mds[ty*TILE_WIDTH+k]*Nds[k*TILE_WIDTH+tx];
         Pvalue2 += Mds[ty*TILE_WIDTH+k]*Nds[k*TILE_WIDTH+tx+s_offset1];
         Pvalue3 += Mds[ty*TILE_WIDTH+k+s_offset2]*Nds[k*TILE_WIDTH+tx];
         Pvalue4 += Mds[ty*TILE_WIDTH+k+s_offset2]*Nds[k*TILE_WIDTH+tx+s_offset1];

         __syncthreads();

      }

   }
   Pd[(Width*Row)+Col] = Pvalue1;  
   Pd[(Width*Row)+Col+g_offset1] = Pvalue2;  
   Pd[(Width*Row)+Col+g_offset2] = Pvalue3;  
   Pd[(Width*Row)+Col+g_offset3] = Pvalue4;  
}
/*
__global__ void kernelmult(float *Md, float *Nd, float *Pd, int Width,
      int TILE_WIDTH){
   
   extern __shared__ float smem[];

   float *Mds = smem;
   float *Nds = smem + (TILE_WIDTH*TILE_WIDTH);

   int bx = blockIdx.x; int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

   float Pvalue =0;
   for(int m=0;m<Width/TILE_WIDTH;++m){
      Mds[ty*TILE_WIDTH+tx]=Md[Row*Width+(m*TILE_WIDTH+tx)];
      Nds[ty*TILE_WIDTH+tx]=Nd[(m*TILE_WIDTH+ty)*Width+Col];
      __syncthreads();

      for(int k=0; k<TILE_WIDTH;++k){
	      Pvalue += (Mds[(ty*TILE_WIDTH)+k] *  Nds[(k*TILE_WIDTH)+tx]);
      } 
      __syncthreads();
   }
   Pd[(Width*Row)+Col] = Pvalue;  
}
*/


/*
__global__ void kernelmult(int *matrix1, int *matrix2, int *output, int n){

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	output[row*n+col]=0;
	for(int i=0;i<n;i++)
		output[row*n+col] += (matrix1[i+(row*n)] * matrix2[(i*n)+col]);

}
*/
void multiply(int tilewidth, float *matrix1,float *matrix2, float *output, int n,
FILE *kerneltime){

   int smem_size=2*tilewidth*tilewidth*sizeof(float *);
   int size=n*n*sizeof(float *);
   float *m1,*m2,*o;
	double start, stop, diff = 0;
   struct timeval startTime, stopTime;
   cudaMalloc((void **) &m1, size);
   cudaMemcpy(m1, matrix1, size, cudaMemcpyHostToDevice);
   cudaMalloc((void **) &m2, size);
   cudaMemcpy(m2, matrix2, size, cudaMemcpyHostToDevice);
   cudaMalloc((void **) &o, size);

   dim3 DimGrid((int)ceil((double)n/(double)tilewidth),
	(int)ceil((double)n/(double)tilewidth),1);
   dim3 DimBlock(tilewidth/2,tilewidth/2,1);
	gettimeofday(&startTime,NULL);
   
   kernelmult<<<DimGrid,DimBlock,smem_size>>>(m1, m2, o, n, tilewidth);

   gettimeofday(&stopTime,NULL);
	start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
	stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0);
	diff = stop - start;
	fprintf(kerneltime, ",%lf", diff);
   cudaMemcpy(output, o, size, cudaMemcpyDeviceToHost);

   cudaFree(m1);
   cudaFree(m2);
   cudaFree(o);

}
void outfunc(FILE *outfile, float *output, int n){
	fprintf(outfile,"%d\n", n);
	for(int i=0;i<n*n;i++){
			fprintf(outfile,"%.0f", output[i]);
			if(0!=(i+1)%n)
				fprintf(outfile," ");
			else
				fprintf(outfile,"\n");
	}
}
