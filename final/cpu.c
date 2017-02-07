/*
	Name: Matthew Matze
	Date: 11/1/2016
	Class: csc4310
	Location: ~/csc4310/cuda_mult3

   General Summary of Program

   The program is designed to take two matrices via input files and output
   the result into the resultant file. 

   To Compile: 

   gcc cudamultv3.cu -o cudamultv3

   To Execute:

   a.out Input_File Output_File
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

//void load(FILE *file1, float *matrix, int n);
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

//void multiply(int tilewidth, float *matrix1,float *matrix2, float *output, int n,
//FILE *kerneltime);
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
//void outfunc(FILE *outfile, float *output, int n);
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
struct pixel {
   char r;
   char g;
   char b;
};
void zeromat(struct pixel **output,int width,int height);
   
void readfile(struct pixel **image,int width,int height, FILE *file1);

void sobelx(struct pixel **image,struct pixel **output,int *sobel_x,int width,int height,double* maxrgb,double* minrgb,double max);

void writefile(struct pixel **output,int width, int height, FILE *outfile,char*type,int max);

void calcmax(struct pixel **image, int width, int height, int *sobel, double *minrgb, double *maxrgb, int min, double max);
int main(int argc, char *argv[]){
	
	FILE *file1;
	FILE *outfile;
	float *matrix1;
	float *matrix2;
	//float *output;
	char *type;
   int width;
   int height;
   double max;
   int min;

   struct pixel **image;
   struct pixel **output;
   int sobel_x[9] = {-1,0,1,
                     -2,0,2,
                     -1,0,1};
   int sobel_y[9] = {1, 2, 1,
                     0, 0, 0,
                    -1,-2,-1};
	//Intialize Variables

	file1=fopen(argv[1],"r");
	outfile=fopen(argv[2],"w");
	//Open input and output files
	type = (char *)malloc(5*sizeof(char *));

	fscanf(file1,"%s\n%d %d\n%d\n",type,&width,&height,&max);
   min = -max;
   double minrgb[3] = {min,min,min};
   double maxrgb[3] = {max,max,max};
	image = (struct pixel **)malloc(width*sizeof(struct pixel *));
   for(int i=0;i<width;i++)
      image[i]=(struct pixel *)malloc(height*sizeof(struct pixel));

	output = (struct pixel **)malloc(width*sizeof(struct pixel *));
   for(int i=0;i<width;i++)
      output[i]=(struct pixel *)malloc(height*sizeof(struct pixel));

   readfile(image,width,height,file1);
   zeromat(output,width,height);
   calcmax(image, width, height, sobel_x, minrgb, maxrgb, min, max);
   sobelx(image,output,sobel_x,width,height,maxrgb,minrgb,max);
   writefile(output,width, height, outfile,type,max);

//   int red=abs(ceil(sqrt((px.p1*px.p1)+(py.p1*py.p1))));
//   int green=abs(ceil(sqrt((px.p2*px.p2)+(py.p2*py.p2))));
//   int blue=abs(ceil(sqrt((px.p3*px.p3)+(py.p3*py.p3))));
//   image[i].p1 = image[i].p2 = image[i].p3 = (red+green+blue)/3;

	fclose(file1);
	fclose(outfile);
	
	return 0;
}
void readfile(struct pixel **image,int width,int height,FILE *file1){

   for(int i=0;i<width;i++){
      for(int j=0;j<height;j++){
         fread(&image[i][j].r, 1,1,file1);
         fread(&image[i][j].g, 1,1,file1);
         fread(&image[i][j].b, 1,1,file1);
      }
   }
}

void zeromat(struct pixel **output,int width,int height){

   for(int i=0;i<width;i++){
      for(int j=0;j<height;j++){
         output[i][j].r=(char)0;
         output[i][j].g=(char)0;
         output[i][j].b=(char)0;
      }
   }

}
void calcmax(struct pixel **image, int width, int height, int *sobel_x, double *minrgb, double *maxrgb, int min, double max){
   for (int y = 1; y < width - 1; y++) {
      for (int x = 1; x < height - 1; x++) {
         double pixel_value[3] = {0,0,0};
         for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
               pixel_value[0] += sobel_x[j+i+2] * (int)image[y + j][x + i].r;
               pixel_value[1] += sobel_x[j+i+2] * (int)image[y + j][x + i].g;
               pixel_value[2] += sobel_x[j+i+2] * (int)image[y + j][x + i].b;
            }
         }
         for(int k=0;k<3;k++){
            if (pixel_value[k] < minrgb[k]) minrgb[k] = pixel_value[k];
            if (pixel_value[k] > maxrgb[k]) maxrgb[k] = pixel_value[k];
         }
      } 
   }
   for(int k=0;k<3;k++){
      if ((int)(maxrgb[k] - minrgb[k]) == 0) {
         printf("Nothing exists!!!\n\n");
         exit(1);
      }
   }
}
void sobelx(struct pixel **image,struct pixel **output,int *sobel_x,int width,int height,double *maxrgb,double *minrgb,double max){
   for(int row=1;row<width-1;row++){
      for(int col=1;col<height-1;col++){
         double rgb[3] ={0,0,0};
         for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
               rgb[0]+=sobel_x[i*3+j]*(int)image[row-1+i][col-1+j].r;
               rgb[1]+=sobel_x[i*3+j]*(int)image[row-1+i][col-1+j].g;
               rgb[2]+=sobel_x[i*3+j]*(int)image[row-1+i][col-1+j].b;
            }
         }
         for(int k=0;k<3;k++){
            rgb[k]=(max * (rgb[k] - minrgb[k]) / (maxrgb[k] - minrgb[k]));
         }
         output[row][col].r=(char)rgb[0];
         output[row][col].g=(char)rgb[1];
         output[row][col].b=(char)rgb[2];
      }
   }
}
void writefile(struct pixel **output,int width, int height, FILE *outfile,char * type, int max){
   fprintf(outfile,"%s\n%d %d\n%d\n", type, width, height, 255);
   for(int i=0;i<width;i++){
      for(int j=0;j<height;j++){
         fwrite(&output[i][j].r, 1,1,outfile);
         fwrite(&output[i][j].g, 1,1,outfile);
         fwrite(&output[i][j].b, 1,1,outfile);
      }
   }
}
