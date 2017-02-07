#include<stdio.h>
#include<stdlib.h>
int main (){
FILE *bigmat1=fopen("mat3046","w");
int n=3046;
fprintf(bigmat1,"%d\n",n);
for(int j=0;j<n;j++){
	for(int i=0;i<n;i++){
		fprintf(bigmat1,"1 ");
	}
	fprintf(bigmat1,"\n");
}
fclose(bigmat1);
} 
