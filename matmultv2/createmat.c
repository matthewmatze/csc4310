#include<stdio.h>
#include<stdlib.h>
int main (){
FILE *bigmat1=fopen("bigmat1","w");
FILE *bigmat2=fopen("bigmat2","w");

fprintf(bigmat1,"2048\n");
fprintf(bigmat2,"2048\n");
for(int j=0;j<2048;j++){
	for(int i=0;i<2048;i++){
		fprintf(bigmat1,"1 ");
		if(i==j){
			fprintf(bigmat2,"1 ");
		} else{
			fprintf(bigmat2,"0 ");
		}
	}
	fprintf(bigmat1,"\n");
	fprintf(bigmat2,"\n");
}
fclose(bigmat1);
fclose(bigmat2);
} 
