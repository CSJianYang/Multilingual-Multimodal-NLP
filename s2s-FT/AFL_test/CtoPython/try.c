#include<stdlib.h>
#include<stdio.h>

int main() {
  char *output_sequence;

  int check_system = system("cd /home/li/CtoPython/PyDOC && python moduleload.py");
  if (check_system == -1)
    exit(1);
  
  FILE * out_fp = fopen("/home/li/CtoPython/docset/out.txt","r");
  
  fseek(out_fp , 0 , SEEK_END);
  int lSize = ftell (out_fp);
  rewind (out_fp);
  
  output_sequence = malloc(lSize);
  fread(output_sequence, 1, lSize, out_fp);
  
  fclose(out_fp);
  
  out_fp = fopen("/home/li/CtoPython/docset/out.txt","w"); 
  
  fclose(out_fp);
  
  printf("%s\n", output_sequence);
  return 0;
}
