#include <stdio.h>
#include <stdlib.h>

double *CreateArray(int N, int D[]);

long int Len(int N, int D[]);

void Fill(double value, double *A, int N, int D[]);

double *Element(double *A, int N, int D[], int I[]);

int matrix_multiplication(double alpha, double* A, int ar, int ac, double* B, int br, int bc, double beta, double* C);