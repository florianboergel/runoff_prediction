/*  Note:  For demonstration purposes only.  Depending on needs, other types
    might be used for indices and sizes, and the array type might be wrapped
    in an opaque struct rather than exposed as "int *".
*/
#include "ndarray.h"

//  Create an array with N dimensions with sizes specified in D.
double *CreateArray(int N, int D[])
{
    //  Calculate size needed.
    long int s = sizeof(double);
    for (int n = 0; n < N; ++n)
        s *= D[n];

    //  Allocate space.
    return (double*) malloc(s);
}



//  Create an array with N dimensions with sizes specified in D.
long int Len(int N, int D[])
{
    //  Calculate size needed.
    long int s = 1;
    for (int n = 0; n < N; ++n)
        s *= D[n];

    //  Allocate space.
    return s;
}

void Fill(double value, double *A, int N, int D[])
{
    long int L = Len(N,D);

    for (int l = 0; l < L; l++){
        A[l] = value;
    }
}

/*  Return a pointer to an element in an N-dimensional A array with sizes
    specified in D and indices to the particular element specified in I.
*/
double *Element(double *A, int N, int D[], int I[])
{
    //  Handle degenerate case.
    if (N == 0)
        return A;

    //  Map N-dimensional indices to one dimension.
    long int index = I[0];
    for (int n = 1; n < N; ++n)
        index = index * D[n] + I[n];

    //  Return address of element.
    return &A[index];
}


extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double
        *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c,
        int *ldc );

#define mm_success 0
#define mm_failure 1

// performs C = beta*C + alpha* A.B -> ac = br ! 
int matrix_multiplication(double alpha, double* A, int ar, int ac, double* B, int br, int bc, double beta, double* C)
{
    if (ac != br){
        printf("ERROR! %d != %d\n", ac, br);
        return mm_failure;
    } 
    
    dgemm_("n", "n", &bc, &ar, &br, &alpha, B, &bc, A, &br, &beta, C, &bc);
    return mm_success;
}
