#include <omp.h>

#include "ndarray.h"
#include "activation_functions.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int main(int argc, char **argv)
{
    int my_seq;
    
    if (argc > 1){
        my_seq = atoi(argv[1]);
    }
    else{
        my_seq = 0;
    }

    const int num_threads = 192;

    const int n = 3;
     
    const int Nx_full = 222;
    const int Ny_full = 244;

    const int Nx = 222/n;
    const int Ny = 244/n;    
    
    //const int Nx = 36;
    //const int Ny = 40;

    //const int Nx = 110;
    //const int Ny = 121;    

    //const int Nxi = 9;
    //const int Neta = 9;

    const int Nxi = 7;
    const int Neta = 7;   

    //const int Nxi = 3;
    //const int Neta = 3;    

    //const int Ntau = 3;
    const int Ntau = 30;

    const int Nk = 4;

    const int Ns = 2;
    const int Nh = 9;

    const int Na = 512;
    const int Nb = 256;

    const int Nr = 97;


    // counting indices
    //int x, y, r, tau, k, xp, xi, yp, eta, j, s, h, hp;

    const int Nxy = Nx*Ny;  

    enum {
        g_i = 0, g_f, g_o, g_c,
        Ng
    };

    printf("Read in gates, weights and bias...\n");

    // initialize these from files
    #define size_g 5
    int shape_g[size_g] = { Ntau, Nx, Ny, Nh, Ng };
    double *g = CreateArray(size_g, shape_g);   

    char file_name[32];
    sprintf(file_name, "g0.bin.%d", my_seq);
    FILE *fp = fopen(file_name,"rb");
    fread(g, sizeof(double), Len(size_g, shape_g), fp);  
    fclose(fp);

    #define size_H 3
    int shape_H[size_H] = {Nx, Ny, Nh};
    double *H = CreateArray(size_H, shape_H);

    sprintf(file_name, "H0.bin.%d", my_seq);
    fp = fopen(file_name,"rb");
    fread(H, sizeof(double), Len(size_H, shape_H), fp);    
    fclose(fp);

    #define size_M 5
    int shape_M[size_M] = { Nxi, Neta, Nh, Nk, Ng };
    double *M = CreateArray(size_M, shape_M);      

    fp = fopen("M.bin","rb");
    fread(M, sizeof(double), Len(size_M, shape_M), fp);
    fclose(fp);

    #define size_N 5
    int shape_N[size_N] = { Nxi, Neta, Nh, Nh, Ng };
    double *N = CreateArray(size_N, shape_N); 

    fp = fopen("N.bin","rb");
    fread(N, sizeof(double), Len(size_N, shape_N), fp);
    fclose(fp);

    #define size_W1 4
    int shape_W1[size_W1] = {Na, Nx, Ny, Nh};
    double *W1 = CreateArray(size_W1, shape_W1);

    fp = fopen("W1.bin","rb");
    fread(W1, sizeof(double), Len(size_W1, shape_W1), fp);
    fclose(fp);

    #define size_W2 2
    int shape_W2[size_W2] = {Nb, Na};
    double *W2 = CreateArray(size_W2, shape_W2);

    fp = fopen("W2.bin","rb");
    fread(W2, sizeof(double), Len(size_W2, shape_W2), fp); 
    fclose(fp);

    #define size_W3 2
    int shape_W3[size_W3] = {Nr, Nb};
    double *W3 = CreateArray(size_W3, shape_W3);
    
    fp = fopen("W3.bin","rb");
    fread(W3, sizeof(double), Len(size_W3, shape_W3), fp);
    fclose(fp);

    // initialize rhos with bias
    #define size_rho1 1
    int shape_rho1[size_rho1] = {Na};
    double *rho1 = CreateArray(size_rho1, shape_rho1);

    fp = fopen("B1.bin","rb");
    fread(rho1, sizeof(double), Len(size_rho1, shape_rho1), fp); 
    fclose(fp);    

    #define size_rho2 1
    int shape_rho2[size_rho2] = {Nb};
    double *rho2 = CreateArray(size_rho2, shape_rho2);   

    fp = fopen("B2.bin","rb");
    fread(rho2, sizeof(double), Len(size_rho2, shape_rho2), fp);    
    fclose(fp);       


    // these are intitiaized in the following

    #define size_C 4
    int shape_C[size_C] = { Ntau, Nx, Ny, Nh};
    double *C = CreateArray(size_C, shape_C);    

    #define size_Lamda 9
    int shape_Lamda[size_Lamda] = { Ntau, Nx, Ny, Nxi, Neta, Ns, Nh, Ns, Nh };
    long int len_Lamda = Len(size_Lamda, shape_Lamda);
    double *Lamda = CreateArray(size_Lamda, shape_Lamda);

    #define size_lamda 8 
    int shape_lamda[size_lamda] = { Ntau, Nk, Nx, Ny, Nxi, Neta, Ns, Nh };
    long int len_lamda = Len(size_lamda, shape_lamda);
    double *lamda = CreateArray(size_lamda, shape_lamda);    

    #define size_chi 4 
    int shape_chi[size_chi] = { Nx, Ny, Ns, Nh };
    long int len_chi =  Len(size_chi, shape_chi);
    double** chi = (double**) malloc(num_threads * sizeof(double*));
    for (int th = 0; th < num_threads; th++)
    {
        chi[th] = CreateArray(size_chi, shape_chi);  
    }

    double** chi0 = (double**) malloc(num_threads * sizeof(double*));
    for (int th = 0; th < num_threads; th++)
    {
        chi0[th] = CreateArray(size_chi, shape_chi);  
    }

    double** chi_tmp = (double**) malloc(num_threads * sizeof(double*));
    for (int th = 0; th < num_threads; th++)
    {
        chi_tmp[th] = (double*) malloc(sizeof(double) * Ns*Nh);  
    }

    #define size_Xi 3 
    int shape_Xi[size_Xi] = { Nx, Ny, Nh };
    long int len_Xi =  Len(size_Xi, shape_Xi);
    double** Xi = (double**) malloc(num_threads * sizeof(double*));
    for (int th = 0; th < num_threads; th++)
    {
        Xi[th] = CreateArray(size_Xi, shape_Xi);  
    }    



    #define size_Omega 4
    int shape_Omega[size_Omega] = {Nr, Nx, Ny, Nh};
    double *Omega = CreateArray(size_Omega, shape_Omega);   

    #define size_omega 5
    int shape_omega[size_omega] = {Nk, Ntau, Nx, Ny, Nr};
    double *omega = CreateArray(size_omega, shape_omega);           

    omp_set_dynamic(0);     // Explicitly disable dynamic teams

    int Nxi2 = (Nxi-1)/2;
    int Neta2 = (Neta-1)/2;

    omp_set_num_threads(num_threads);

    printf("...done.\n");


    printf("Prepare Omega...\n");
    fflush(stdout);  

    const int Nxyh = Nx * Ny * Nh;

    // rho1[a] = W1[a, hxy] * H[hxy] + B1[a]
    matrix_multiplication(1.0, W1, Na, Nxyh, H, Nxyh, 1, 1.0, rho1);  // rho1 is initialized with B1, thus += (beta = 1.0)


    for (int a = 0; a < Na; a++){
        rho1[a] = MAX(rho1[a], 0.0); // apply ReLU
        //printf("%lf ", rho1[a]);
    }
    //printf("\n\n");

    // rho2[b] = W2[b,a] * rho1[a] + B2[b]
    matrix_multiplication(1.0, W2, Nb, Na, rho1, Na, 1, 1.0, rho2); // rho2 is initialized with B2, thus += (beta = 1.0)
    

    for (int a = 0; a < Na; a++){
        rho1[a] = (rho1[a] > 0.0) ? 1.0 : 0.0; // apply Heaviside
        //printf("%lf ", rho1[a]);
    }
    //printf("\n\n");

    for (int b = 0; b < Nb; b++){
        rho2[b] = (rho2[b] > 0.0) ? 1.0 : 0.0; // apply Heaviside
        //printf("%lf ", rho2[b]);
    }  
    //printf("\n\n");   
    fflush(stdout);

    // replace W's by products with rho's
    for (int b = 0; b < Nb; b++){
        for (int a = 0; a < Na; a++){
            *Element(W2, size_W2, shape_W2, (int []){b, a}) *= rho1[a];
        }
    }

    for (int r = 0; r < Nr; r++){
        for (int b = 0; b < Nb; b++){
            *Element(W3, size_W3, shape_W3, (int []){r, b}) *= rho2[b];
        }
    }


    #define size_tmp1 2
    int shape_tmp1[size_tmp1] = {Nxyh, Nb};
    double *tmp1 = CreateArray(size_tmp1, shape_tmp1);


    // tmp[b,xyh] = (W2 * rho1)[b, a].W1[a, xyh] 
    matrix_multiplication(1.0, W2, Nb, Na, W1, Na, Nxyh, 0.0, tmp1);

    // Omega[r,xyh] = (W3 * rho2)[r,b].tmp[b,xyh]
    matrix_multiplication(1.0, W3, Nr, Nb, tmp1, Nb, Nxyh, 0.0, Omega); 


    sprintf(file_name, "Omega.bin.%d", my_seq);
    fp = fopen(file_name,"wb");
    fwrite(Omega, sizeof(double), Len(size_Omega, shape_Omega), fp);
    fclose(fp);

    printf("...done.\n");
    printf("Prepare lamdas...\n");
    fflush(stdout); 

    for (long int l = 0; l < len_Lamda; l++){
        Lamda[l] = 0.0;
        if (l < len_lamda)  lamda[l] = 0.0;
    }

    for (int j = 0; j < Ntau; j++){   
        for (int xp = 0; xp < Nx; xp++){
            for (int yp = 0; yp < Ny; yp++){
                for (int h = 0; h < Nh; h++){
                    double I, F, O, C_now, C_1, tC;
                    double i, f, o, c, tc;

                    i = *Element(g, size_g, shape_g, (int []) {j, xp, yp, h, g_i}); 
                    I = sigma(i); 


                    f = *Element(g, size_g, shape_g, (int []) {j, xp, yp, h, g_f}); 
                    F = sigma(f); 

                    o = *Element(g, size_g, shape_g, (int []) {j, xp, yp, h, g_o}); 
                    O = sigma(o); 
                    
                    c = *Element(g, size_g, shape_g, (int []) {j, xp, yp, h, g_c});

                    tc = tanh(c);

                    C_now = I*tc;
                    if (j > 0){
                        C_1 = *Element(C, size_C, shape_C, (int []) {j-1, xp, yp, h});
                        C_now += F * C_1;
                    }
                    *Element(C, size_C, shape_C, (int []) {j, xp, yp, h}) = C_now;

                    tC = tanh(C_now);

                    // local elements of Lamda (midpoints of the convolution kernels)
                    *Element(Lamda, size_Lamda, shape_Lamda, (int []) {j, xp, yp, Nxi2, Neta2, 0, h, 0, h}) = F;
                    *Element(Lamda, size_Lamda, shape_Lamda, (int []) {j, xp, yp, Nxi2, Neta2, 1, h, 0, h}) = O * F * (1.0 - tC*tC) ;
        

                    // non-local lements of lamda and Lamda
                    for (int xi = 0; xi < Nxi; xi++){
                        int xis = xi - Nxi2;
                        if ((xp+xis < 0) || (xp+xis >= Nx)) continue;  
                        for (int eta = 0; eta < Neta; eta++){
                            int etas = eta - Neta2;
                            if ((yp+etas < 0) || (yp+etas >= Ny)) continue;  

                            for (int k = 0; k < Nk; k++){
                                double Mi = *Element(M, size_M, shape_M, (int []) {xi, eta, h, k, g_i});
                                double Mf = *Element(M, size_M, shape_M, (int []) {xi, eta, h, k, g_f});
                                double Mo = *Element(M, size_M, shape_M, (int []) {xi, eta, h, k, g_o});
                                double Mc = *Element(M, size_M, shape_M, (int []) {xi, eta, h, k, g_c});

                                double l = F * (1.0 - F) * C_1 * Mf + I * (1.0 - I) * tc * Mi + I * (1.0 - tc*tc) * Mc;
                                *Element(lamda, size_lamda, shape_lamda, (int []) {j, k, xp, yp, xi, eta, 0, h }) = l;
                                *Element(lamda, size_lamda, shape_lamda, (int []) {j, k, xp, yp, xi, eta, 1, h }) =  O * (1.0 - O) * tC * Mo +  O * (1.0 - tC*tC) * l;   
                            }          
                            
                            for (int hp = 0; hp < Nh; hp++){
                                // s = 0, s' = 1
                                double Ni = *Element(N, size_N, shape_N, (int []) {xi, eta, h, hp, g_i});
                                double Nf = *Element(N, size_N, shape_N, (int []) {xi, eta, h, hp, g_f});
                                double No = *Element(N, size_N, shape_N, (int []) {xi, eta, h, hp, g_o});
                                double Nc = *Element(N, size_N, shape_N, (int []) {xi, eta, h, hp, g_c});

                                double L = F * (1.0 - F) * C_1 * Nf + I * (1.0 - I) * tc * Ni + I * (1.0 - tc*tc) * Nc;
                                *Element(Lamda, size_Lamda, shape_Lamda, (int []) {j, xp, yp, xi, eta, 0, h, 1, hp}) = L;
                                *Element(Lamda, size_Lamda, shape_Lamda, (int []) {j, xp, yp, xi, eta, 1, h, 1, hp}) =   O * (1.0 - O) * tC * No + O * (1.0 - tC*tC) * L;
                            }
                        }
                    }
                }                    

            }
        } 
        printf("...done for j = %d\n", j);
        fflush(stdout);  

    }  

    printf("...done.\n");    

    sprintf(file_name, "lamda.bin.%d", my_seq);
    fp = fopen(file_name,"wb");
    fwrite(lamda, sizeof(double), Len(size_lamda, shape_lamda), fp);
    fclose(fp); 

    printf("Process omega map... \n");
    fflush(stdout);  

    // int shift(int xi){
    //    return 0;
    // }    

    int shift(int xi){
         return ((xi > 1) || (xi < -1)) ? 1 : 0;
    }

    #pragma omp parallel for
    for (int xy = 0; xy < Nxy; xy++){
        int x = xy / Ny;
        int y = xy % Ny;

        int th = omp_get_thread_num();

        for (int k=0; k < Nk; k++){
            for (int tau = 0; tau < Ntau; tau++)
            {      
                // initialize chi as lamda
                for (long int l = 0; l < len_chi; l++) 
                {
                    chi0[th][l] = 0.0; 
                    chi[th][l] = 0.0; 
                }

                for (int xi = 0; xi < Nxi; xi++){
                    int xis = shift(xi - Nxi2);
                    if ((x+xis < 0) || (x+xis >= Nx)) continue;
                    for (int eta = 0; eta < Neta; eta++){  
                        int etas = shift(eta - Neta2);
                        if ((y+etas < 0) || (y+etas >= Ny)) continue;
                        for (int h = 0; h < Nh; h++){
                            *Element(chi0[th], size_chi, shape_chi, (int []) {x+xis, y+etas, 0, h}) += *Element(lamda, size_lamda, shape_lamda, (int []) {tau, k, x, y, xi, eta, 0, h }) ; 
                            *Element(chi0[th], size_chi, shape_chi, (int []) {x+xis, y+etas, 1, h}) += *Element(lamda, size_lamda, shape_lamda, (int []) {tau, k, x, y, xi, eta, 1, h }) ;  
                        }
                    }                        
                }  

                // case when tau = Ntau-1 : no propagation
                if (tau == Ntau-1){
                    for (long int l = 0; l < len_chi; l++) 
                        chi[th][l] = chi0[th][l];    

                }

                // propagate chi from tau to Ntau with Lamda
                for (int j = tau; j < Ntau-1; j++){

                    //Lamda = fread(lamda.bin, (j+1)*len_Lamda, len_Lamda)

                    if(j > tau){
                        for (long int l = 0; l < len_chi; l++) 
                            chi0[th][l] = chi[th][l];    
                    }


                    for (int xp = MAX(0,x-(j-tau+1)*shift(Nxi2)); xp < MIN(Nx, x+(j-tau+1)*shift(Nxi2)+1); xp++){
                        for (int yp = MAX(0,y-(j-tau+1)*shift(Neta2)); yp < MIN(Ny,y+(j-tau+1)*shift(Neta2)+1); yp++){

                            for (int s = 0; s < Ns; s++){
                                for(int h = 0; h < Nh; h++){
                                    chi_tmp[th][s*Nh+h] = 0.0;
                                }
                            }

                            for (int xi = 0; xi < Nxi; xi++){
                                int xis = shift(xi - Nxi2);
                                if ((xp-xis < 0) || (xp-xis >= Nx)) continue;
                                for (int eta = 0; eta < Neta; eta++){
                                    int etas = shift(eta - Neta2);
                                    if ((yp-etas < 0) || (yp-etas >= Ny)) continue;

                                    // pointers to matrices and vectors
                                    double* A = Element(Lamda, size_Lamda, shape_Lamda, (int []) {j+1, xp, yp, xi, eta, 0, 0, 0, 0});
                                    double* B = Element(chi0[th], size_chi, shape_chi, (int []) {xp-xis, yp-etas, 0, 0});
                                    double* C = chi_tmp[th];


                                    int dim1 = Ns * Nh; 
                                    int dim2 = 1;

                                    // A is dim1 x dim1, B is dim1 x dim2, C is dim1 x dim2

                                    double alpha = 1.0;
                                    double beta = 1.0; // for +=

                                    // calculate C += A*B
                                    matrix_multiplication(alpha, A, dim1, dim1, B, dim1, dim2, beta, C);
                                    //chi[xp,yp] += Lamda[j, xp, xi, yp, eta].chi0[xp-xi,yp-eta]

                                    
                                }
                            }

                            for (int s = 0; s < Ns; s++){
                                for(int h = 0; h < Nh; h++){
                                    *Element(chi[th], size_chi, shape_chi, (int []) {xp, yp, s, h}) = chi_tmp[th][s*Nh+h];
                                }
                            }
                            //printf("%lf ", *Element(chi[th], size_chi, shape_chi, (int []) {xp, yp, 0, 0}));
                        }
                        //printf("\n");
                    }

                }



                for (int xp = 0; xp < Nx; xp++){
                    for (int yp = 0; yp < Ny; yp++){
                        for(int h = 0; h < Nh; h++){
                            *Element(Xi[th], size_Xi, shape_Xi, (int []) {xp, yp, h}) = *Element(chi[th], size_chi, shape_chi, (int []) {xp, yp, 1, h});
                            //printf("% 5.3e ", *Element(Xi[th], size_Xi, shape_Xi, (int []) {xp, yp, h}));
                        }
                    }
                }

                
                int dim1 = Nr;
                int dim2 = Nx * Ny * Nh; 
                int dim3 = 1;
                matrix_multiplication(1.0, Omega, dim1, dim2, Xi[th], dim2, dim3, 0.0, Element(omega, size_omega, shape_omega, (int []) {k, tau, x, y, 0}));
                // Element(omega, size_omega, shape_omega, (int []) {k, tau, x, y, r}) = Omega[r, x', y', h'] * Xi[x', y', h']
                // store chi

                //printf("%-5.2e ", *Element(omega, size_omega, shape_omega, (int []) {k, tau, x, y, 0}));

            }

        }
        printf("...done for x = %d, y = %d.\n", x, y);
        fflush(stdout);  
    }

    printf("\n");
    printf("##################################\n");
    printf("\n");

    sprintf(file_name, "omega.bin.%d", my_seq);
    fp = fopen(file_name,"wb");
    fwrite(omega, sizeof(double), Len(size_omega, shape_omega), fp);
    fclose(fp);

    return 0;

}