#include "activation_functions.h"

double sigma(double x){
    return 1.0/(1.0 + exp(-x));
}

