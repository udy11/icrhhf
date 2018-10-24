// Faddeeva functions supposed to be called as subroutines
// from fortran-95 program. To use a particular function
// from fortran, e.g. erfi, type in your f95 code:
//   call fdverfi(x, f)
// where x and f should be complex*16 in your f95 code

// Then this file should be complied to an object file as
//   gcc faddeeva_f95.c -c
// Then use the generated faddeeva_f95.o with your f95 code
//   gfortran main.f95 faddeeva_f95.o

// In case of any problem or slow performance, try to increase relerr

// For details on the implementation of original functions, check Faddeeva.cc
// and their website http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package

#include "Faddeeva.cc"

#define relerr 2.222e-16

void fdvw_(double complex *x, double complex *f)
{
    *f = Faddeeva_w(*x, relerr);
}

void fdverf_(double complex *x, double complex *f)
{
    *f = Faddeeva_erf(*x, relerr);
}

void fdverfc_(double complex *x, double complex *f)
{
    *f = Faddeeva_erfc(*x, relerr);
}

void fdverfcx_(double complex *x, double complex *f)
{
    *f = Faddeeva_erfcx(*x, relerr);
}

void fdverfi_(double complex *x, double complex *f)
{
    *f = Faddeeva_erfi(*x, relerr);
}

void fdvdawson_(double complex *x, double complex *f)
{
    *f = Faddeeva_Dawson(*x, relerr);
}

