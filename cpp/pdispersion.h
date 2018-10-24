// Include pdispersion.h in your main code and compile with pdispersion.cpp and Faddeeva.cc

#include <complex>

// Plasma Dispersion Function (64-bit)
std::complex<double> pdispersion(std::complex<double> z);

// Plasma Dispersion Function First Derivative (64-bit)
std::complex<double> dpdispersion(std::complex<double> z);

// Plasma Dispersion Function Second Derivative (64-bit)
std::complex<double> d2pdispersion(std::complex<double> z);

