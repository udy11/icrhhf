/*
Plasma Dispersion Function and its Derivatives (64-bit)
Include pdispersion.h in your main code and compile with pdispersion.cpp and Faddeeva.cc
Sources for function's description:
  Kinetic Theory of Plasma Waves - Marco Brambilla, Chapter 4
  http://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node87.html
It is related to Faddeeva function, so check that too
*/

#include "Faddeeva.hh"

#define relerr 2.222e-16

// Plasma Dispersion Function (64-bit)
std::complex<double> pdispersion(std::complex<double> z) {
    double srpi = std::sqrt(4.0 * std::atan(1.0));    // Square Root of pi
    return -srpi * std::exp(-z * z) * (Faddeeva::erfi(z, relerr) - std::complex<double> (0.0, 1.0));
}

// Plasma Dispersion Function First Derivative (64-bit)
std::complex<double> dpdispersion(std::complex<double> z) {
    return -2.0 * (1.0 + z * pdispersion(z));
}

// Plasma Dispersion Function Second Derivative (64-bit)
std::complex<double> d2pdispersion(std::complex<double> z) {
    return -2.0 * (pdispersion(z) + z * dpdispersion(z));
}

