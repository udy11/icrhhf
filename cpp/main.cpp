#include <iostream>
#include "pdispersion.h"

int main()
{
    std::cout.precision(15);
    std::complex<double> z (-1.0, -1.0);
    std::cout << pdispersion(z) << '\n';
    std::cout << dpdispersion(z) << '\n';
    std::cout << d2pdispersion(z) << '\n';
    return 0;
}

