! Plasma Dispersion Function and its Derivatives (64-bit)
! Needs faddeeva_f95.o
! Sources for function's description:
!   Kinetic Theory of Plasma Waves - Marco Brambilla, Chapter 4
!   http://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node87.html
! It is related to Faddeeva function, so check that too

! Plasma Dispersion Function (64-bit)
    subroutine pdispersion(z, pd)
    implicit none
    complex*16 z, pd, fi1
    real*8 srpi
    srpi = sqrt(4.0d0 * atan(1.0d0))    ! Square Root of pi
    call fdverfi(z, fi1)
    pd = -srpi * exp(-z * z) * (fi1 - (0.0d0, 1.0d0))
    endsubroutine

! Plasma Dispersion Function First Derivative (64-bit)
    subroutine dpdispersion(z, dpd)
    implicit none
    complex*16 z, dpd, pd
    call pdispersion(z, pd)
    dpd = -2.0d0 * (1.0d0 + z * pd)
    endsubroutine

! Plasma Dispersion Function Second Derivative (64-bit)
    subroutine d2pdispersion(z, d2pd)
    implicit none
    complex*16 z, d2pd, pd, dpd
    call pdispersion(z, pd)
    call dpdispersion(z, dpd)
    d2pd = -2.0d0 * (pd + z * dpd)
    endsubroutine

