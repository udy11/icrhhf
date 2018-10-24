    implicit none
    complex*16 z, p, p1, p2
    z = (1.0d0, -1.0d0)
    call pdispersion(z, p)
    call dpdispersion(z, p1)
    call d2pdispersion(z, p2)
    write(6,*) p
    write(6,*) p1
    write(6,*) p2
    end

