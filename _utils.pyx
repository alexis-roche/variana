# -*- Mode: Python -*-  

__version__ = '0.1'

import numpy as np
cimport numpy as np
cimport cython


cdef set_offsets(size_t* off, np.ndarray x, unsigned int bytesize):
    cdef unsigned int i
    for i in range(x.ndim):
        off[i] = x.strides[i] / bytesize

    
@cython.boundscheck(False)
def _sdot(np.ndarray[double, ndim=3] A, np.ndarray[double, ndim=3] B):
    """
    A (n, p, r)
    B (n, r, q)
    => C (n, p, q)
    """
    cdef size_t n = A.shape[0]
    cdef size_t p = A.shape[1]
    cdef size_t r = A.shape[2]
    cdef size_t q = B.shape[2]
    cdef double* a 
    cdef double* b
    cdef double* c
    cdef size_t[3] off_a
    cdef size_t[3] off_b
    cdef size_t[3] off_c
    cdef double aux
    cdef unsigned int t   
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef size_t off_a0_t = 0
    cdef size_t off_b0_t = 0
    cdef size_t off_c0_t = 0
    cdef size_t off_a1_i = 0
    cdef size_t off_c1_i = 0
    cdef size_t off_b2_j = 0
    cdef size_t off_c2_j = 0
    
    cdef np.ndarray C = np.zeros([n, p, q])
    
    set_offsets(off_a, A, sizeof(double))
    set_offsets(off_b, B, sizeof(double))
    set_offsets(off_c, C, sizeof(double))
    
    if B.shape[0] != n or B.shape[1] != r:
        raise ValueError('Inconsistent shape for B array')

    for t in range(n):
        off_a1_i = 0 
        off_c1_i = 0

        for i in range(p):
            off_b2_j = 0
            off_c2_j = 0

            for j in range(q):
                a = <double*>A.data + off_a0_t + off_a1_i 
                b = <double*>B.data + off_b0_t + off_b2_j 
                c = <double*>C.data + off_c0_t + off_c1_i + off_c2_j 
                aux = 0

                for k in range(r):
                    aux += a[0] * b[0]
                    a += off_a[2] # a_tik
                    b += off_b[1] # b_tkj

                c[0] = aux
                off_b2_j += off_b[2]
                off_c2_j += off_c[2]
                
            off_a1_i += off_a[1]
            off_c1_i += off_c[1]

        off_a0_t += off_a[0]
        off_b0_t += off_b[0]
        off_c0_t += off_c[0]

    return C
    
    



