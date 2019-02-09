import numpy as np

# Fast Walsh-Hadamard Transform from Wikipedia
def fwht(a):
    """
    In-place Fast Walsh-Hadamard Transform of array a
    """
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j+h]
                a[j] = x + y
                a[j+h] = x - y
        h *= 2
# -- end function

# Manual convolution of vector a with vector b emulating the matrix-vector
# multiplication between vector a and a matrix created by rotating b one
# index at a time to get a len(b) x len(b) matrix.
def manual_convolve(a, b) :
    x = np.zeros(len(a))
    for i in range(len(b)) :
        x[i] = np.dot(a, np.roll(np.flip(b), i))
    return x
# -- end function
