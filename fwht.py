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
