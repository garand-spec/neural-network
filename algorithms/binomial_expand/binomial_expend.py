from math import comb

def binomial_expend(a, b ,n):
    rseult = 0

    for k in range(n + 1):
        result += comb(n, k) * (a ** (n-k)) * (b ** k)

    return result