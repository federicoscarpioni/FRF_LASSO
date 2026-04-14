import numpy as np

def rational_poly4(
        x,
        a0,a1,a2,a3,a4,
        b1,b2,b3,b4,
        ):
    s = np.sqrt(1j * x)
    nominator = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4
    denominator = 1 + b1*s + b2*s**2 + b3*s**3 + b4*s**4
    return nominator / denominator

def rational_poly5(
        x,
        a0,a1,a2,a3,a4,a5,
        b1,b2,b3,b4,b5,
        ):
    s = np.sqrt(1j * x)
    nominator = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5
    denominator = 1 + b1*s + b2*s**2 + b3*s**3 + b4*s**4 + b5*s**5
    return nominator / denominator

def rational_poly6(
        x,
        a0,a1,a2,a3,a4,a5,a6,
        b1,b2,b3,b4,b5,b6,
        ):
    s = np.sqrt(1j * x)
    nominator = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5 + a6*s**6
    denominator = 1 + b1*s + b2*s**2 + b3*s**3 + b4*s**4 + b5*s**5 + b6*s**6
    return nominator / denominator

def rational_poly7(
        x,
        a0,a1,a2,a3,a4,a5,a6,a7,
        b1,b2,b3,b4,b5,b6,b7,
        ):
    s = np.sqrt(1j * x)
    nominator = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5 + a6*s**6 + a7*s**7
    denominator = 1 + b1*s + b2*s**2 + b3*s**3 + b4*s**4 + b5*s**5 + b6*s**6 + b7*s**7
    return nominator / denominator

def rational_poly8(
        x,
        a0,a1,a2,a3,a4,a5,a6,a7,a8,
        b1,b2,b3,b4,b5,b6,b7,b8,
        ):
    s = np.sqrt(1j * x)
    nominator = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5 + a6*s**6 + a7*s**7 + a8*s**8
    denominator = 1 + b1*s + b2*s**2 + b3*s**3 + b4*s**4 + b5*s**5 + b6*s**6 + b7*s**7 + b8*s**8
    return nominator / denominator

def rational_poly9(
        x,
        a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,
        b1,b2,b3,b4,b5,b6,b7,b8,b9,
        ):
    s = np.sqrt(1j * x)
    nominator = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5 + a6*s**6 + a7*s**7 + a8*s**8 + a9*s**9
    denominator = 1 + b1*s + b2*s**2 + b3*s**3 + b4*s**4 + b5*s**5 + b6*s**6 + b7*s**7 + b8*s**8 + b9*s**9
    return nominator / denominator