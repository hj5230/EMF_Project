import numpy as np

def func(A, B, C, x, y):
    return A * x**2 - B * x * y + C * y**2 + x - y

def gradient(A, B, C, x, y):
    return np.array([2 * A * x - B * y + 1, -B * x + 2 * C * y - 1])

def hessian(A, B, C):
    return np.array([[2 * A, -B], [-B, 2 * C]])

def dynamic_alpha(xk, gk, Hk):
    return (gk.T @ gk) / (gk.T @ Hk @ gk)

def newtons_method(A, B, C, x0, y0, itera):
    xk = np.array([x0, y0])
    Hk = hessian(A, B, C)
    
    for i in range(itera):
        gk = gradient(A, B, C, xk[0], xk[1])
        alpha = dynamic_alpha(xk, gk, Hk)
        xk = xk - alpha * np.linalg.inv(Hk) @ gk
        
        print(f"Iteration {i + 1}: x = {xk[0]}, y = {xk[1]}, f(x, y) = {func(A, B, C, xk[0], xk[1])}")


if __name__ == '__main__':
    A = 3
    B = -2
    C = 1
    x0 = 3
    y0 = 10
    iteration = 60

    newtons_method(A, B, C, x0, y0, iteration)
