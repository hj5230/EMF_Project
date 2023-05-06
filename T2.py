import numpy as np

def func(A, B, C, x, y):
    return A * x**2 - B * x * y + C * y**2 + x - y

def numerical_gradient(A, B, C, x, y, h=1e-6):
    fx = (func(A, B, C, x + h, y) - func(A, B, C, x, y)) / h
    fy = (func(A, B, C, x, y + h) - func(A, B, C, x, y)) / h
    return np.array([fx, fy])

def numerical_hessian(A, B, C, x, y, h=1e-6):
    fxx = (numerical_gradient(A, B, C, x + h, y)[0] - numerical_gradient(A, B, C, x, y)[0]) / h
    fxy = (numerical_gradient(A, B, C, x + h, y)[1] - numerical_gradient(A, B, C, x, y)[1]) / h
    fyx = (numerical_gradient(A, B, C, x, y + h)[0] - numerical_gradient(A, B, C, x, y)[0]) / h
    fyy = (numerical_gradient(A, B, C, x, y + h)[1] - numerical_gradient(A, B, C, x, y)[1]) / h
    return np.array([[fxx, fxy], [fyx, fyy]])

def dynamic_alpha(xk, gk, Hk):
    return (gk.T @ gk) / (gk.T @ Hk @ gk)

def newtons_method_numerical(A, B, C, x0, y0, iter=100):
    xk = np.array([x0, y0])
    
    for i in range(iter):
        gk = numerical_gradient(A, B, C, xk[0], xk[1])
        
        Hk = numerical_hessian(A, B, C, xk[0], xk[1])
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

    newtons_method_numerical(A, B, C, x0, y0, iteration)