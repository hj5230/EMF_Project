#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double func(double A, double B, double C, double x, double y) {
    return A * pow(x, 2) - B * x * y + C * pow(y, 2) + x - y;
}

void numerical_gradient(double A, double B, double C, double x, double y, double h, double *grad) {
    grad[0] = (func(A, B, C, x + h, y) - func(A, B, C, x, y)) / h;
    grad[1] = (func(A, B, C, x, y + h) - func(A, B, C, x, y)) / h;
}

void numerical_hessian(double A, double B, double C, double x, double y, double h, double *hess) {
    double grad_x[2], grad_y[2];
    numerical_gradient(A, B, C, x + h, y, h, grad_x);
    numerical_gradient(A, B, C, x, y, h, grad_y);
    hess[0] = (grad_x[0] - grad_y[0]) / h;
    hess[1] = (grad_x[1] - grad_y[1]) / h;
    hess[2] = hess[1];
    hess[3] = (grad_x[1] - grad_y[1]) / h;
}

double dynamic_alpha(double *xk, double *gk, double *Hk) {
    return (gk[0] * gk[0] + gk[1] * gk[1]) / (gk[0] * (Hk[0] * gk[0] + Hk[1] * gk[1]) + gk[1] * (Hk[2] * gk[0] + Hk[3] * gk[1]));
}

void newtons_method_numerical(double A, double B, double C, double x0, double y0, int iteration, FILE* f) {
    double xk[] = {x0, y0};
    double h = 1e-5;
    for (int i = 0; i < iteration; ++i) {
        double gk[2], Hk[4];
        numerical_gradient(A, B, C, xk[0], xk[1], h, gk);
        numerical_hessian(A, B, C, xk[0], xk[1], h, Hk);

        double alpha = dynamic_alpha(xk, gk, Hk);
        double invHk_times_gk[] = {(Hk[3] * gk[0] - Hk[1] * gk[1]) / (Hk[0] * Hk[3] - Hk[1] * Hk[2]), (-Hk[2] * gk[0] + Hk[0] * gk[1]) / (Hk[0] * Hk[3] - Hk[1] * Hk[2])};
        xk[0] -= alpha * invHk_times_gk[0];
        xk[1] -= alpha * invHk_times_gk[1];

        double fxy = func(A, B, C, xk[0], xk[1]);
        printf("Iteration %d:\n x = %.10f, y = %.10f, f(x, y) = %.10f\n", i + 1, xk[0], xk[1], fxy);
        fprintf(f, "%d;%.9f;%.9f;%.6f\n", i + 1, xk[0], xk[1], fxy);
    }
}

int main() {
    char* fName = "./T2_iterations.txt";
    FILE* f = fopen(fName, "w");
    int iteration;
    double A, B, C, x0, y0;
    newtons_method_numerical(A = 3, B = -2, C = 1, x0 = 3, y0 = 10, iteration = 100, f);
    fclose(f);
    return 0;
}
