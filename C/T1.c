#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void func(double A, double B, double C, double x, double y, double *result) {
    *result = A * pow(x, 2) - B * x * y + C * pow(y, 2) + x - y;
}

void gradient(double A, double B, double C, double x, double y, double *g) {
    g[0] = 2 * A * x - B * y + 1;
    g[1] = -B * x + 2 * C * y - 1;
}

void hessian(double A, double B, double C, double *H) {
    H[0] = 2 * A;
    H[1] = -B;
    H[2] = -B;
    H[3] = 2 * C;
}

double dynamic_alpha(double *xk, double *gk, double *Hk) {
    return (gk[0] * gk[0] + gk[1] * gk[1]) / (gk[0] * (Hk[0] * gk[0] + Hk[1] * gk[1]) + gk[1] * (Hk[2] * gk[0] + Hk[3] * gk[1]));
}

void newtons_method(double A, double B, double C, double x0, double y0, int iteration, FILE* f) {
    double xk[] = {x0, y0};
    double Hk[4];
    hessian(A, B, C, Hk);
    for (int i = 0; i < iteration; ++i) {
        double gk[2];
        gradient(A, B, C, xk[0], xk[1], gk);
        double alpha = dynamic_alpha(xk, gk, Hk);
        double invHk_times_gk[] = {(Hk[3] * gk[0] - Hk[1] * gk[1]) / (Hk[0] * Hk[3] - Hk[1] * Hk[2]), (-Hk[2] * gk[0] + Hk[0] * gk[1]) / (Hk[0] * Hk[3] - Hk[1] * Hk[2])};
        xk[0] -= alpha * invHk_times_gk[0];
        xk[1] -= alpha * invHk_times_gk[1];
        double fxy;
        func(A, B, C, xk[0], xk[1], &fxy);
        printf("Iteration %d:\n x = %.10f, y = %.10f, f(x, y) = %.10f\n", i + 1, xk[0], xk[1], fxy);
        fprintf(f, "%d;%.6f;%.6f;%.9f\n", i + 1, xk[0], xk[1], fxy);
    }
}

int main() {
    char* fName = "./T1_iterations.txt";
    FILE* f = fopen(fName, "w");
    int iteration;
    double A, B, C, x0, y0;
    newtons_method(A = 3, B = -2, C = 1, x0 = 3, y0 = 10, iteration = 100, f);
    fclose(f);
    return 0;
}
