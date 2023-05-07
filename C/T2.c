#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double func(double A, double B, double C, double x, double y)
{
    return A * pow(x, 2) - B * x * y + C * pow(y, 2) + x - y;
}

void numerical_gradient(double A, double B, double C, double x, double y, double h, double *fx, double *fy)
{
    *fx = (func(A, B, C, x + h, y) - func(A, B, C, x, y)) / h;
    *fy = (func(A, B, C, x, y + h) - func(A, B, C, x, y)) / h;
}

void numerical_hessian(double A, double B, double C, double x, double y, double h, double hessian[2][2])
{
    double fx_x, fx_y, fy_x, fy_y;
    numerical_gradient(A, B, C, x + h, y, h, &fx_x, &fy_x);
    numerical_gradient(A, B, C, x, y, h, &fx_y, &fy_y);

    hessian[0][0] = (fx_x - fx_y) / h;
    hessian[0][1] = (fy_x - fy_y) / h;

    numerical_gradient(A, B, C, x, y + h, h, &fx_x, &fy_x);
    numerical_gradient(A, B, C, x, y, h, &fx_y, &fy_y);

    hessian[1][0] = (fx_x - fx_y) / h;
    hessian[1][1] = (fy_x - fy_y) / h;
}

double dynamic_alpha(double *xk, double *gk, double Hk[2][2])
{
    return (gk[0] * gk[0] + gk[1] * gk[1]) / (gk[0] * (Hk[0][0] * gk[0] + Hk[0][1] * gk[1]) + gk[1] * (Hk[1][0] * gk[0] + Hk[1][1] * gk[1]));
}

void newtons_method_numerical(double A, double B, double C, double x0, double y0, int iteration, FILE* f)
{
    double xk[2] = {x0, y0};
    double gk[2], Hk[2][2], alpha;
    int i;

    for (i = 0; i < iteration; i++)
    {
        numerical_gradient(A, B, C, xk[0], xk[1], 1e-6, &gk[0], &gk[1]);
        numerical_hessian(A, B, C, xk[0], xk[1], 1e-6, Hk);
        alpha = dynamic_alpha(xk, gk, Hk);
        double det = Hk[0][0] * Hk[1][1] - Hk[0][1] * Hk[1][0];
        double inv_Hk[2][2] = {{Hk[1][1] / det, -Hk[0][1] / det}, {-Hk[1][0] / det, Hk[0][0] / det}};
        xk[0] = xk[0] - alpha * (inv_Hk[0][0] * gk[0] + inv_Hk[0][1] * gk[1]);
        xk[1] = xk[1] - alpha * (inv_Hk[1][0] * gk[0] + inv_Hk[1][1] * gk[1]);
        double updated_gk[2];
        numerical_gradient(A, B, C, xk[0], xk[1], 1e-6, &updated_gk[0], &updated_gk[1]);
        double dot_product = gk[0] * updated_gk[0] + gk[1] * updated_gk[1];
        printf("Iteration %d: x = %.10f, y = %.10f,\nDot product = %.10f, f(x, y) = %.10f\n", i + 1, xk[0], xk[1], dot_product, func(A, B, C, xk[0], xk[1]));
        fprintf(f, "%d;%.6f;%.6f;%.9f;%.9f\n", i + 1, xk[0], xk[1], dot_product, func(A, B, C, xk[0], xk[1]));
    }
}

int main()
{
    char* fName = "./T2_iterations.txt";
    FILE* f = fopen(fName, "w");
    int iteration;
    double A, B, C, x0, y0;
    newtons_method_numerical(A = 3, B = -2, C = 1, x0 = 3, y0 = 10, iteration = 100, f);
    fclose(f);
    return 0;
}