#include <stdio.h>
#include <stdlib.h>
#include <math.h>

long double func(long double A, long double B, long double C, long double x, long double y)
{
    return A * powl(x, 2) - B * x * y + C * powl(y, 2) + x - y;
}

void numerical_gradient(long double A, long double B, long double C, long double x, long double y, long double h, long double *fx, long double *fy)
{
    *fx = (func(A, B, C, x + h, y) - func(A, B, C, x, y)) / h;
    *fy = (func(A, B, C, x, y + h) - func(A, B, C, x, y)) / h;
}

void numerical_hessian(long double A, long double B, long double C, long double x, long double y, long double h, long double hessian[2][2])
{
    long double fx_x, fx_y, fy_x, fy_y;
    numerical_gradient(A, B, C, x + h, y, h, &fx_x, &fy_x);
    numerical_gradient(A, B, C, x, y, h, &fx_y, &fy_y);

    hessian[0][0] = (fx_x - fx_y) / h;
    hessian[0][1] = (fy_x - fy_y) / h;

    numerical_gradient(A, B, C, x, y + h, h, &fx_x, &fy_x);
    numerical_gradient(A, B, C, x, y, h, &fx_y, &fy_y);

    hessian[1][0] = (fx_x - fx_y) / h;
    hessian[1][1] = (fy_x - fy_y) / h;
}

long double dynamic_alpha(long double *xk, long double *gk, long double Hk[2][2])
{
    return (gk[0] * gk[0] + gk[1] * gk[1]) / (gk[0] * (Hk[0][0] * gk[0] + Hk[0][1] * gk[1]) + gk[1] * (Hk[1][0] * gk[0] + Hk[1][1] * gk[1]));
}

void newtons_method_numerical(long double A, long double B, long double C, long double x0, long double y0, long double h, int iteration, FILE *f)
{
    long double xk[2] = {x0, y0};
    long double gk[2], Hk[2][2], alpha;
    int i;
    for (i = 0; i < iteration; i++)
    {
        numerical_gradient(A, B, C, xk[0], xk[1], h, &gk[0], &gk[1]);
        numerical_hessian(A, B, C, xk[0], xk[1], h, Hk);
        alpha = dynamic_alpha(xk, gk, Hk);
        long double det = Hk[0][0] * Hk[1][1] - Hk[0][1] * Hk[1][0];
        long double inv_Hk[2][2] = {{Hk[1][1] / det, -Hk[0][1] / det}, {-Hk[1][0] / det, Hk[0][0] / det}};
        xk[0] = xk[0] - alpha * (inv_Hk[0][0] * gk[0] + inv_Hk[0][1] * gk[1]);
        xk[1] = xk[1] - alpha * (inv_Hk[1][0] * gk[0] + inv_Hk[1][1] * gk[1]);
        long double updated_gk[2];
        numerical_gradient(A, B, C, xk[0], xk[1], h, &updated_gk[0], &updated_gk[1]);
        long double dot_product = gk[0] * updated_gk[0] + gk[1] * updated_gk[1];
        printf("Iteration %d: x = %.10Lf, y = %.10Lf,\nDot product = %.10Lf, f(x, y) = %.10Lf\n", i + 1, xk[0], xk[1], dot_product, func(A, B, C, xk[0], xk[1]));
        fprintf(f, "%d;%.6Lf;%.6Lf;%.9Lf;%.9Lf\n", i + 1, xk[0], xk[1], dot_product, func(A, B, C, xk[0], xk[1]));
    }
}

int main()
{
    char *fName = "./T3_-5.txt";
    FILE *f = fopen(fName, "w");
    int iteration;
    long double A, B, C, x0, y0, h;
    newtons_method_numerical(A = 3, B = -2, C = 1, x0 = 3, y0 = 10, h = 1e-5, iteration = 150, f);
    fclose(f);
    return 0;
}