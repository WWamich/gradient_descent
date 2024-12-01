#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <windows.h>
#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

double z(double x, double y) {
    return x * y * (6 - x - y);
}

std::vector<double> gradient(double x, double y) {
    double dz_dx = y * (6 - 2 * x - y);
    double dz_dy = x * (6 - x - 2 * y);
    return {dz_dx, dz_dy};
}

std::pair<std::vector<std::pair<double, double>>, int> gradient_descent(
        std::pair<double, double> start_point, double alpha, double tolerance, int max_iterations) {

    auto [x, y] = start_point;
    std::vector<std::pair<double, double>> path = {start_point};

    plt::figure();
    plt::plot({x}, {y}, "ro");
    plt::title("Градиентный спуск");
    plt::xlabel("x");
    plt::ylabel("y");

    for (int i = 0; i < max_iterations; ++i) {
        auto grad = gradient(x, y);
        double x_new = x - alpha * grad[0];
        double y_new = y - alpha * grad[1];

        path.emplace_back(x_new, y_new);

        plt::plot({x_new}, {y_new}, "bo");
        plt::pause(0.01);

        if (std::sqrt(std::pow(x_new - x, 2) + std::pow(y_new - y, 2)) < tolerance) {
            return {path, i + 1};
        }

        x = x_new;
        y = y_new;
    }

    return {path, max_iterations};
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::pair<double, double> initial_point = {1.0, 1.0};
    double alpha = 0.01;
    double tolerance = 1e-12;
    int max_iterations = 500;

    auto start_time = std::chrono::high_resolution_clock::now();

    auto [path, iterations] = gradient_descent(initial_point, alpha, tolerance, max_iterations);
    auto [x_ext, y_ext] = path.back();
    double z_ext = z(x_ext, y_ext);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Время выполнения: " << duration.count() << " секунд\n";
    std::cout << "Итоговая точка: (" << x_ext << ", " << y_ext << ")\n";
    std::cout << "Итоговое значение функции z(x, y): " << z_ext << "\n";
    std::cout << "Количество итераций: " << iterations << "\n";

    std::vector<double> x_vals, y_vals;
    for (const auto& point : path) {
        x_vals.push_back(point.first);
        y_vals.push_back(point.second);
    }

    plt::plot(x_vals, y_vals, "b-");
    plt::plot({x_vals[0]}, {y_vals[0]}, "ro");
    plt::plot({x_vals.back()}, {y_vals.back()}, "go");
    plt::legend();
    plt::show();

    return 0;
}
