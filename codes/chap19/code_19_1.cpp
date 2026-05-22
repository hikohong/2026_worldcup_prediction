#include <iostream>
#include <vector>
using namespace std;

// 用最小二乘法求回歸直線 y = a*x + b 的係數 (a, b)
pair<double, double> linear_regression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < n; ++i) {
        sum_x  += x[i];
        sum_y  += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }
    double denom = n * sum_x2 - sum_x * sum_x;
    double a = (n * sum_xy - sum_x * sum_y) / denom;
    double b = (sum_y - a * sum_x) / n;
    return {a, b};
}

int main() {
    // 輸入資料點數與欲預測的 x 值
    int n;
    double future_x;
    cin >> n >> future_x;

    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) cin >> x[i] >> y[i];

    // 以線性回歸求係數
    auto [a, b] = linear_regression(x, y);

    // 預測未來值
    double predicted = a * future_x + b;

    cout << "回歸直線: y = " << a << " * x + " << b << endl;
    cout << "x = " << future_x << " 時的預測值: " << predicted << endl;
}
