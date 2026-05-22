#include <iostream>
#include <vector>
using namespace std;

// 指數平滑化: S_t = alpha * y_t + (1 - alpha) * S_{t-1}
// 下一步的預測值為 S_n（最後的平滑值）
vector<double> exponential_smoothing(const vector<double>& data, double alpha) {
    int n = data.size();
    vector<double> s(n);
    s[0] = data[0]; // 初始值為第一個資料點
    for (int t = 1; t < n; ++t) {
        s[t] = alpha * data[t] + (1.0 - alpha) * s[t - 1];
    }
    return s;
}

int main() {
    int n;
    double alpha;
    cin >> n >> alpha;

    vector<double> data(n);
    for (int i = 0; i < n; ++i) cin >> data[i];

    vector<double> s = exponential_smoothing(data, alpha);

    cout << "指數平滑化 (alpha = " << alpha << "):" << endl;
    for (int t = 0; t < n; ++t) {
        cout << "  S[" << t << "] = " << s[t] << endl;
    }

    // S_n 為下一步的預測值
    cout << "下一步的預測值: " << s.back() << endl;
}
