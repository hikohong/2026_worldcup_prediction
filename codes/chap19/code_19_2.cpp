#include <iostream>
#include <vector>
using namespace std;

// 計算最近 k 個值的簡單移動平均，並以此預測下一個值
vector<double> moving_average(const vector<double>& data, int k) {
    int n = data.size();
    vector<double> ma;
    for (int i = k - 1; i < n; ++i) {
        double sum = 0;
        for (int j = i - k + 1; j <= i; ++j) sum += data[j];
        ma.push_back(sum / k);
    }
    return ma;
}

int main() {
    int n, k;
    cin >> n >> k;

    vector<double> data(n);
    for (int i = 0; i < n; ++i) cin >> data[i];

    vector<double> ma = moving_average(data, k);

    cout << "移動平均 (視窗大小 k = " << k << "):" << endl;
    for (int i = 0; i < (int)ma.size(); ++i) {
        cout << "  區間 [" << i << ", " << i + k - 1 << "] 的平均: " << ma[i] << endl;
    }

    // 以最後一個移動平均值作為下一步的預測值
    cout << "下一步的預測值: " << ma.back() << endl;
}
