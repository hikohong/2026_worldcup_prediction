#include <iostream>
#include <vector>
#include <cmath>
#include <string>
using namespace std;

// ======== 受測函式（與 code_19_1〜4 相同實作）========

pair<double,double> linear_regression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sx=0, sy=0, sxy=0, sx2=0;
    for (int i = 0; i < n; ++i) {
        sx += x[i]; sy += y[i]; sxy += x[i]*y[i]; sx2 += x[i]*x[i];
    }
    double D = n*sx2 - sx*sx;
    if (fabs(D) < 1e-9) return {0.0, sy/n};
    double a = (n*sxy - sx*sy) / D;
    double b = (sy - a*sx) / n;
    return {a, b};
}

// 視窗式線性回歸：使用 s[n-window..n-1] 的 window 個點預測下一點
double pred_linear_windowed(const vector<double>& s, int n, int window) {
    int from = max(0, n - window);
    int cnt  = n - from;
    double sx=0, sy=0, sxy=0, sx2=0;
    for (int i = from; i < n; ++i) {
        double xi = i - from;
        sx += xi; sy += s[i]; sxy += xi*s[i]; sx2 += xi*xi;
    }
    double D = cnt*sx2 - sx*sx;
    if (fabs(D) < 1e-9) return sy / cnt;
    double a = (cnt*sxy - sx*sy) / D;
    double b = (sy - a*sx) / cnt;
    return a*cnt + b;
}

vector<double> moving_average(const vector<double>& data, int k) {
    int n = data.size();
    vector<double> ma;
    for (int i = k-1; i < n; ++i) {
        double sum = 0;
        for (int j = i-k+1; j <= i; ++j) sum += data[j];
        ma.push_back(sum / k);
    }
    return ma;
}

vector<double> exponential_smoothing(const vector<double>& data, double alpha) {
    int n = data.size();
    vector<double> s(n);
    s[0] = data[0];
    for (int t = 1; t < n; ++t) s[t] = alpha*data[t] + (1.0-alpha)*s[t-1];
    return s;
}

double mse(const vector<double>& pred, const vector<double>& actual) {
    double sum = 0;
    for (int i = 0; i < (int)pred.size(); ++i) {
        double e = pred[i] - actual[i];
        sum += e*e;
    }
    return sum / pred.size();
}

// ======== 測試框架 ========

int pass_count = 0, fail_count = 0;

void check(const string& name, bool ok) {
    if (ok) {
        cout << "  [通過] " << name << "\n";
        ++pass_count;
    } else {
        cout << "  [失敗] " << name << "\n";
        ++fail_count;
    }
}

bool near(double a, double b, double eps = 1e-6) {
    return fabs(a - b) < eps;
}

// ======== 線性回歸測試 ========

void test_linear_regression() {
    cout << "\n[線性回歸 測試]\n";

    // TC1: 完全線性 y = 2x + 1
    {
        vector<double> x = {0,1,2,3,4};
        vector<double> y = {1,3,5,7,9};
        auto [a, b] = linear_regression(x, y);
        check("TC1: 斜率 a = 2.0",       near(a, 2.0));
        check("TC1: 截距 b = 1.0",       near(b, 1.0));
        double pred = a*5 + b;
        check("TC1: x=5 的預測值 = 11.0", near(pred, 11.0));
    }

    // TC2: 水平線 y = 5（斜率為零）
    {
        vector<double> x = {0,1,2,3,4};
        vector<double> y = {5,5,5,5,5};
        auto [a, b] = linear_regression(x, y);
        check("TC2: 水平線 a = 0.0",  near(a, 0.0));
        check("TC2: 水平線 b = 5.0",  near(b, 5.0));
    }

    // TC3: 單回歸 MSE 為零（完全擬合）
    {
        vector<double> x = {1,2,3};
        vector<double> y = {3,5,7}; // y = 2x + 1
        auto [a, b] = linear_regression(x, y);
        vector<double> fitted;
        for (double xi : x) fitted.push_back(a*xi + b);
        check("TC3: 完全線性資料的 MSE = 0", near(mse(fitted, y), 0.0, 1e-9));
    }

    // TC4: 兩點（斜率確定）
    {
        vector<double> x = {0, 4};
        vector<double> y = {1, 9}; // a=2, b=1
        auto [a, b] = linear_regression(x, y);
        check("TC4: 兩點回歸 a = 2.0", near(a, 2.0));
        check("TC4: 兩點回歸 b = 1.0", near(b, 1.0));
    }

    // TC5: 負斜率 y = -3x + 10
    {
        vector<double> x = {0,1,2,3};
        vector<double> y = {10,7,4,1};
        auto [a, b] = linear_regression(x, y);
        check("TC5: 負斜率 a = -3.0",  near(a, -3.0));
        check("TC5: 負斜率 b = 10.0",  near(b, 10.0));
    }

    // TC6: 視窗式回歸 — window=3 僅使用末尾3點
    // 資料: [0,0,0,10,20,30]（末尾3點: 10,20,30 → 斜率=10）
    // 下一預測值 = 40
    {
        vector<double> s = {0,0,0,10,20,30};
        int n = s.size();
        double pred = pred_linear_windowed(s, n, 3);
        check("TC6: window=3 僅用末尾3點回歸，預測=40", near(pred, 40.0));
    }

    // TC7: window >= n → 等同使用全部資料點
    {
        vector<double> s = {1,3,5,7,9}; // y=2x+1
        int n = s.size();
        double pred_full   = pred_linear_windowed(s, n, 100); // window>n
        double pred_normal = pred_linear_windowed(s, n, n);   // window=n
        check("TC7: window>=n 等同使用全部資料點", near(pred_full, pred_normal));
        check("TC7: 完全線性資料 window=n 的預測=11", near(pred_full, 11.0));
    }

    // TC8: window=10 能偵測趨勢反轉
    // 前12點：上升，後10點：下降 → window=10 捕捉到下降趨勢
    {
        vector<double> s = {1,2,3,4,5,6,7,8,9,10,11,12, // 上升
                            11,10,9,8,7,6,5,4,3,2};      // 下降（10點）
        int n = s.size(); // 22
        double pred = pred_linear_windowed(s, n, 10);
        check("TC8: window=10 捕捉下降趨勢，pred < 2", pred < 2.0);
        double pred_all = pred_linear_windowed(s, n, 22); // 全部點（window=n）
        check("TC8: window=22（全部點）的預測大於 window=10", pred_all > pred);
    }
}

// ======== 移動平均測試 ========

void test_moving_average() {
    cout << "\n[移動平均 測試]\n";

    // TC1: 常數數列 → 所有 MA 值相同
    {
        vector<double> data = {5,5,5,5,5};
        auto ma = moving_average(data, 3);
        bool ok = true;
        for (double v : ma) if (!near(v, 5.0)) ok = false;
        check("TC1: 常數數列的 MA = 常數",  ok);
    }

    // TC2: 等差數列 [10,20,30,40,50], k=3
    {
        vector<double> data = {10,20,30,40,50};
        auto ma = moving_average(data, 3);
        // 預期值: 20, 30, 40
        check("TC2: MA[0] = 20.0", near(ma[0], 20.0));
        check("TC2: MA[1] = 30.0", near(ma[1], 30.0));
        check("TC2: MA[2] = 40.0", near(ma[2], 40.0));
        check("TC2: MA 的元素數 = n-k+1", ma.size() == 3u);
    }

    // TC3: k=1 → MA 等於原始資料
    {
        vector<double> data = {3,1,4,1,5};
        auto ma = moving_average(data, 1);
        bool ok = true;
        for (int i = 0; i < (int)data.size(); ++i) if (!near(ma[i], data[i])) ok = false;
        check("TC3: k=1 的 MA = 原始資料", ok);
    }

    // TC4: k=n → 只有一個全體平均值
    {
        vector<double> data = {2,4,6,8};
        auto ma = moving_average(data, 4);
        check("TC4: k=n 的 MA 元素數 = 1",  ma.size() == 1u);
        check("TC4: k=n 的 MA 值 = 全體平均", near(ma[0], 5.0));
    }

    // TC5: MSE（常數數列的 MA 之 MSE = 0）
    {
        vector<double> data = {7,7,7,7,7};
        auto ma = moving_average(data, 3);
        vector<double> actual(ma.size(), 7.0);
        check("TC5: 常數數列 MA 的 MSE = 0", near(mse(ma, actual), 0.0, 1e-9));
    }
}

// ======== 指數平滑化測試 ========

void test_exponential_smoothing() {
    cout << "\n[指數平滑化 測試]\n";

    // TC1: alpha=1.0 → S[t] = y[t]（僅使用最新值）
    {
        vector<double> data = {3,7,2,9,4};
        auto s = exponential_smoothing(data, 1.0);
        bool ok = true;
        for (int i = 0; i < (int)data.size(); ++i) if (!near(s[i], data[i])) ok = false;
        check("TC1: alpha=1.0 時 S[t]=y[t]", ok);
    }

    // TC2: alpha=0.0 → S[t] = S[0] = y[0]（初始值固定）
    {
        vector<double> data = {5,10,15,20};
        auto s = exponential_smoothing(data, 0.0);
        bool ok = true;
        for (double v : s) if (!near(v, 5.0)) ok = false;
        check("TC2: alpha=0.0 時 S[t]=y[0]", ok);
    }

    // TC3: 手動計算驗證 alpha=0.5, data=[10,20]
    //   S[0]=10, S[1]=0.5*20 + 0.5*10 = 15
    {
        vector<double> data = {10, 20};
        auto s = exponential_smoothing(data, 0.5);
        check("TC3: S[0] = 10.0", near(s[0], 10.0));
        check("TC3: S[1] = 15.0", near(s[1], 15.0));
    }

    // TC4: 常數數列 → 所有 S[t] 相同
    {
        vector<double> data = {4,4,4,4,4};
        auto s = exponential_smoothing(data, 0.3);
        bool ok = true;
        for (double v : s) if (!near(v, 4.0)) ok = false;
        check("TC4: 常數數列的 ES 為常數", ok);
    }

    // TC5: 手動計算驗證 alpha=0.3, data=[10,20,15]
    //   S[0]=10
    //   S[1]=0.3*20 + 0.7*10 = 6 + 7 = 13
    //   S[2]=0.3*15 + 0.7*13 = 4.5 + 9.1 = 13.6
    {
        vector<double> data = {10, 20, 15};
        auto s = exponential_smoothing(data, 0.3);
        check("TC5: S[0] = 10.0",  near(s[0], 10.0));
        check("TC5: S[1] = 13.0",  near(s[1], 13.0));
        check("TC5: S[2] = 13.6",  near(s[2], 13.6, 1e-6));
    }

    // TC6: MSE（常數數列的 ES 之 MSE = 0）
    {
        vector<double> data = {6,6,6,6,6};
        auto s = exponential_smoothing(data, 0.4);
        check("TC6: 常數數列 ES 的 MSE = 0", near(mse(s, data), 0.0, 1e-9));
    }
}

// ======== MSE 函式測試 ========

void test_mse() {
    cout << "\n[MSE 函式測試]\n";

    check("TC1: 完全一致時 MSE = 0",
        near(mse({1,2,3}, {1,2,3}), 0.0));

    check("TC2: 全部差值=2 時 MSE = 4",
        near(mse({3,4,5}, {1,2,3}), 4.0));

    check("TC3: 單點 MSE = 誤差的平方",
        near(mse({5}, {3}), 4.0));

    check("TC4: 對稱性 MSE(a,b) == MSE(b,a)",
        near(mse({1,3,5}, {2,4,6}), mse({2,4,6}, {1,3,5})));
}

// ======== 主程式 ========

int main() {
    cout << "================================================\n";
    cout << "  第19章 回歸測試\n";
    cout << "================================================\n";

    test_linear_regression();
    test_moving_average();
    test_exponential_smoothing();
    test_mse();

    cout << "\n================================================\n";
    cout << "  結果: " << pass_count << " 通過 / "
         << fail_count << " 失敗\n";
    cout << "================================================\n";

    return fail_count > 0 ? 1 : 0;
}
