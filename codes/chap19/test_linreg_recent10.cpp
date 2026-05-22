#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
using namespace std;

const int WC_YEARS[] = {
    1930,1934,1938,1950,1954,1958,1962,1966,
    1970,1974,1978,1982,1986,1990,1994,1998,
    2002,2006,2010,2014,2018,2022
};
const int NUM_WC = 22;
const int WINDOW  = 10; // 最近10屆

struct Team {
    string name;
    vector<double> s;
};

// 最小二乘法（x=0,1,...,n-1 的等距索引）
pair<double,double> linear_regression(const vector<double>& y) {
    int n = y.size();
    double sx=0, sy=0, sxy=0, sx2=0;
    for (int i = 0; i < n; ++i) {
        sx += i; sy += y[i]; sxy += i*y[i]; sx2 += i*i;
    }
    double D = n*sx2 - sx*sx;
    if (fabs(D) < 1e-9) return {0.0, sy/n};
    double a = (n*sxy - sx*sy) / D;
    double b = (sy - a*sx) / n;
    return {a, b};
}

int main() {
    vector<Team> teams = {
        {"Brazil",    {5,2,5,6,3,7,7,1,7,4,5,2,3,2,7,6,7,3,3,4,3,3}},
        {"Germany",   {0,5,2,0,7,4,3,6,5,7,2,6,6,7,3,3,6,5,5,7,1,1}},
        {"Argentina", {6,2,0,0,0,1,1,3,0,2,7,2,7,6,2,3,1,3,3,6,2,7}},
        {"France",    {1,2,3,0,1,5,0,1,0,1,1,4,5,0,0,7,1,6,1,3,7,6}},
        {"Italy",     {0,7,7,1,1,0,1,1,6,1,4,7,2,5,6,3,2,7,1,1,0,0}},
        {"Spain",     {3,3,0,4,1,0,1,1,0,1,1,2,3,2,3,1,3,2,7,1,2,3}},
    };

    int from = NUM_WC - WINDOW; // 最近10屆的起始索引

    cout << fixed << setprecision(3);
    cout << "================================================\n";
    cout << " 最近10屆 (";
    cout << WC_YEARS[from] << "–" << WC_YEARS[NUM_WC-1];
    cout << ") 線性回歸測試\n";
    cout << "================================================\n\n";

    // 表頭
    cout << left << setw(11) << "隊伍";
    for (int i = from; i < NUM_WC; ++i)
        cout << right << setw(5) << WC_YEARS[i];
    cout << right << setw(9) << "斜率(a)"
         << setw(10) << "截距(b)"
         << setw(12) << "2026預測" << "\n";
    cout << string(110, '-') << "\n";

    for (auto& t : teams) {
        // 截取最近 WINDOW 個資料點
        vector<double> win(t.s.begin() + from, t.s.end());
        auto [a, b] = linear_regression(win);
        double pred = a * WINDOW + b; // 預測下一點（索引=WINDOW）
        pred = max(0.0, pred);

        cout << left << setw(11) << t.name;
        for (double v : win)
            cout << right << setw(5) << (int)v;
        cout << right
             << setw(9) << a
             << setw(10) << b
             << setw(12) << pred << "\n";

        // 回歸直線與各屆殘差
        cout << setw(11) << "";
        cout << "  回歸直線: y = " << a << " * x + " << b << "\n";

        cout << setw(11) << "";
        cout << "  殘差: ";
        for (int i = 0; i < WINDOW; ++i) {
            double fitted = a * i + b;
            double resid  = win[i] - fitted;
            cout << (resid >= 0 ? "+" : "") << setprecision(2) << resid;
            if (i < WINDOW-1) cout << "  ";
        }
        cout << setprecision(3) << "\n\n";
    }

    cout << "================================================\n";
    cout << " 2026 奪冠分數預測（僅線性回歸）\n";
    cout << "================================================\n";

    vector<pair<double,string>> ranking;
    double total = 0;
    for (auto& t : teams) {
        vector<double> win(t.s.begin() + from, t.s.end());
        auto [a, b] = linear_regression(win);
        double pred = max(0.0, a * WINDOW + b);
        total += pred;
        ranking.push_back({pred, t.name});
    }
    sort(ranking.rbegin(), ranking.rend());

    cout << setw(6) << "排名"
         << setw(12) << "隊伍"
         << setw(12) << "預測分數"
         << setw(10) << "奪冠機率" << "\n";
    cout << string(40, '-') << "\n";
    int rank = 1;
    for (auto& [score, name] : ranking) {
        cout << setw(6) << rank++
             << setw(12) << name
             << setw(12) << score
             << setw(9) << score/total*100 << "%" << "\n";
    }
    cout << "\n注意：僅為線性回歸的預測結果，與其他統計方法結合可提高精度。\n";

    return 0;
}
