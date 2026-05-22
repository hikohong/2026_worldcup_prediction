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
const int WINDOW  = 10; // 直近10大会

struct Team {
    string name;
    vector<double> s;
};

// 最小二乗法 (x=0,1,...,n-1 の等間隔インデックス)
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

    int from = NUM_WC - WINDOW; // 直近10大会の開始インデックス

    cout << fixed << setprecision(3);
    cout << "================================================\n";
    cout << " 直近10大会 (";
    cout << WC_YEARS[from] << "–" << WC_YEARS[NUM_WC-1];
    cout << ") 線形回帰テスト\n";
    cout << "================================================\n\n";

    // ヘッダ
    cout << left << setw(11) << "チーム";
    for (int i = from; i < NUM_WC; ++i)
        cout << right << setw(5) << WC_YEARS[i];
    cout << right << setw(9) << "傾き(a)"
         << setw(10) << "切片(b)"
         << setw(12) << "2026予測" << "\n";
    cout << string(110, '-') << "\n";

    for (auto& t : teams) {
        // 直近 WINDOW 点を切り出す
        vector<double> win(t.s.begin() + from, t.s.end());
        auto [a, b] = linear_regression(win);
        double pred = a * WINDOW + b; // 次の点 (インデックス=WINDOW) を予測
        pred = max(0.0, pred);

        cout << left << setw(11) << t.name;
        for (double v : win)
            cout << right << setw(5) << (int)v;
        cout << right
             << setw(9) << a
             << setw(10) << b
             << setw(12) << pred << "\n";

        // 回帰直線と各大会の残差
        cout << setw(11) << "";
        cout << "  回帰直線: y = " << a << " * x + " << b << "\n";

        cout << setw(11) << "";
        cout << "  残差: ";
        for (int i = 0; i < WINDOW; ++i) {
            double fitted = a * i + b;
            double resid  = win[i] - fitted;
            cout << (resid >= 0 ? "+" : "") << setprecision(2) << resid;
            if (i < WINDOW-1) cout << "  ";
        }
        cout << setprecision(3) << "\n\n";
    }

    cout << "================================================\n";
    cout << " 2026 優勝スコア予測 (線形回帰のみ)\n";
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

    cout << setw(6) << "順位"
         << setw(12) << "チーム"
         << setw(12) << "予測スコア"
         << setw(10) << "優勝確率" << "\n";
    cout << string(40, '-') << "\n";
    int rank = 1;
    for (auto& [score, name] : ranking) {
        cout << setw(6) << rank++
             << setw(12) << name
             << setw(12) << score
             << setw(9) << score/total*100 << "%" << "\n";
    }
    cout << "\n注意: 線形回帰のみによる予測。他の統計手法と組み合わせると精度が向上します。\n";

    return 0;
}
