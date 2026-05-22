#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <array>
using namespace std;

// ======================================================
// 世界盃歷史資料（1930-2022，共22屆）
// 分數: 7=冠軍 6=亞軍 5=第三名 4=第四名
//       3=八強  2=十六強 1=小組出局 0=未參賽/預選賽落敗
// ======================================================
const int WC_YEARS[] = {
    1930,1934,1938,1950,1954,1958,1962,1966,
    1970,1974,1978,1982,1986,1990,1994,1998,
    2002,2006,2010,2014,2018,2022
};
const int NUM_WC = 22;
const int WINDOW  = 10; // 最近10屆

struct Team { string name; vector<double> s; };

vector<Team> TEAMS = {
    {"Brazil",    {5,2,5,6,3,7,7,1,7,4,5,2,3,2,7,6,7,3,3,4,3,3}},
    {"Germany",   {0,5,2,0,7,4,3,6,5,7,2,6,6,7,3,3,6,5,5,7,1,1}},
    {"Argentina", {6,2,0,0,0,1,1,3,0,2,7,2,7,6,2,3,1,3,3,6,2,7}},
    {"France",    {1,2,3,0,1,5,0,1,0,1,1,4,5,0,0,7,1,6,1,3,7,6}},
    {"Italy",     {0,7,7,1,1,0,1,1,6,1,4,7,2,5,6,3,2,7,1,1,0,0}},
    {"Spain",     {3,3,0,4,1,0,1,1,0,1,1,2,3,2,3,1,3,2,7,1,2,3}},
};

// ======================================================
// 方法1：加權線性回歸（chap05 DP 啟發）
// 對每個時間點給指數衰減權重：越近的大會權重越大
// w[i] = r^(n-1-i)，r < 1，最新一點 w=1
// 加權最小二乘法：最小化 Σ w_i*(y_i - a*i - b)²
// ======================================================
pair<double,double> weighted_linreg(const vector<double>& y, double r = 0.85) {
    int n = y.size();
    double sw=0, swx=0, swy=0, swx2=0, swxy=0;
    for (int i = 0; i < n; ++i) {
        double wi = pow(r, n-1-i); // 越舊越小
        sw   += wi;
        swx  += wi * i;
        swy  += wi * y[i];
        swx2 += wi * i * i;
        swxy += wi * i * y[i];
    }
    double D = sw*swx2 - swx*swx;
    if (fabs(D) < 1e-9) return {0.0, swy/sw};
    double a = (sw*swxy - swx*swy) / D;
    double b = (swy - a*swx) / sw;
    return {a, b};
}

// ======================================================
// 方法2：多項式回歸（二次曲線）
// 配合 y = c₂x² + c₁x + c₀
// 用高斯消去法解 3×3 正規方程組
// ======================================================
array<double,3> poly_regress(const vector<double>& y) {
    int n = y.size();
    double s0=n, s1=0, s2=0, s3=0, s4=0;
    double t0=0, t1=0, t2=0;
    for (int i = 0; i < n; ++i) {
        double xi = i;
        s1 += xi;       s2 += xi*xi;
        s3 += xi*xi*xi; s4 += xi*xi*xi*xi;
        t0 += y[i];     t1 += xi*y[i]; t2 += xi*xi*y[i];
    }
    // 增廣矩陣 [A | b]
    double A[3][4] = {
        {s0, s1, s2, t0},
        {s1, s2, s3, t1},
        {s2, s3, s4, t2}
    };
    // 高斯-喬登消去
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col+1; row < 3; ++row)
            if (fabs(A[row][col]) > fabs(A[pivot][col])) pivot = row;
        for (int j = 0; j < 4; ++j) swap(A[col][j], A[pivot][j]);
        if (fabs(A[col][col]) < 1e-9) continue;
        double div = A[col][col];
        for (int j = col; j < 4; ++j) A[col][j] /= div;
        for (int row = 0; row < 3; ++row) {
            if (row == col) continue;
            double f = A[row][col];
            for (int j = col; j < 4; ++j) A[row][j] -= f * A[col][j];
        }
    }
    return {A[0][3], A[1][3], A[2][3]}; // c₀, c₁, c₂
}

// ======================================================
// 方法3：三分搜尋自動調參（chap06 二分搜尋啟發）
// 對指數平滑化的 α∈(0,1) 做三分搜尋，
// 最小化交叉驗證 MSE，自動找最佳 α
// ======================================================
double exp_smooth(const vector<double>& y, int n, double alpha) {
    double s = y[0];
    for (int i = 1; i < n; ++i) s = alpha*y[i] + (1.0-alpha)*s;
    return s;
}

double cv_mse(const vector<double>& y, double alpha, int cv = 3) {
    int n = y.size();
    double err = 0;
    for (int i = n-cv; i < n; ++i) {
        double e = exp_smooth(y, i, alpha) - y[i];
        err += e*e;
    }
    return err / cv;
}

// 三分搜尋：在 [lo, hi] 上找讓 cv_mse 最小的 α
double best_alpha(const vector<double>& y, int cv = 3) {
    double lo = 0.01, hi = 0.99;
    for (int i = 0; i < 60; ++i) {
        double m1 = lo + (hi-lo)/3;
        double m2 = hi - (hi-lo)/3;
        if (cv_mse(y, m1, cv) <= cv_mse(y, m2, cv))
            hi = m2;
        else
            lo = m1;
    }
    return (lo+hi)/2;
}

// ======================================================
// 工具：MSE
// ======================================================
double mse_vec(const vector<double>& pred, const vector<double>& actual) {
    double s = 0;
    for (int i = 0; i < (int)pred.size(); ++i) {
        double e = pred[i]-actual[i]; s += e*e;
    }
    return s / pred.size();
}

int main() {
    int from = NUM_WC - WINDOW;

    cout << fixed << setprecision(3);
    cout << "============================================================\n";
    cout << " 最近10屆 (" << WC_YEARS[from] << "–" << WC_YEARS[NUM_WC-1]
         << ") 三種新演算法預測測試\n";
    cout << "============================================================\n\n";

    // ── 方法說明 ──────────────────────────────────────────
    cout << "【方法說明】\n";
    cout << "  方法A  加權線性回歸  (chap05 DP啟發)  : 越近的大會權重越大(r=0.85衰減)\n";
    cout << "  方法B  多項式回歸    (二次曲線)        : 配合 y = c₂x² + c₁x + c₀\n";
    cout << "  方法C  自動調參指數平滑 (chap06啟發)   : 三分搜尋找最佳 α\n\n";

    // ── 各隊詳細結果 ──────────────────────────────────────
    cout << left << setw(11) << "隊伍";
    for (int i = from; i < NUM_WC; ++i)
        cout << right << setw(5) << WC_YEARS[i];
    cout << "\n" << string(63, '-') << "\n";

    vector<double> predA, predB, predC;
    vector<double> bestAlphas;

    for (auto& t : TEAMS) {
        vector<double> win(t.s.begin()+from, t.s.end()); // 最近10屆

        // 方法A：加權線性回歸
        auto [aA, bA] = weighted_linreg(win);
        double pA = max(0.0, aA * WINDOW + bA);

        // 方法B：多項式回歸（二次）
        auto [c0, c1, c2] = poly_regress(win);
        double xN = WINDOW;
        double pB = max(0.0, c2*xN*xN + c1*xN + c0);

        // 方法C：三分搜尋最佳 α
        double alpha = best_alpha(win, 3);
        double pC = max(0.0, exp_smooth(win, win.size(), alpha));
        bestAlphas.push_back(alpha);

        predA.push_back(pA);
        predB.push_back(pB);
        predC.push_back(pC);

        // 印出資料列
        cout << left << setw(11) << t.name;
        for (double v : win) cout << right << setw(5) << (int)v;
        cout << "\n";

        // 方法A
        cout << setw(11) << ""
             << "  A 加權線性回歸  y = " << aA << "*x + " << bA
             << "  → 2026預測: " << pA << "\n";

        // 方法B
        cout << setw(11) << ""
             << "  B 多項式回歸    y = " << c2 << "*x² + "
             << c1 << "*x + " << c0
             << "  → 2026預測: " << pB << "\n";

        // 方法C
        cout << setw(11) << ""
             << "  C 自動調參ES   最佳α = " << alpha
             << "  → 2026預測: " << pC << "\n\n";
    }

    // ── 2026 奪冠機率排名 ────────────────────────────────
    auto print_ranking = [&](const string& title, const vector<double>& preds) {
        double total = 0;
        for (double p : preds) total += p;
        if (total < 1e-9) total = 1;

        vector<pair<double,int>> rank;
        for (int i = 0; i < (int)preds.size(); ++i)
            rank.push_back({preds[i], i});
        sort(rank.rbegin(), rank.rend());

        cout << "  " << title << "\n";
        cout << "  " << string(38, '-') << "\n";
        int r = 1;
        for (auto [score, idx] : rank) {
            cout << "  " << setw(3) << r++ << ". "
                 << left << setw(10) << TEAMS[idx].name
                 << right << setw(8) << score
                 << "  " << setw(6) << score/total*100 << "%\n";
        }
        cout << "\n";
    };

    cout << "============================================================\n";
    cout << " 2026 世界盃奪冠機率預測（三種新方法）\n";
    cout << "============================================================\n\n";
    print_ranking("方法A：加權線性回歸 (r=0.85，chap05 DP啟發)", predA);
    print_ranking("方法B：多項式回歸（二次曲線）",                predB);
    print_ranking("方法C：三分搜尋自動調參指數平滑 (chap06啟發)", predC);

    // ── 三方法綜合平均 ──────────────────────────────────
    cout << "============================================================\n";
    cout << " 三方法綜合平均預測\n";
    cout << "============================================================\n";
    vector<double> avg(TEAMS.size());
    for (int i = 0; i < (int)TEAMS.size(); ++i)
        avg[i] = (predA[i] + predB[i] + predC[i]) / 3.0;
    print_ranking("綜合平均（A+B+C）/3", avg);

    // ── 最佳 α 彙整 ──────────────────────────────────────
    cout << "============================================================\n";
    cout << " 方法C 各隊自動調參結果（三分搜尋）\n";
    cout << "============================================================\n";
    cout << left << setw(12) << "隊伍" << right << setw(10) << "最佳α"
         << setw(16) << "CV-MSE(α*)" << "\n";
    cout << string(38, '-') << "\n";
    for (int i = 0; i < (int)TEAMS.size(); ++i) {
        vector<double> win(TEAMS[i].s.begin()+from, TEAMS[i].s.end());
        double a = bestAlphas[i];
        cout << left << setw(12) << TEAMS[i].name
             << right << setw(10) << a
             << setw(16) << cv_mse(win, a, 3) << "\n";
    }

    cout << "\n注意：本預測僅基於歷史分數統計，未考慮球員、賽程與戰術等因素。\n";
    return 0;
}
