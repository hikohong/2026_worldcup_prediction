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
// 歷史資料
// ======================================================
const int WC_YEARS[] = {
    1930,1934,1938,1950,1954,1958,1962,1966,
    1970,1974,1978,1982,1986,1990,1994,1998,
    2002,2006,2010,2014,2018,2022
};
const int NUM_WC = 22;
const int TEST_START = 12; // 回測從第13屆(1986)開始，保證至少有12筆訓練資料

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
// 各預測函式（以前 n 筆資料預測第 n+1 筆）
// ======================================================

// 原方法：窗口式線性回歸
double pred_linreg(const vector<double>& s, int n, int window = 10) {
    int from = max(0, n - window);
    int cnt = n - from;
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

// 原方法：移動平均
double pred_ma(const vector<double>& s, int n, int k) {
    int from = max(0, n - k);
    double sum = 0;
    for (int i = from; i < n; ++i) sum += s[i];
    return sum / (n - from);
}

// 原方法：指數平滑化（固定 α）
double pred_exp(const vector<double>& s, int n, double alpha) {
    double sm = s[0];
    for (int i = 1; i < n; ++i) sm = alpha*s[i] + (1.0-alpha)*sm;
    return sm;
}

// 新方法A：加權線性回歸（指數衰減權重）
double pred_wlinreg(const vector<double>& s, int n, int window = 10, double r = 0.85) {
    int from = max(0, n - window);
    int cnt = n - from;
    double sw=0, swx=0, swy=0, swx2=0, swxy=0;
    for (int i = from; i < n; ++i) {
        double xi = i - from;
        double wi = pow(r, cnt-1-(i-from)); // 最新點權重最大
        sw += wi; swx += wi*xi; swy += wi*s[i];
        swx2 += wi*xi*xi; swxy += wi*xi*s[i];
    }
    double D = sw*swx2 - swx*swx;
    if (fabs(D) < 1e-9) return swy/sw;
    double a = (sw*swxy - swx*swy) / D;
    double b = (swy - a*swx) / sw;
    return a*cnt + b;
}

// 新方法B：多項式回歸（二次）
double pred_poly(const vector<double>& s, int n, int window = 10) {
    int from = max(0, n - window);
    int cnt = n - from;
    double s0=cnt, s1=0, s2=0, s3=0, s4=0;
    double t0=0, t1=0, t2=0;
    for (int i = from; i < n; ++i) {
        double xi = i - from;
        s1 += xi; s2 += xi*xi; s3 += xi*xi*xi; s4 += xi*xi*xi*xi;
        t0 += s[i]; t1 += xi*s[i]; t2 += xi*xi*s[i];
    }
    double A[3][4] = {
        {s0,s1,s2,t0},{s1,s2,s3,t1},{s2,s3,s4,t2}
    };
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
            for (int j = col; j < 4; ++j) A[row][j] -= f*A[col][j];
        }
    }
    double c0=A[0][3], c1=A[1][3], c2=A[2][3];
    double xN = cnt;
    return c2*xN*xN + c1*xN + c0;
}

// 新方法C：三分搜尋自動調參指數平滑
double cv_mse_alpha(const vector<double>& s, int n, double alpha, int cv = 3) {
    double err = 0;
    int lo = max(1, n - cv);
    int count = n - lo;
    for (int i = lo; i < n; ++i) {
        double sm = s[0];
        for (int j = 1; j < i; ++j) sm = alpha*s[j] + (1-alpha)*sm;
        double e = sm - s[i]; err += e*e;
    }
    return count > 0 ? err/count : 1e9;
}

double find_best_alpha(const vector<double>& s, int n) {
    if (n <= 2) return 0.3;
    double lo = 0.01, hi = 0.99;
    for (int i = 0; i < 60; ++i) {
        double m1 = lo + (hi-lo)/3, m2 = hi - (hi-lo)/3;
        if (cv_mse_alpha(s, n, m1) <= cv_mse_alpha(s, n, m2)) hi = m2;
        else lo = m1;
    }
    return (lo+hi)/2;
}

double pred_auto_exp(const vector<double>& s, int n) {
    double alpha = find_best_alpha(s, n);
    double sm = s[0];
    for (int i = 1; i < n; ++i) sm = alpha*s[i] + (1-alpha)*sm;
    return sm;
}

// ======================================================
// 主程式：逐屆回測
// ======================================================
int main() {
    const vector<string> METHOD_NAMES = {
        "線性回歸(w=10)",
        "MA(3)",
        "MA(4)",
        "Exp(α=0.3)",
        "Exp(α=0.5)",
        "加權線性回歸",
        "多項式回歸",
        "自動調參ES",
    };
    const int M = METHOD_NAMES.size();
    int T = NUM_WC - TEST_START; // 回測屆數 = 10

    cout << fixed << setprecision(2);
    cout << "================================================================\n";
    cout << " 近10屆回測報告（" << WC_YEARS[TEST_START]
         << "–" << WC_YEARS[NUM_WC-1] << "，共" << T << "屆）\n";
    cout << " 方法：用過去資料預測當屆，比對實際結果\n";
    cout << "================================================================\n\n";

    // 收集每方法、每隊的誤差
    // err[method][team] = 各屆誤差平方的總和
    vector<vector<double>> sum_sq(M, vector<double>(TEAMS.size(), 0));
    vector<vector<double>> sum_abs(M, vector<double>(TEAMS.size(), 0));
    int total_pred = 0;

    for (int ti = 0; ti < (int)TEAMS.size(); ++ti) {
        auto& t = TEAMS[ti];
        cout << "──────────────────────────────────────────────────────────────\n";
        cout << " " << t.name << "\n";
        cout << "──────────────────────────────────────────────────────────────\n";
        cout << left << setw(6) << "年份"
             << right << setw(7) << "實際";
        for (auto& nm : METHOD_NAMES)
            cout << setw(10) << nm.substr(0,9);
        cout << "\n" << string(6+7+M*10, '-') << "\n";

        for (int i = TEST_START; i < NUM_WC; ++i) {
            int n = i; // 用前 n 筆預測第 i 筆
            double actual = t.s[i];

            vector<double> preds = {
                pred_linreg (t.s, n, 10),
                pred_ma     (t.s, n, 3),
                pred_ma     (t.s, n, 4),
                pred_exp    (t.s, n, 0.3),
                pred_exp    (t.s, n, 0.5),
                pred_wlinreg(t.s, n, 10),
                pred_poly   (t.s, n, 10),
                pred_auto_exp(t.s, n),
            };

            cout << left << setw(6) << WC_YEARS[i]
                 << right << setw(7) << (int)actual;
            for (int m = 0; m < M; ++m) {
                double p = max(0.0, preds[m]);
                double e = p - actual;
                sum_sq [m][ti] += e*e;
                sum_abs[m][ti] += fabs(e);
                cout << setw(10) << p;
            }
            cout << "\n";
            if (i == TEST_START) total_pred++;
        }
        total_pred = T; // 10 屆

        // 各方法對此隊的 MSE / MAE
        cout << "\n";
        cout << right << setw(13) << "MSE";
        for (int m = 0; m < M; ++m)
            cout << setw(10) << sum_sq[m][ti]/T;
        cout << "\n";
        cout << right << setw(13) << "MAE";
        for (int m = 0; m < M; ++m)
            cout << setw(10) << sum_abs[m][ti]/T;
        cout << "\n\n";
    }

    // ── 全隊合計 MSE / MAE ──────────────────────────────
    int N_teams = TEAMS.size();
    cout << "================================================================\n";
    cout << " 全隊合計比較（6隊 × 10屆 = 60筆預測）\n";
    cout << "================================================================\n";

    vector<double> total_mse(M, 0), total_mae(M, 0);
    for (int m = 0; m < M; ++m)
        for (int ti = 0; ti < N_teams; ++ti) {
            total_mse[m] += sum_sq [m][ti];
            total_mae[m] += sum_abs[m][ti];
        }
    int N_total = N_teams * T;
    for (int m = 0; m < M; ++m) {
        total_mse[m] /= N_total;
        total_mae[m] /= N_total;
    }

    // 排序（MSE 由小到大）
    vector<int> rank_idx(M);
    iota(rank_idx.begin(), rank_idx.end(), 0);
    sort(rank_idx.begin(), rank_idx.end(),
         [&](int a, int b){ return total_mse[a] < total_mse[b]; });

    cout << "\n  排名  方法              MSE     MAE\n";
    cout << "  " << string(42, '-') << "\n";
    for (int r = 0; r < M; ++r) {
        int m = rank_idx[r];
        cout << "  " << setw(3) << r+1 << ".  "
             << left << setw(16) << METHOD_NAMES[m]
             << right << setw(7) << total_mse[m]
             << setw(8) << total_mae[m];
        if (r == 0) cout << "  ← 最準確";
        cout << "\n";
    }

    // ── 各隊最佳方法 ────────────────────────────────────
    cout << "\n================================================================\n";
    cout << " 各隊最佳方法（依 MSE）\n";
    cout << "================================================================\n";
    cout << "  " << left << setw(12) << "隊伍"
         << setw(18) << "最佳方法"
         << right << setw(7) << "MSE"
         << setw(8) << "MAE" << "\n";
    cout << "  " << string(44, '-') << "\n";
    for (int ti = 0; ti < N_teams; ++ti) {
        int best_m = 0;
        for (int m = 1; m < M; ++m)
            if (sum_sq[m][ti] < sum_sq[best_m][ti]) best_m = m;
        cout << "  " << left << setw(12) << TEAMS[ti].name
             << setw(18) << METHOD_NAMES[best_m]
             << right << setw(7) << sum_sq[best_m][ti]/T
             << setw(8) << sum_abs[best_m][ti]/T << "\n";
    }

    cout << "\n注意：MSE 越小代表預測越貼近實際結果。\n";
    return 0;
}
