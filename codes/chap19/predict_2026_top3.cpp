#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
using namespace std;

// ======================================================
// 歷史資料（1930-2022，共22屆）
// ======================================================
const int WC_YEARS[] = {
    1930,1934,1938,1950,1954,1958,1962,1966,
    1970,1974,1978,1982,1986,1990,1994,1998,
    2002,2006,2010,2014,2018,2022
};
const int NUM_WC = 22;

struct Team { string name; vector<double> s; };

vector<Team> TEAMS = {
    {"Brazil",    {5,2,5,6,3,7,7,1,7,4,5,2,3,2,7,6,7,3,3,4,3,3}},
    {"Germany",   {0,5,2,0,7,4,3,6,5,7,2,6,6,7,3,3,6,5,5,7,1,1}},
    {"Argentina", {6,2,0,0,0,1,1,3,0,2,7,2,7,6,2,3,1,3,3,6,2,7}},
    {"France",    {1,2,3,0,1,5,0,1,0,1,1,4,5,0,0,7,1,6,1,3,7,6}},
    {"Italy",     {0,7,7,1,1,0,1,1,6,1,4,7,2,5,6,3,2,7,1,1,0,0}},
    {"Spain",     {3,3,0,4,1,0,1,1,0,1,1,2,3,2,3,1,3,2,7,1,2,3}},
};

// ── 三種方法 ────────────────────────────────────────────

// 🥇 方法1：Exp(α=0.3)  ← 回測 MSE 最低
double pred_exp03(const vector<double>& s) {
    double sm = s[0];
    for (int i = 1; i < (int)s.size(); ++i)
        sm = 0.3*s[i] + 0.7*sm;
    return sm;
}

// 🥈 方法2：MA(4)  ← 回測 MSE 第二
double pred_ma4(const vector<double>& s) {
    int n = s.size();
    double sum = 0;
    for (int i = n-4; i < n; ++i) sum += s[i];
    return sum / 4.0;
}

// 🥉 方法3：MA(3)  ← 回測 MSE 第三
double pred_ma3(const vector<double>& s) {
    int n = s.size();
    double sum = 0;
    for (int i = n-3; i < n; ++i) sum += s[i];
    return sum / 3.0;
}

// ── 排名輸出 ────────────────────────────────────────────
void print_ranking(const string& title, const string& mse_info,
                   const vector<double>& preds) {
    double total = 0;
    for (double p : preds) total += p;
    if (total < 1e-9) total = 1;

    vector<pair<double,int>> rank;
    for (int i = 0; i < (int)preds.size(); ++i)
        rank.push_back({preds[i], i});
    sort(rank.rbegin(), rank.rend());

    cout << "┌─────────────────────────────────────────────┐\n";
    cout << "│ " << left << setw(43) << title << "│\n";
    cout << "│ " << left << setw(43) << mse_info << "│\n";
    cout << "├──────┬────────────┬──────────┬──────────────┤\n";
    cout << "│ 排名 │ 隊伍       │ 預測分數 │   奪冠機率   │\n";
    cout << "├──────┼────────────┼──────────┼──────────────┤\n";

    const string medals[] = {"🥇", "🥈", "🥉", "  ", "  ", "  "};
    for (int r = 0; r < (int)rank.size(); ++r) {
        auto [score, idx] = rank[r];
        double prob = score / total * 100.0;
        cout << "│ " << medals[r] << setw(2) << r+1
             << " │ " << left << setw(10) << TEAMS[idx].name
             << " │ " << right << setw(8) << fixed << setprecision(2) << score
             << " │ " << setw(10) << prob << "%   │\n";
    }
    cout << "└──────┴────────────┴──────────┴──────────────┘\n\n";
}

int main() {
    cout << fixed << setprecision(2);

    cout << "\n";
    cout << "╔══════════════════════════════════════════════╗\n";
    cout << "║   2026 FIFA 世界盃奪冠機率預測               ║\n";
    cout << "║   基於回測最佳前三方法（1986–2022，60筆）     ║\n";
    cout << "╚══════════════════════════════════════════════╝\n\n";

    // 近4屆/3屆資料預覽
    cout << "【近4屆實際成績（2010-2022）】\n";
    cout << left << setw(12) << "隊伍";
    for (int i = NUM_WC-4; i < NUM_WC; ++i)
        cout << right << setw(6) << WC_YEARS[i];
    cout << "\n" << string(36, '-') << "\n";
    for (auto& t : TEAMS) {
        cout << left << setw(12) << t.name;
        for (int i = NUM_WC-4; i < NUM_WC; ++i)
            cout << right << setw(6) << (int)t.s[i];
        cout << "\n";
    }
    cout << "\n";

    // 三方法各自預測
    vector<double> p1, p2, p3;
    for (auto& t : TEAMS) {
        p1.push_back(max(0.0, pred_exp03(t.s)));
        p2.push_back(max(0.0, pred_ma4(t.s)));
        p3.push_back(max(0.0, pred_ma3(t.s)));
    }

    print_ranking(
        "🥇 方法1：指數平滑化 Exp(α=0.3)",
        "   回測 MSE=5.82（60筆中最準）",
        p1);

    print_ranking(
        "🥈 方法2：移動平均 MA(4)  近4屆均值",
        "   回測 MSE=6.18",
        p2);

    print_ranking(
        "🥉 方法3：移動平均 MA(3)  近3屆均值",
        "   回測 MSE=6.19",
        p3);

    // 三方法平均
    vector<double> avg(TEAMS.size());
    for (int i = 0; i < (int)TEAMS.size(); ++i)
        avg[i] = (p1[i] + p2[i] + p3[i]) / 3.0;

    print_ranking(
        "⭐ 三方法綜合平均",
        "   Exp(0.3) + MA(4) + MA(3)",
        avg);

    // 詳細數字表
    cout << "【各隊預測分數明細】\n";
    cout << left << setw(12) << "隊伍"
         << right << setw(12) << "Exp(α=0.3)"
         << setw(8) << "MA(4)"
         << setw(8) << "MA(3)"
         << setw(10) << "平均" << "\n";
    cout << string(50, '-') << "\n";
    for (int i = 0; i < (int)TEAMS.size(); ++i) {
        cout << left << setw(12) << TEAMS[i].name
             << right << setw(12) << p1[i]
             << setw(8) << p2[i]
             << setw(8) << p3[i]
             << setw(10) << avg[i] << "\n";
    }

    cout << "\n注意：本預測僅基於歷史分數統計，未考慮球員、賽程與戰術等因素。\n";
    return 0;
}
