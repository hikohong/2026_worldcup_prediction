#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace std;

// 表現分數定義
// 7=冠軍, 6=亞軍, 5=第三名, 4=第四名, 3=八強, 2=十六強, 1=小組出局, 0=未參賽/預選賽落敗

const int NUM_WC = 22;

const int WC_YEARS[NUM_WC] = {
    1930,1934,1938,1950,1954,1958,1962,1966,
    1970,1974,1978,1982,1986,1990,1994,1998,
    2002,2006,2010,2014,2018,2022
};

struct Team {
    string name;
    vector<double> s; // 分數序列
};

// ======== 三種預測模型 ========
// 線性回歸：用 s[from..n-1] 的 window 個點求係數，預測下一點
// window=0 時使用全部資料點

double pred_linear(const vector<double>& s, int n, int window = 0) {
    int from = (window > 0) ? max(0, n - window) : 0;
    int cnt = n - from;
    double sx=0, sy=0, sxy=0, sx2=0;
    for (int i = from; i < n; ++i) {
        double xi = i - from; // 區域索引 0,1,...,cnt-1
        sx += xi; sy += s[i]; sxy += xi*s[i]; sx2 += xi*xi;
    }
    double D = cnt*sx2 - sx*sx;
    if (fabs(D) < 1e-9) return sy / cnt;
    double a = (cnt*sxy - sx*sy) / D;
    double b = (sy - a*sx) / cnt;
    return a*cnt + b; // 預測下一點（區域索引 = cnt）
}

// 移動平均：最近 window 個點的平均

double pred_ma(const vector<double>& s, int n, int window) {
    int from = max(0, n - window);
    double sum = 0;
    for (int i = from; i < n; ++i) sum += s[i];
    return sum / (n - from);
}

// 指數平滑化：S_t = alpha*y_t + (1-alpha)*S_{t-1}，以最後的平滑值作為預測值

double pred_exp(const vector<double>& s, int n, double alpha) {
    double sm = s[0];
    for (int i = 1; i < n; ++i) sm = alpha*s[i] + (1.0-alpha)*sm;
    return sm;
}

// ======== MSE 計算（以最後 k 屆進行交叉驗證）========

struct MSEResult {
    double lin, ma3, ma4, exp03, exp05;
    string best_name() const {
        vector<pair<double,string>> v = {
            {lin,"線性回歸"},{ma3,"MA(3)"},{ma4,"MA(4)"},
            {exp03,"Exp(α=0.3)"},{exp05,"Exp(α=0.5)"}
        };
        return min_element(v.begin(), v.end())->second;
    }
    double best_val() const {
        return min({lin, ma3, ma4, exp03, exp05});
    }
};

// window=10：線性回歸僅使用最近10屆資料

const int LIN_WINDOW = 10;

MSEResult compute_mse(const vector<double>& s, int k = 5) {
    int n = s.size();
    MSEResult m = {0,0,0,0,0};
    for (int i = n-k; i < n; ++i) {
        double actual = s[i];
        auto sq = [](double x){ return x*x; };
        m.lin   += sq(actual - pred_linear(s, i, LIN_WINDOW));
        m.ma3   += sq(actual - pred_ma(s, i, 3));
        m.ma4   += sq(actual - pred_ma(s, i, 4));
        m.exp03 += sq(actual - pred_exp(s, i, 0.3));
        m.exp05 += sq(actual - pred_exp(s, i, 0.5));
    }
    m.lin /= k; m.ma3 /= k; m.ma4 /= k; m.exp03 /= k; m.exp05 /= k;
    return m;
}

// 以最佳模型預測 2026 年結果

double predict_2026(const vector<double>& s, const MSEResult& m) {
    int n = s.size();
    string best = m.best_name();
    double p;
    if (best == "線性回歸")   p = pred_linear(s, n, LIN_WINDOW);
    else if (best == "MA(3)") p = pred_ma(s, n, 3);
    else if (best == "MA(4)") p = pred_ma(s, n, 4);
    else if (best == "Exp(α=0.3)") p = pred_exp(s, n, 0.3);
    else p = pred_exp(s, n, 0.5);
    return max(0.0, p);
}

int main() {
    // ======== 歷史資料（1930-2022，共22屆）========
    vector<Team> teams = {
        // Brazil：參加全部屆次（5次奪冠）
        {"Brazil", {5,2,5,6,3,7,7,1,7,4,5,2,3,2,7,6,7,3,3,4,3,3}},
        // Germany/W.Germany：1930未參加，1950禁賽（4次奪冠）
        {"Germany", {0,5,2,0,7,4,3,6,5,7,2,6,6,7,3,3,6,5,5,7,1,1}},
        // Argentina（3次奪冠）
        {"Argentina", {6,2,0,0,0,1,1,3,0,2,7,2,7,6,2,3,1,3,3,6,2,7}},
        // France（2次奪冠）
        {"France", {1,2,3,0,1,5,0,1,0,1,1,4,5,0,0,7,1,6,1,3,7,6}},
        // Italy（4次奪冠，2018/2022預選賽落敗）
        {"Italy", {0,7,7,1,1,0,1,1,6,1,4,7,2,5,6,3,2,7,1,1,0,0}},
        // Spain（1次奪冠）
        {"Spain", {3,3,0,4,1,0,1,1,0,1,1,2,3,2,3,1,3,2,7,1,2,3}},
    };

    cout << fixed << setprecision(2);
    cout << "========================================\n";
    cout << " FIFA 世界盃歷史資料分析\n";
    cout << " 分數: 7=冠軍 6=亞軍 5=第三名 4=第四名\n";
    cout << " 3=八強 2=十六強 1=小組出局 0=未參賽\n";
    cout << "========================================\n\n";

    // 歷史分數一覽
    cout << setw(6) << "年份";
    for (auto& t : teams) cout << setw(11) << t.name;
    cout << "\n" << string(72, '-') << "\n";
    for (int i = 0; i < NUM_WC; ++i) {
        cout << setw(6) << WC_YEARS[i];
        for (auto& t : teams) cout << setw(11) << (int)t.s[i];
        cout << "\n";
    }

    // 平均與最高分
    cout << string(72, '-') << "\n";
    cout << setw(6) << "平均";
    for (auto& t : teams) {
        double avg = accumulate(t.s.begin(), t.s.end(), 0.0) / t.s.size();
        cout << setw(11) << avg;
    }
    cout << "\n";
    cout << setw(6) << "最高";
    for (auto& t : teams) cout << setw(11) << (int)*max_element(t.s.begin(), t.s.end());
    cout << "\n\n";

    // 近5屆表現趨勢（2002-2022）
    cout << "========================================\n";
    cout << " 近5屆表現趨勢 (2002-2022)\n";
    cout << "========================================\n";
    cout << setw(6) << "年份";
    for (auto& t : teams) cout << setw(11) << t.name;
    cout << "\n";
    for (int i = NUM_WC-5; i < NUM_WC; ++i) {
        cout << setw(6) << WC_YEARS[i];
        for (auto& t : teams) cout << setw(11) << (int)t.s[i];
        cout << "\n";
    }

    // MSE 交叉驗證
    cout << "\n========================================\n";
    cout << " 各模型 MSE（近5屆交叉驗證）\n";
    cout << " ※ 數值越小表示對歷史資料的擬合度越高\n";
    cout << "========================================\n";
    cout << setw(11) << "隊伍"
        << setw(13) << "線性回歸(10屆)"
        << setw(7) << "MA(3)"
        << setw(7) << "MA(4)"
        << setw(10) << "Exp(0.3)"
        << setw(10) << "Exp(0.5)"
        << setw(12) << "最佳模型" << "\n";
    cout << string(66, '-') << "\n";

    vector<MSEResult> mse_results;
    for (auto& t : teams) {
        MSEResult m = compute_mse(t.s, 5);
        mse_results.push_back(m);
        cout << setw(11) << t.name
            << setw(13) << m.lin
            << setw(7) << m.ma3
            << setw(7) << m.ma4
            << setw(10) << m.exp03
            << setw(10) << m.exp05
            << setw(12) << m.best_name() << "\n";
    }

    // 2026 預測
    cout << "\n========================================\n";
    cout << " 2026 世界盃奪冠機率預測\n";
    cout << "========================================\n";
    vector<pair<double,int>> ranked;
    vector<double> raw_pred;
    for (int i = 0; i < (int)teams.size(); ++i) {
        double p = predict_2026(teams[i].s, mse_results[i]);
        raw_pred.push_back(p);
        ranked.push_back({p, i});
    }
    sort(ranked.rbegin(), ranked.rend());

    double total = accumulate(raw_pred.begin(), raw_pred.end(), 0.0);
    if (total < 1e-9) total = 1.0;

    cout << setw(4) << "排名"
        << setw(11) << "隊伍"
        << setw(12) << "預測分數"
        << setw(10) << "奪冠機率"
        << setw(14) << "使用模型" << "\n";
    cout << string(51, '-') << "\n";
    for (int rank = 0; rank < (int)ranked.size(); ++rank) {
        auto [score, idx] = ranked[rank];
        double prob = score / total * 100.0;
        cout << setw(4) << rank+1
            << setw(11) << teams[idx].name
            << setw(12) << score
            << setw(9) << prob << "%"
            << setw(14) << mse_results[idx].best_name() << "\n";
    }

    cout << "\n注意：本預測僅基於過去表現分數的統計模型。\n";
    cout << " 未考慮球員狀態、賽程抽籤及戰術等因素。\n";

    return 0;
}
