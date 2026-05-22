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
const int NUM_WC = 22;

struct Team { string name; vector<double> s; };

const vector<Team> TEAMS = {
    {"Brazil",    {5,2,5,6,3,7,7,1,7,4,5,2,3,2,7,6,7,3,3,4,3,3}},
    {"Germany",   {0,5,2,0,7,4,3,6,5,7,2,6,6,7,3,3,6,5,5,7,1,1}},
    {"Argentina", {6,2,0,0,0,1,1,3,0,2,7,2,7,6,2,3,1,3,3,6,2,7}},
    {"France",    {1,2,3,0,1,5,0,1,0,1,1,4,5,0,0,7,1,6,1,3,7,6}},
    {"Italy",     {0,7,7,1,1,0,1,1,6,1,4,7,2,5,6,3,2,7,1,1,0,0}},
    {"Spain",     {3,3,0,4,1,0,1,1,0,1,1,2,3,2,3,1,3,2,7,1,2,3}},
};

// ======================================================
// 四項情境因素（每隊）
// ======================================================
struct Factors {
    string name;

    // 1. 球員狀態與陣容深度（0-10）
    //    依據 2025/26 球季主力陣容與板凳深度評估
    double squad;

    // 2. 賽程抽籤難度倍率（>1 = 較輕鬆）
    //    2026 擴充至48隊，依 FIFA 排名種子簽分組
    double draw;

    // 3. 教練戰術評分（0-10）
    //    考量現任主帥執教風格、大賽經驗與近期成績
    double coach;

    // 4. 主場優勢倍率（>1 = 有利）
    //    2026 由美國、加拿大、墨西哥聯合主辦
    //    南美洲球隊旅程較近、球迷支持較強；歐洲隊跨洲遠征較不利
    double home;
};

// ──────────────────────────────────────────────────────
// 因素說明依據：
//
// 球員陣容
//   Argentina 9.0 — Messi（可能最後一屆）、Lautaro、新生代接班人
//   France    9.2 — Mbappé（27歲巔峰）、Griezmann、Camavinga、Tchouaméni
//   Spain     8.7 — Lamine Yamal（18歲）、Pedri、Gavi、Morata
//   Brazil    8.2 — Vinícius Jr 巔峰期、Rodrygo、新星 Endrick
//   Germany   7.8 — Wirtz、Musiala 新黃金世代重建
//   Italy     7.2 — Barella、Tonali 為核心的重建世代
//
// 賽程抽籤
//   高排名球隊種子組優先抽籤，通常可避開其他強隊
//   Italy 若以附加賽資格出線，種子排名較低
//
// 教練戰術
//   Scaloni（阿根廷）  9.2 — 2022 世界盃冠軍教頭，戰術彈性高
//   De la Fuente（西班牙）8.5 — 2024 歐洲盃冠軍，傳控升級版
//   Deschamps 接班人（法國）8.3 — 延續冠軍文化
//   Nagelsmann（德國）  7.8 — 年輕主帥，2024 歐洲盃主辦準決賽
//   新教練（巴西）      7.6 — 正在建立新體系
//   Spalletti（義大利） 7.2 — 帶領義大利重返後的重建期
//
// 主場優勢
//   南美（巴西、阿根廷）：1.05 — 美洲大陸、時差小、球迷多
//   歐洲（法、德、西、意）：0.97 — 跨洲遠征、-7 至 -9 小時時差
// ──────────────────────────────────────────────────────

const vector<Factors> FACTORS = {
    //          squad  draw   coach  home
    {"Brazil",    8.2, 1.05,  7.6,  1.05},
    {"Germany",   7.8, 1.04,  7.8,  0.97},
    {"Argentina", 9.0, 1.08,  9.2,  1.05},
    {"France",    9.2, 1.07,  8.3,  0.97},
    {"Italy",     7.2, 0.97,  7.2,  0.97},
    {"Spain",     8.7, 1.05,  8.5,  0.97},
};

// ======================================================
// 歷史預測（三種回測最佳方法）
// ======================================================
double pred_exp03(const vector<double>& s) {
    double sm = s[0];
    for (int i = 1; i < (int)s.size(); ++i) sm = 0.3*s[i] + 0.7*sm;
    return sm;
}
double pred_ma4(const vector<double>& s) {
    int n = s.size(); double sum = 0;
    for (int i = n-4; i < n; ++i) sum += s[i];
    return sum / 4.0;
}
double pred_ma3(const vector<double>& s) {
    int n = s.size(); double sum = 0;
    for (int i = n-3; i < n; ++i) sum += s[i];
    return sum / 3.0;
}

// ======================================================
// 情境調整
// 各因素先正規化（除以全隊平均），再相乘得調整倍率
// factor_multiplier = (squad/μ_squad) × draw × (coach/μ_coach) × home
// final_score = hist_score × factor_multiplier
// ======================================================
int main() {
    int N = TEAMS.size();
    cout << fixed << setprecision(3);

    // ── 歷史基線分數（三方法平均）─────────────────────
    vector<double> hist(N);
    for (int i = 0; i < N; ++i) {
        double p1 = max(0.0, pred_exp03(TEAMS[i].s));
        double p2 = max(0.0, pred_ma4  (TEAMS[i].s));
        double p3 = max(0.0, pred_ma3  (TEAMS[i].s));
        hist[i] = (p1 + p2 + p3) / 3.0;
    }

    // ── 正規化因素 ────────────────────────────────────
    double mu_squad = 0, mu_coach = 0;
    for (auto& f : FACTORS) { mu_squad += f.squad; mu_coach += f.coach; }
    mu_squad /= N; mu_coach /= N;

    vector<double> adj_factor(N);
    for (int i = 0; i < N; ++i) {
        double squad_n = FACTORS[i].squad / mu_squad;
        double coach_n = FACTORS[i].coach / mu_coach;
        adj_factor[i]  = squad_n * FACTORS[i].draw * coach_n * FACTORS[i].home;
    }

    // ── 最終調整分數 ──────────────────────────────────
    vector<double> final_score(N);
    for (int i = 0; i < N; ++i)
        final_score[i] = max(0.0, hist[i] * adj_factor[i]);

    // ======================================================
    // 輸出
    // ======================================================
    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════╗\n";
    cout << "║   2026 FIFA World Cup — 完整因素預測                     ║\n";
    cout << "║   歷史統計 × 球員陣容 × 賽程抽籤 × 教練戰術 × 主場優勢  ║\n";
    cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ── 因素明細 ──────────────────────────────────────
    cout << "【四項情境因素明細】\n";
    cout << string(72, '-') << "\n";
    cout << left << setw(12) << "隊伍"
         << right << setw(8)  << "陣容(0-10)"
         << setw(10) << "抽籤×"
         << setw(10) << "教練(0-10)"
         << setw(10) << "主場×"
         << setw(10) << "綜合倍率" << "\n";
    cout << string(72, '-') << "\n";
    for (int i = 0; i < N; ++i) {
        auto& f = FACTORS[i];
        cout << left  << setw(12) << f.name
             << right << setw(8)  << f.squad
             << setw(10) << f.draw
             << setw(10) << f.coach
             << setw(10) << f.home
             << setw(10) << adj_factor[i] << "\n";
    }
    cout << string(72, '-') << "\n\n";

    // ── 調整前後對比 ──────────────────────────────────
    cout << "【調整前後對比（歷史分數 vs 完整因素調整後）】\n";
    cout << string(60, '-') << "\n";
    cout << left  << setw(12) << "隊伍"
         << right << setw(10) << "歷史分數"
         << setw(10) << "調整倍率"
         << setw(10) << "最終分數"
         << setw(10) << "變化" << "\n";
    cout << string(60, '-') << "\n";
    for (int i = 0; i < N; ++i) {
        double delta = final_score[i] - hist[i];
        string arrow = delta > 0.01 ? " ▲" : (delta < -0.01 ? " ▼" : " ─");
        cout << left  << setw(12) << TEAMS[i].name
             << right << setw(10) << hist[i]
             << setw(10) << adj_factor[i]
             << setw(10) << final_score[i]
             << "  " << (delta >= 0 ? "+" : "") << delta << arrow << "\n";
    }
    cout << "\n";

    // ── 最終排名（含機率）────────────────────────────
    auto print_ranking = [&](const string& title,
                              const vector<double>& scores) {
        double total = 0;
        for (double s : scores) total += s;
        if (total < 1e-9) total = 1;

        vector<pair<double,int>> rank;
        for (int i = 0; i < N; ++i) rank.push_back({scores[i], i});
        sort(rank.rbegin(), rank.rend());

        const string medals[] = {"🥇","🥈","🥉","  ","  ","  "};
        cout << "┌──────────────────────────────────────────────────────┐\n";
        cout << "│ " << left << setw(52) << title << "│\n";
        cout << "├──────┬────────────┬──────────┬───────────────────────┤\n";
        cout << "│ 排名 │ 隊伍       │ 最終分數 │      奪冠機率         │\n";
        cout << "├──────┼────────────┼──────────┼───────────────────────┤\n";
        for (int r = 0; r < (int)rank.size(); ++r) {
            auto [sc, idx] = rank[r];
            int bar_width = (int)(sc / total * 24);
            string bar(max(0, bar_width), '#');
            cout << "│ " << medals[r] << setw(2) << r+1 << " │ "
                 << left  << setw(10) << TEAMS[idx].name << " │ "
                 << right << setw(8)  << sc << " │ "
                 << setw(5) << sc/total*100 << "%  "
                 << left  << setw(16) << bar << "│\n";
        }
        cout << "└──────┴────────────┴──────────┴───────────────────────┘\n\n";
    };

    // 純歷史（對比用）
    print_ranking("歷史統計（Exp0.3+MA4+MA3 平均，無情境因素）", hist);

    // 完整因素調整
    print_ranking("完整因素調整（歷史 × 陣容 × 抽籤 × 教練 × 主場）", final_score);

    // ── 因素拆解影響分析 ─────────────────────────────
    cout << "【各因素對奪冠機率的影響（百分點差異）】\n";
    cout << string(70, '-') << "\n";

    auto get_probs = [&](const vector<double>& sc) {
        double tot = 0; for (double s : sc) tot += s;
        vector<double> p(N);
        for (int i = 0; i < N; ++i) p[i] = sc[i]/tot*100;
        return p;
    };

    // Base: pure history
    auto base_p = get_probs(hist);
    auto full_p = get_probs(final_score);

    // Impact of each factor separately (apply one at a time)
    auto apply_one = [&](int factor_idx) {
        vector<double> sc(N);
        for (int i = 0; i < N; ++i) {
            double f = 1.0;
            if (factor_idx == 0) f = FACTORS[i].squad / mu_squad;
            if (factor_idx == 1) f = FACTORS[i].draw;
            if (factor_idx == 2) f = FACTORS[i].coach / mu_coach;
            if (factor_idx == 3) f = FACTORS[i].home;
            sc[i] = max(0.0, hist[i] * f);
        }
        return get_probs(sc);
    };

    auto p_squad = apply_one(0);
    auto p_draw  = apply_one(1);
    auto p_coach = apply_one(2);
    auto p_home  = apply_one(3);

    cout << left << setw(12) << "隊伍"
         << right << setw(9) << "純歷史%"
         << setw(9) << "陣容Δ"
         << setw(9) << "抽籤Δ"
         << setw(9) << "教練Δ"
         << setw(9) << "主場Δ"
         << setw(9) << "最終%" << "\n";
    cout << string(70, '-') << "\n";
    for (int i = 0; i < N; ++i) {
        cout << left  << setw(12) << TEAMS[i].name
             << right << setw(9)  << base_p[i]
             << setw(9) << (p_squad[i]-base_p[i])
             << setw(9) << (p_draw [i]-base_p[i])
             << setw(9) << (p_coach[i]-base_p[i])
             << setw(9) << (p_home [i]-base_p[i])
             << setw(9) << full_p[i] << "\n";
    }
    cout << string(70, '-') << "\n";
    cout << "  Δ = 僅套用該單一因素後，與純歷史機率的差距（百分點）\n\n";

    cout << "注意：因素分數為主觀評估，反映 2025/26 球季資訊，\n";
    cout << "      非官方預測，請自行判斷參考價值。\n\n";

    return 0;
}
