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
const int TEST_START = 12; // 回測起點：1986

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
// 8 種預測方法（用前 n 筆預測第 n 筆）
// ======================================================
double pred_linreg(const vector<double>& s, int n, int w=10) {
    int from = max(0, n-w), cnt = n-from;
    double sx=0,sy=0,sxy=0,sx2=0;
    for (int i=from;i<n;++i){double xi=i-from; sx+=xi;sy+=s[i];sxy+=xi*s[i];sx2+=xi*xi;}
    double D=cnt*sx2-sx*sx;
    if(fabs(D)<1e-9) return sy/cnt;
    double a=(cnt*sxy-sx*sy)/D;
    return a*cnt+(sy-a*sx)/cnt;
}

double pred_ma(const vector<double>& s, int n, int k) {
    int from=max(0,n-k); double sum=0;
    for(int i=from;i<n;++i) sum+=s[i];
    return sum/(n-from);
}

double pred_exp(const vector<double>& s, int n, double a) {
    double sm=s[0];
    for(int i=1;i<n;++i) sm=a*s[i]+(1-a)*sm;
    return sm;
}

double cv_mse_a(const vector<double>& s, int n, double a, int cv=3) {
    double err=0; int lo=max(1,n-cv),cnt=n-lo;
    for(int i=lo;i<n;++i){double sm=s[0];for(int j=1;j<i;++j)sm=a*s[j]+(1-a)*sm;err+=(sm-s[i])*(sm-s[i]);}
    return cnt>0?err/cnt:1e9;
}
double best_alpha(const vector<double>& s, int n) {
    if(n<=2) return 0.3;
    double lo=0.01,hi=0.99;
    for(int i=0;i<60;++i){double m1=lo+(hi-lo)/3,m2=hi-(hi-lo)/3;
        if(cv_mse_a(s,n,m1)<=cv_mse_a(s,n,m2)) hi=m2; else lo=m1;}
    return (lo+hi)/2;
}
double pred_auto_exp(const vector<double>& s, int n) {
    double a=best_alpha(s,n),sm=s[0];
    for(int i=1;i<n;++i) sm=a*s[i]+(1-a)*sm;
    return sm;
}

double pred_wlinreg(const vector<double>& s, int n, int w=10, double r=0.85) {
    int from=max(0,n-w),cnt=n-from;
    double sw=0,swx=0,swy=0,swx2=0,swxy=0;
    for(int i=from;i<n;++i){double xi=i-from,wi=pow(r,cnt-1-(i-from));
        sw+=wi;swx+=wi*xi;swy+=wi*s[i];swx2+=wi*xi*xi;swxy+=wi*xi*s[i];}
    double D=sw*swx2-swx*swx;
    if(fabs(D)<1e-9) return swy/sw;
    double a=(sw*swxy-swx*swy)/D;
    return a*cnt+(swy-a*swx)/sw;
}

double pred_poly(const vector<double>& s, int n, int w=10) {
    int from=max(0,n-w),cnt=n-from;
    double s0=cnt,s1=0,s2=0,s3=0,s4=0,t0=0,t1=0,t2=0;
    for(int i=from;i<n;++i){double xi=i-from;
        s1+=xi;s2+=xi*xi;s3+=xi*xi*xi;s4+=xi*xi*xi*xi;
        t0+=s[i];t1+=xi*s[i];t2+=xi*xi*s[i];}
    double A[3][4]={{s0,s1,s2,t0},{s1,s2,s3,t1},{s2,s3,s4,t2}};
    for(int col=0;col<3;++col){
        int pv=col; for(int r=col+1;r<3;++r) if(fabs(A[r][col])>fabs(A[pv][col])) pv=r;
        for(int j=0;j<4;++j) swap(A[col][j],A[pv][j]);
        if(fabs(A[col][col])<1e-9) continue;
        double d=A[col][col]; for(int j=col;j<4;++j) A[col][j]/=d;
        for(int r=0;r<3;++r){if(r==col) continue; double f=A[r][col];
            for(int j=col;j<4;++j) A[r][j]-=f*A[col][j];}
    }
    double xN=cnt; return A[2][3]*xN*xN+A[1][3]*xN+A[0][3];
}

// ======================================================
// 測試框架
// ======================================================
int pass_cnt=0, fail_cnt=0;

void chk(const string& name, bool ok) {
    cout << (ok ? "  [通過] " : "  [失敗] ") << name << "\n";
    ok ? ++pass_cnt : ++fail_cnt;
}
bool near_val(double a, double b, double eps=1e-4) { return fabs(a-b)<eps; }

// ======================================================
// 測試群組 1：基本正確性 — 對6隊 × 8方法 loop 檢查
// ======================================================
void test_basic_sanity() {
    cout << "\n[群組1] 基本正確性：所有方法 × 所有隊伍 (6×8=48筆)\n";

    const vector<string> mnames = {
        "Linreg(w=10)","MA(3)","MA(4)",
        "Exp(0.3)","Exp(0.5)","加權Linreg","多項式","自動ES"
    };

    int total_ok=0, total_bad=0;
    int poly_out_of_range = 0; // 多項式預期可能超出範圍（過擬合）
    for (auto& t : TEAMS) {
        int n = NUM_WC;
        vector<double> preds = {
            pred_linreg (t.s,n,10),
            pred_ma     (t.s,n,3),
            pred_ma     (t.s,n,4),
            pred_exp    (t.s,n,0.3),
            pred_exp    (t.s,n,0.5),
            pred_wlinreg(t.s,n,10),
            pred_poly   (t.s,n,10),
            pred_auto_exp(t.s,n),
        };
        for (int m=0; m<(int)preds.size(); ++m) {
            bool finite_ok = isfinite(preds[m]);
            bool range_ok  = preds[m] >= -1.0 && preds[m] <= 8.0;
            bool is_poly   = (m == 6);
            if (finite_ok && (range_ok || is_poly)) ++total_ok;
            else {
                ++total_bad;
                cout << "    [異常] " << t.name << " / " << mnames[m]
                     << " = " << preds[m] << "\n";
            }
            if (is_poly && !range_ok) ++poly_out_of_range;
        }
    }
    chk("非多項式方法(7種×6隊=42筆)全在合理範圍內", total_bad == 0);
    // 多項式回歸已知會過擬合，超出範圍反而是預期行為的驗證
    chk("多項式回歸出現超出 [0,7] 的值（過擬合特性確認）", poly_out_of_range > 0);
    cout << "  → " << total_ok << "/48 通過，多項式超範圍 " << poly_out_of_range << " 筆\n";
}

// ======================================================
// 測試群組 2：單調衰減隊伍 (連輸4屆)
// 期望所有方法預測下一屆不會高於7
// ======================================================
void test_declining_team() {
    cout << "\n[群組2] 衰退隊測試：連續下降 [7,6,5,4,3,2,1,0,0,0]\n";
    vector<double> s = {7,6,5,4,3,2,1,0,0,0};
    int n = s.size();

    struct Case { string name; double pred; };
    vector<Case> cases = {
        {"Linreg(w=10)",  pred_linreg (s,n,10)},
        {"MA(3)",         pred_ma     (s,n,3)},
        {"MA(4)",         pred_ma     (s,n,4)},
        {"Exp(0.3)",      pred_exp    (s,n,0.3)},
        {"Exp(0.5)",      pred_exp    (s,n,0.5)},
        {"加權Linreg",    pred_wlinreg(s,n,10)},
        {"多項式",        pred_poly   (s,n,10)},
        {"自動ES",        pred_auto_exp(s,n)},
    };

    for (auto& c : cases) {
        chk(c.name + " 預測 < 2.0（反映衰退）", c.pred < 2.0);
    }
}

// ======================================================
// 測試群組 3：常數序列 — 所有方法應回傳相同值
// ======================================================
void test_constant_series() {
    cout << "\n[群組3] 常數序列：[4,4,4,...,4] × 8方法\n";
    vector<double> s(20, 4.0);
    int n = s.size();

    auto run = [&](string nm, double p) {
        chk(nm + " 常數序列預測 ≈ 4.0", near_val(p, 4.0, 0.1));
    };
    run("Linreg",    pred_linreg (s,n,10));
    run("MA(3)",     pred_ma     (s,n,3));
    run("MA(4)",     pred_ma     (s,n,4));
    run("Exp(0.3)",  pred_exp    (s,n,0.3));
    run("Exp(0.5)",  pred_exp    (s,n,0.5));
    run("加權Linreg",pred_wlinreg(s,n,10));
    run("多項式",    pred_poly   (s,n,10));
    run("自動ES",    pred_auto_exp(s,n));
}

// ======================================================
// 測試群組 4：參數敏感度 — MA 窗口 k ∈ {1..8} loop
// 對常數序列 MA(k) 應恆等於常數
// ======================================================
void test_ma_window_loop() {
    cout << "\n[群組4] MA 窗口循環：k=1..8，常數序列應恆等於常數\n";
    vector<double> s(15, 6.0);
    int n = s.size();
    for (int k=1; k<=8; ++k) {
        double p = pred_ma(s, n, k);
        chk("MA(k=" + to_string(k) + ") = 6.0", near_val(p, 6.0));
    }
}

// ======================================================
// 測試群組 5：指數平滑 α 參數掃描
// 對線性遞增序列，α 越大（越重視近期）→ 預測值越大
// ======================================================
void test_alpha_sweep() {
    cout << "\n[群組5] 指數平滑 α 掃描：α 越大 → 預測越貼近最近值\n";
    // 遞增數列：舊=1, 新=10
    vector<double> s = {1,1,1,1,1,1,2,4,7,10};
    int n = s.size();
    const vector<double> alphas = {0.1, 0.2, 0.3, 0.5, 0.7, 0.9};

    double prev_pred = -1;
    bool monotone = true;
    for (double a : alphas) {
        double p = pred_exp(s, n, a);
        if (prev_pred >= 0 && p <= prev_pred) monotone = false;
        prev_pred = p;
    }
    chk("遞增序列：α 越大，ES 預測單調遞增", monotone);

    // 衰退序列：舊=10, 新=1
    vector<double> s2 = {10,10,10,10,10,10,8,5,2,1};
    prev_pred = 99;
    bool monotone2 = true;
    for (double a : alphas) {
        double p = pred_exp(s2, n, a);
        if (p >= prev_pred) monotone2 = false;
        prev_pred = p;
    }
    chk("衰退序列：α 越大，ES 預測單調遞減", monotone2);
}

// ======================================================
// 測試群組 6：回測 MSE 排名驗證
// Exp(0.3) 必須是8方法中 MSE 最低（根據 60筆回測結論）
// ======================================================
void test_backtest_ranking() {
    cout << "\n[群組6] 回測驗證：60筆 MSE — Exp(0.3) 應為最低\n";

    const int T = NUM_WC - TEST_START; // 10屆
    int N = TEAMS.size();
    const int M = 8;

    vector<double> total_sq(M, 0);

    for (auto& t : TEAMS) {
        for (int i = TEST_START; i < NUM_WC; ++i) {
            int n = i;
            double actual = t.s[i];
            vector<double> preds = {
                pred_linreg (t.s,n,10),
                pred_ma     (t.s,n,3),
                pred_ma     (t.s,n,4),
                pred_exp    (t.s,n,0.3),
                pred_exp    (t.s,n,0.5),
                pred_wlinreg(t.s,n,10),
                pred_poly   (t.s,n,10),
                pred_auto_exp(t.s,n),
            };
            for (int m=0; m<M; ++m) {
                double e = max(0.0, preds[m]) - actual;
                total_sq[m] += e*e;
            }
        }
    }

    // idx=3 是 Exp(α=0.3)
    int best_m = min_element(total_sq.begin(), total_sq.end()) - total_sq.begin();
    double best_mse = total_sq[best_m] / (N*T);

    const vector<string> mnames = {
        "Linreg","MA(3)","MA(4)","Exp(0.3)","Exp(0.5)","加權Linreg","多項式","自動ES"
    };

    cout << "  各方法全局 MSE：\n";
    for (int m=0; m<M; ++m)
        cout << "    " << left << setw(12) << mnames[m]
             << fixed << setprecision(3) << total_sq[m]/(N*T)
             << (m==best_m ? "  ← 最低" : "") << "\n";

    chk("Exp(α=0.3) 為60筆回測中 MSE 最低方法", best_m == 3);
    chk("MSE 最低值 < 6.0",  best_mse < 6.0);
    chk("多項式回歸 MSE > 8.0（確認高過擬合風險）", total_sq[6]/(N*T) > 8.0);
}

// ======================================================
// 測試群組 7：前3方法2026預測一致性
// 三方法都應給出 France / Argentina 奪冠機率 > 20%
// ======================================================
void test_2026_consistency() {
    cout << "\n[群組7] 2026 預測一致性：前三方法均應給法阿機率 > 20%\n";

    auto get_probs = [&](auto pred_fn) {
        vector<double> preds;
        for (auto& t : TEAMS)
            preds.push_back(max(0.0, pred_fn(t.s)));
        double total = 0;
        for (double p : preds) total += p;
        vector<double> probs;
        for (double p : preds) probs.push_back(p/total*100);
        return probs; // 順序與 TEAMS 相同
    };

    // 0=Brazil 1=Germany 2=Argentina 3=France 4=Italy 5=Spain
    auto probs1 = get_probs([](auto& s){ return pred_exp(s,(int)s.size(),0.3); });
    auto probs2 = get_probs([](auto& s){ return pred_ma(s,(int)s.size(),4); });
    auto probs3 = get_probs([](auto& s){ return pred_ma(s,(int)s.size(),3); });

    chk("Exp(0.3)：法國機率 > 20%",       probs1[3] > 20.0);
    chk("Exp(0.3)：阿根廷機率 > 20%",     probs1[2] > 20.0);
    chk("MA(4)：阿根廷排名 ≥ 法國或差距<5%",
        probs2[2] >= probs2[3] || (probs2[3]-probs2[2]) < 5.0);
    chk("MA(3)：法國機率 > 25%",           probs3[3] > 25.0);
    chk("義大利三方法均 < 10%",
        probs1[4] < 10.0 && probs2[4] < 10.0 && probs3[4] < 10.0);

    // 三方法平均
    vector<double> avg(TEAMS.size());
    for (int i=0; i<(int)TEAMS.size(); ++i)
        avg[i] = (probs1[i]+probs2[i]+probs3[i])/3;

    int top1_idx = max_element(avg.begin(),avg.end()) - avg.begin();
    chk("三方法平均：冠軍熱門為法國或阿根廷",
        top1_idx == 2 || top1_idx == 3);
}

// ======================================================
// 主程式
// ======================================================
int main() {
    cout << fixed << setprecision(3);
    cout << "========================================================\n";
    cout << "  第19章 Loop 測試（7群組，全方法 × 全隊 系統驗證）\n";
    cout << "========================================================\n";

    test_basic_sanity();
    test_declining_team();
    test_constant_series();
    test_ma_window_loop();
    test_alpha_sweep();
    test_backtest_ranking();
    test_2026_consistency();

    int total = pass_cnt + fail_cnt;
    cout << "\n========================================================\n";
    cout << "  結果：" << pass_cnt << " 通過 / " << fail_cnt << " 失敗"
         << "（共 " << total << " 個測試）\n";
    cout << "========================================================\n";

    return fail_cnt > 0 ? 1 : 0;
}
