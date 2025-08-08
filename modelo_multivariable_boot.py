import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Math

class ModeloMultivariableCompleto:
    def __init__(self, datos: pd.DataFrame, col_target="sol_sus", max_features=3, n_boot=300):
        self.datos = datos.copy()
        self.col_target = col_target
        self.bandas = [c for c in datos.columns if c.startswith("B")]
        self.max_features = max_features
        self.n_boot = n_boot
        self._prepare_data()
        self.resultados = []
        self.mejor_comb = None

    def _prepare_data(self):
        df = self.datos.copy()
        mask = (df[self.col_target] > 0) & (df[self.bandas] > 0).all(axis=1)
        df = df.loc[mask].reset_index(drop=True)
        logs = pd.DataFrame({f"log_{b}": np.log(df[b]) for b in self.bandas})
        ratios = {
            f"ratio_{b1}_{b2}": df[b1] / df[b2]
            for b1, b2 in combinations(self.bandas, 2)
        }
        log_ratios = {
            key.replace("ratio_", "log_ratio_"): np.log(val)
            for key, val in ratios.items()
        }
        log_target = pd.Series(np.log(df[self.col_target]), name=f"log_{self.col_target}")
        self.df_pre = pd.concat([logs, pd.DataFrame(ratios),
                                 pd.DataFrame(log_ratios), log_target], axis=1)
        self.X_total = self.df_pre[
            [c for c in self.df_pre.columns
             if c.startswith("log_") and c != f"log_{self.col_target}"]
        ]
        self.y_total = self.df_pre[f"log_{self.col_target}"]

    def aic_gauss(self, y, y_pred, k):
        n = len(y)
        rss = ((y - y_pred) ** 2).sum()
        return n * np.log(rss / n) + 2 * k

    def _bootstrap(self, X, y):
        rng = np.random.default_rng(0)
        stats = {"RMSE": [], "R²": [], "R²aj": []}
        n, p = len(y), X.shape[1]
        for _ in range(self.n_boot):
            idx = rng.choice(n, n, True)
            Xs, ys = X.iloc[idx], y.iloc[idx]
            pipe = Pipeline([
                ("s", StandardScaler()),
                ("m", LinearRegression())
            ]).fit(Xs, ys)
            pred = pipe.predict(Xs)
            rm = np.sqrt(mean_squared_error(ys, pred))
            r2 = r2_score(ys, pred)
            r2aj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
            stats["RMSE"].append(rm)
            stats["R²"].append(r2)
            stats["R²aj"].append(r2aj)
        return {
            k: (np.mean(v), *np.percentile(v, [2.5, 97.5]))
            for k, v in stats.items()
        }

    def forward_selection(self):
        feats = list(self.X_total.columns)
        selected, best_aic = [], np.inf
        while len(selected) < self.max_features:
            candidates = [selected + [f] for f in feats if f not in selected]
            metrics_list = []
            for comb in candidates:
                mdl_full = LinearRegression().fit(self.X_total[comb], self.y_total)
                pred_full = mdl_full.predict(self.X_total[comb])
                aic_value = self.aic_gauss(self.y_total, pred_full, len(comb) + 1)
                stats = self._bootstrap(self.X_total[comb], self.y_total)
                rm, r2, r2aj = stats["RMSE"][0], stats["R²"][0], stats["R²aj"][0]
                metrics_list.append((comb, rm, r2, r2aj, aic_value))
            comb, rm, r2m, r2ajm, aicm = min(metrics_list, key=lambda x: x[4])
            if aicm < best_aic:
                best_aic = aicm
                selected = comb
                self.resultados.append({
                    "Variables": comb,
                    "RMSE": rm,
                    "R²": r2m,
                    "R²aj": r2ajm,
                    "AIC": aicm
                })
            else:
                break
        self.mejor_comb = selected

    def ajustar_modelo_final(self):
        return LinearRegression().fit(self.X_total[self.mejor_comb], self.y_total)

    def mostrar_resultados(self):
        def label(f):
            if f.startswith("log_ratio_"):
                p1, p2 = f.replace("log_ratio_", "").split("_")
                return f"{p1}/{p2}"
            return f.replace("log_", "")

        # 1) Selección
        df_sel = pd.DataFrame(self.resultados)
        display(self._style_table(df_sel, "Selección variables (Bootstrap completo)"))
        display(Markdown("**Variables finales:** " +
                         ", ".join(label(f) for f in self.mejor_comb)))

        # 2) Split y predicciones
        Xtr, Xte, ytr, yte = train_test_split(
            self.X_total[self.mejor_comb], self.y_total, random_state=42)
        pipe = Pipeline([
            ("s", StandardScaler()),
            ("m", LinearRegression())
        ]).fit(Xtr, ytr)
        ytr_p, yte_p = pipe.predict(Xtr), pipe.predict(Xte)

        # 3) Ecuación en escala logarítmica
        a = pipe.named_steps["m"].intercept_
        coefs = pipe.named_steps["m"].coef_
        bandas = self.mejor_comb
        parts_log = [
            rf"{('+' if c >= 0 else '−')}{abs(c):.3f}\,\log({label(f)})"
            for c, f in zip(coefs, bandas)
        ]
        eq_log = "".join(parts_log)
        display(Markdown("**Ecuación en escala logarítmica:**"))
        display(Math(fr"\log(sol\_sus) = {a:.3f}{eq_log}"))

        # 4) Ecuación en escala original
        alpha = np.exp(a)
        parts_orig = []
        for c, f in zip(coefs, bandas):
            var = label(f)
            if "/" in var:
                num, den = var.split("/")
                parts_orig.append(
                    rf"\left(\frac{{{num}}}{{{den}}}\right)^{{{c:.3f}}}"
                )
            else:
                parts_orig.append(rf"{var}^{{{c:.3f}}}")
        eq_orig = r" \times ".join(parts_orig)
        display(Markdown("**Ecuación en escala original:**"))
        display(Math(fr"sol\_sus = {alpha:.3f}{eq_orig}"))

        # 5) Métricas Train/Test
        def calc(y_t, y_p):
            rm = np.sqrt(mean_squared_error(y_t, y_p))
            r2 = r2_score(y_t, y_p)
            r2aj = 1 - ((1 - r2) * (len(y_t) - 1)) / (len(y_t) - len(coefs) - 1)
            return rm, r2, r2aj

        res = {
            "Train (log)": calc(ytr, ytr_p),
            "Test (log)": calc(yte, yte_p),
            "Train (mg/L)": calc(np.exp(ytr), np.exp(ytr_p)),
            "Test (mg/L)": calc(np.exp(yte), np.exp(yte_p))
        }
        df_perf = pd.DataFrame({
            "Métrica": ["RMSE", "R²", "R² adj"],
            **{k: [f"{v:.3f}" for v in vals] for k, vals in res.items()}
        })
        display(self._style_table(df_perf, "Desempeño Train y Test"))

        # 6) Intervalos de confianza (bootstrap)
        mb_tr, mb_te = self._bootstrap(Xtr, ytr), self._bootstrap(Xte, yte)
        df_ic = pd.DataFrame({
            "Métrica": ["RMSE", "R²", "R² adj"],
            "Train prom": [f"{mb_tr[k][0]:.3f}" for k in mb_tr],
            "Train 2.5%": [f"{mb_tr[k][1]:.3f}" for k in mb_tr],
            "Train 97.5%": [f"{mb_tr[k][2]:.3f}" for k in mb_tr],
            "Test prom": [f"{mb_te[k][0]:.3f}" for k in mb_te],
            "Test 2.5%": [f"{mb_te[k][1]:.3f}" for k in mb_te],
            "Test 97.5%": [f"{mb_te[k][2]:.3f}" for k in mb_te],
        })
        display(self._style_table(df_ic, "Intervalos de Confianza 95 % (Bootstrap)"))

        # 7) Gráficos Predicho vs Observado
        plt.rcParams.update({
            "axes.titlesize": 10, "axes.labelsize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7
        })
        TRAIN_CLR, TEST_CLR, ID_CLR, TEST_REG = "#9D50A6", "#FFA24B", "#17A77E", "crimson"
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))

        # Escala log
        ax[0].scatter(ytr, ytr_p, c=TRAIN_CLR, label="Train", alpha=0.6, edgecolor="k")
        ax[0].scatter(yte, yte_p, c=TEST_CLR, marker="s", label="Test", alpha=0.6, edgecolor="k")
        lims = [min(ytr.min(), yte.min()), max(ytr.max(), yte.max())]
        ax[0].plot(lims, lims, "--", c=ID_CLR, lw=2.5, label="Identidad")
        lr = np.polyfit(yte, yte_p, 1)
        xln = np.linspace(*lims, 100)
        ax[0].plot(xln, np.polyval(lr, xln), ":", c=TEST_REG, lw=2, label="Regresión")
        ax[0].set(xlabel="log(sol_sus) observado", ylabel="log(sol_sus) predicho",
                  title="Escala log")
        ax[0].legend(fontsize=7, frameon=False)
        txt0 = (
            f"Train: RMSE={res['Train (log)'][0]:.3f}, R²={res['Train (log)'][1]:.3f}\n"
            f"Test:  RMSE={res['Test (log)'][0]:.3f}, R²={res['Test (log)'][1]:.3f}"
        )
        ax[0].text(0.04, 0.97, txt0, transform=ax[0].transAxes,
                   fontsize=7, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white"))
        ax[0].grid(ls=":")

        # Escala original
        ytr_o, yte_o = np.exp(ytr), np.exp(yte)
        ax[1].scatter(ytr_o, np.exp(ytr_p), c=TRAIN_CLR, label="Train", alpha=0.6, edgecolor="k")
        ax[1].scatter(yte_o, np.exp(yte_p), c=TEST_CLR, marker="s", label="Test", alpha=0.6, edgecolor="k")
        limso = [min(ytr_o.min(), yte_o.min()), max(ytr_o.max(), yte_o.max())]
        ax[1].plot(limso, limso, "--", c=ID_CLR, lw=2.5, label="Identidad")
        lr2 = np.polyfit(yte_o, np.exp(yte_p), 1)
        xlo = np.linspace(*limso, 100)
        ax[1].plot(xlo, np.polyval(lr2, xlo), ":", c=TEST_REG, lw=2, label="Regresión")
        ax[1].set(xlabel="sol_sus observado (mg/L)", ylabel="sol_sus predicho (mg/L)",
                  title="Escala original")
        ax[1].legend(fontsize=7, frameon=False)
        txt1 = (
            f"Train: RMSE={res['Train (mg/L)'][0]:.3f}, R²={res['Train (mg/L)'][1]:.3f}\n"
            f"Test:  RMSE={res['Test (mg/L)'][0]:.3f}, R²={res['Test (mg/L)'][1]:.3f}"
        )
        ax[1].text(0.04, 0.97, txt1, transform=ax[1].transAxes,
                   fontsize=7, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white"))
        ax[1].grid(ls=":")

        fig.suptitle("Predicho vs Observado", fontsize=11, y=1.05)
        fig.tight_layout(pad=1.2)
        plt.show()

    @staticmethod
    def _style_table(df, caption):
        return df.style.hide(axis="index").format(precision=3).set_caption(caption)
