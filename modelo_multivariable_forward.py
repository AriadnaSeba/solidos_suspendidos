import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from IPython.display import display, Markdown, Math

class ModeloForwardCV:
    def __init__(self, datos: pd.DataFrame, col_target="sol_sus"):
        self.datos = datos
        self.col_target = col_target
        self.bandas = [c for c in datos.columns if c.startswith("B")]
        self.datos_log = self.preprocesar()
        self.X_total = self.datos_log[[f"log_{b}" for b in self.bandas]]
        self.y_total = self.datos_log[f"log_{self.col_target}"]
        self.resultados = []
        self.mejor_comb = None
        self.modelo_final = None

    def preprocesar(self):
        mask = (self.datos[self.col_target] > 0) & (self.datos[self.bandas] > 0).all(axis=1)
        df = self.datos.loc[mask].copy()
        df[f"log_{self.col_target}"] = np.log(df[self.col_target])
        for b in self.bandas:
            df[f"log_{b}"] = np.log(df[b])
        return df

    def aic_gauss(self, y, y_pred, k):
        n = len(y)
        rss = np.sum((y - y_pred) ** 2)
        return n * np.log(rss / n) + 2 * k

    def forward_selection(self, max_features=3, n_splits=5, random_state=42):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        disponibles = list(self.X_total.columns)
        seleccionadas = []
        mejor_aic = np.inf

        while len(seleccionadas) < max_features:
            candidatos = [s + [c] for c in disponibles if c not in seleccionadas for s in [seleccionadas]]
            mejores = []

            for comb in candidatos:
                X = self.X_total[comb]
                y = self.y_total

                rmse_list, r2_list, r2_adj_list, aic_list = [], [], [], []

                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    model = LinearRegression().fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    p = X.shape[1]
                    r2_adj = 1 - ((1 - r2) * (len(y_test) - 1)) / (len(y_test) - p - 1)
                    aic = self.aic_gauss(y_test, y_pred, p + 1)

                    rmse_list.append(rmse)
                    r2_list.append(r2)
                    r2_adj_list.append(r2_adj)
                    aic_list.append(aic)

                promedio = {
                    "Variables": comb,
                    "R²": np.mean(r2_list),
                    "R²aj": np.mean(r2_adj_list),
                    "RMSE": np.mean(rmse_list),
                    "AIC": np.mean(aic_list)
                }
                mejores.append(promedio)

            mejores.sort(key=lambda d: d["AIC"])
            if mejores[0]["AIC"] < mejor_aic:
                mejor_aic = mejores[0]["AIC"]
                seleccionadas = mejores[0]["Variables"]
                self.resultados.append(mejores[0])
            else:
                break

        self.mejor_comb = seleccionadas

    def ajustar_modelo_final(self):
        X = self.X_total[self.mejor_comb]
        y = self.y_total
        self.modelo_final = LinearRegression().fit(X, y)
        return self.modelo_final

def mostrar_resultados_completos(modelo: ModeloForwardCV):
    display(style_metrics_table(pd.DataFrame(modelo.resultados), "Resumen métricas por forward selection"))
    display(Markdown("**Mejor combinación de bandas:** " + ", ".join(modelo.mejor_comb)))

    X = modelo.X_total[modelo.mejor_comb]
    y = modelo.y_total
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    pipe_final = Pipeline([
        ("s", StandardScaler()),
        ("m", LinearRegression())
    ]).fit(X_train, y_train)

    a = pipe_final.named_steps["m"].intercept_
    coefs = pipe_final.named_steps["m"].coef_
    bandas = modelo.mejor_comb
    bandas_limpias = [b.replace("log_", "") for b in bandas]

    eq_log = " + ".join([f"{c:.3f}\\cdot\\log({b})" for c, b in zip(coefs, bandas_limpias)])
    display(Markdown("**Ecuación en escala logarítmica:**"))
    display(Math(rf"$$ \log(\mathrm{{sol\_sus}}) = {a:.3f} + {eq_log} $$"))

    alpha = np.exp(a)
    eq_orig = " \\cdot ".join([f"{b}^{{{c:.3f}}}" for c, b in zip(coefs, bandas_limpias)])
    display(Markdown("**Ecuación en escala original:**"))
    display(Math(rf"$$ \mathrm{{sol\_sus}} = {alpha:.3f} \cdot {eq_orig} $$"))

    y_train_log_pred = pipe_final.predict(X_train)
    y_test_log_pred = pipe_final.predict(X_test)
    y_train_lin, y_train_pred = np.exp(y_train), np.exp(y_train_log_pred)
    y_test_lin , y_test_pred  = np.exp(y_test ), np.exp(y_test_log_pred)

    # --- Función de bootstrap ---
    def bootstrap_metricas(X, y, n_boot=1000, random_state=42):
        rng = np.random.default_rng(random_state)
        rmse_log, r2_log, r2_adj_log = [], [], []
        n = len(y)
        p = X.shape[1]

        for _ in range(n_boot):
            indices = rng.choice(n, size=n, replace=True)
            X_sample, y_sample = X.iloc[indices], y.iloc[indices]
            model_b = Pipeline([("s", StandardScaler()), ("m", LinearRegression())]).fit(X_sample, y_sample)
            y_pred = model_b.predict(X_sample)
            rmse = np.sqrt(mean_squared_error(y_sample, y_pred))
            r2 = r2_score(y_sample, y_pred)
            r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
            rmse_log.append(rmse)
            r2_log.append(r2)
            r2_adj_log.append(r2_adj)

        def ic_95(valores):
            return np.percentile(valores, [2.5, 97.5])

        return {
            "RMSE": (np.mean(rmse_log), *ic_95(rmse_log)),
            "R²":   (np.mean(r2_log), *ic_95(r2_log)),
            "R²aj": (np.mean(r2_adj_log), *ic_95(r2_adj_log)),
        }

    # --- Bootstrap sobre Train y Test (log) ---
    metricas_boot_train = bootstrap_metricas(X_train, y_train)
    metricas_boot_test = bootstrap_metricas(X_test, y_test)

    # --- Tabla con IC ---
    metricas_ic_df = pd.DataFrame({
        "Métrica": ["RMSE", "R²", "R² ajustado"],
        "Train (promedio)":   [f"{metricas_boot_train[m][0]:.3f}" for m in metricas_boot_train],
        "Train (IC 2.5%)":    [f"{metricas_boot_train[m][1]:.3f}" for m in metricas_boot_train],
        "Train (IC 97.5%)":   [f"{metricas_boot_train[m][2]:.3f}" for m in metricas_boot_train],
        "Test (promedio)":    [f"{metricas_boot_test[m][0]:.3f}" for m in metricas_boot_test],
        "Test (IC 2.5%)":     [f"{metricas_boot_test[m][1]:.3f}" for m in metricas_boot_test],
        "Test (IC 97.5%)":    [f"{metricas_boot_test[m][2]:.3f}" for m in metricas_boot_test],
    })

    display(style_metrics_table(metricas_ic_df, "Intervalos de Confianza 95 % (Bootstrap, escala log)"))

    def metricas_sin_ic(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        r2_adj = 1 - ((1 - r2) * (len(y_true) - 1)) / (len(y_true) - X.shape[1] - 1)
        return rmse, r2, r2_adj

    res = {
        "Train (log)": metricas_sin_ic(y_train, y_train_log_pred),
        "Test (log)": metricas_sin_ic(y_test, y_test_log_pred),
        "Train (mg/L)": metricas_sin_ic(y_train_lin, y_train_pred),
        "Test (mg/L)": metricas_sin_ic(y_test_lin, y_test_pred)
    }

    metrics_df = pd.DataFrame({
        "Métrica": ["RMSE", "R²", "R² ajustado"],
        "Train (log)": [f"{v:.3f}" for v in res["Train (log)"]],
        "Test (log)":  [f"{v:.3f}" for v in res["Test (log)"]],
        "Train (mg/L)": [f"{v:.3f}" for v in res["Train (mg/L)"]],
        "Test (mg/L)":  [f"{v:.3f}" for v in res["Test (mg/L)"]]
    })

    display(style_metrics_table(metrics_df, "Desempeño en Train y Test (log y mg/L)"))

    plt.rcParams.update({"axes.titlesize":10, "axes.labelsize":8,
                         "xtick.labelsize":7, "ytick.labelsize":7})
    TRAIN_CLR = "#9D50A6"
    TEST_CLR  = "#FFA24B"
    ID_CLR    = "#17A77E"
    TEST_REG  = "crimson"

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # Escala logarítmica
    ax[0].scatter(y_train, y_train_log_pred, alpha=0.6, color=TRAIN_CLR, edgecolor="k", label="Train (●)")
    ax[0].scatter(y_test , y_test_log_pred , alpha=0.6, color=TEST_CLR , marker="s", edgecolor="k", label="Test  (■)")
    lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
    ax[0].plot(lims, lims, "--", color=ID_CLR, lw=2.5, label="y = x")
    coef_test_log = np.polyfit(y_test, y_test_log_pred, 1)
    x_line_log = np.linspace(*lims, 100)
    y_line_log = np.polyval(coef_test_log, x_line_log)
    ax[0].plot(x_line_log, y_line_log, ":", lw=2, color=TEST_REG, label="Regresión Test")
    ax[0].set(xlabel="log(sol_sus) observado", ylabel="log(sol_sus) predicho", title="Escala log")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(ls=":")

    eq_log_str = f"log(sol_sus) = {a:.3f} + " + " + ".join([f"{c:.3f}·log({b})" for c, b in zip(coefs, bandas_limpias)])
    metrics_log_txt = (f"{eq_log_str}\n"
                       f"Train: RMSE={res['Train (log)'][0]:.3f}, R²={res['Train (log)'][1]:.3f}\n"
                       f"Test:  RMSE={res['Test (log)'][0]:.3f}, R²={res['Test (log)'][1]:.3f}")
    ax[0].text(0.04, 0.97, metrics_log_txt, transform=ax[0].transAxes,
               fontsize=7, va="top", linespacing=1.2,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

    # Escala original
    ax[1].scatter(y_train_lin, y_train_pred, alpha=0.6, color=TRAIN_CLR, edgecolor="k", label="Train (●)")
    ax[1].scatter(y_test_lin , y_test_pred , alpha=0.6, color=TEST_CLR , marker="s", edgecolor="k", label="Test  (■)")
    lims_o = [min(y_train_lin.min(), y_test_lin.min(), y_train_pred.min(), y_test_pred.min()),
              max(y_train_lin.max(), y_test_lin.max(), y_train_pred.max(), y_test_pred.max())]
    ax[1].plot(lims_o, lims_o, "--", color=ID_CLR, lw=2.5, label="y = x")
    coef_test_orig = np.polyfit(y_test_lin, y_test_pred, 1)
    x_line_orig = np.linspace(*lims_o, 100)
    y_line_orig = np.polyval(coef_test_orig, x_line_orig)
    ax[1].plot(x_line_orig, y_line_orig, ":", lw=2, color=TEST_REG, label="Regresión Test")
    ax[1].set(xlabel="sol_sus observado (mg/L)", ylabel="sol_sus predicho (mg/L)", title="Escala original")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].grid(ls=":")

    eq_orig_str = f"sol_sus = {alpha:.3f} · " + " · ".join([f"{b}^{{{c:.3f}}}" for c, b in zip(coefs, bandas_limpias)])
    metrics_orig_txt = (f"{eq_orig_str}\n"
                        f"Train: RMSE={res['Train (mg/L)'][0]:.3f}, R²={res['Train (mg/L)'][1]:.3f}\n"
                        f"Test:  RMSE={res['Test (mg/L)'][0]:.3f}, R²={res['Test (mg/L)'][1]:.3f}")
    ax[1].text(0.04, 0.97, metrics_orig_txt, transform=ax[1].transAxes,
               fontsize=7, va="top", linespacing=1.2,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

    fig.suptitle("Predicho vs Observado", fontsize=11, y=1.05)
    fig.tight_layout(pad=1.2)
    plt.show()

def style_metrics_table(df: pd.DataFrame, caption: str):
    formato = {col: "{:.3f}" for col in df.select_dtypes(include=[np.number]).columns}
    return (df.style
             .hide(axis="index")
             .format(formato)
             .set_caption(caption)
             .set_table_styles([{
                 "selector": "caption",
                 "props": [("caption-side", "top"),
                           ("font-weight", "bold"),
                           ("margin-bottom", "0.5em")]
             }], overwrite=False))
