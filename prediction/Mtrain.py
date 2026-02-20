# --------------------------------------------------
#Mtrain creates random forest for a response
#varialbe and exports metrics and figures to a PDF
# --------------------------------------------------

# --------------------------------------------------
#Import Requirements
# --------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

from skopt import BayesSearchCV
from skopt.space import Integer, Real

from matplotlib.backends.backend_pdf import PdfPages

import os
import pickle

# --------------------------------------------------
#Prepare Data (Note this is modifiable for each run)
# --------------------------------------------------
df = pd.read_csv("AllDat5_2023.csv") #Add path to your data
predictor_cols = (
    ["hour_decimal", "WDIRs", "WDIRc", "WSPD", "GST", "WVHT", "DPD", #Include the predictor variables in use
    "APD", "MWDs", "MWDc", "PRES", "ATMP", "WTMP", "DEWP"]
)
response = ["H"] #Include the response variable
# response = ["T"]
# response = ["WS"]
# response = ["WL"]

test_size = 0.2 #Split for testing set
random_state = 42
n_iter_bayes = 50 #Number of iterations
output_pdf = "report_WaveHeight.pdf" #Specify output pdf for review
# output_pdf = "report_WavePeriod.pdf"
# output_pdf = "report_WindSpeed.pdf"
# output_pdf = "report_WaterLevel.pdf"

PICKLE_DIR = "pickle"
os.makedirs(PICKLE_DIR, exist_ok=True)
# --------------------------------------------------
#Clean Criteria (Note this removes bad data)
# --------------------------------------------------
outDat = (
    [99, 99.0, 999, 999.0, 9999, 9999.0, -99, -99.0,
    -999, -999.0, -9999, -9999.0, -7999]
)

# --------------------------------------------------
#Define Bayesian Search Space
# --------------------------------------------------
search_space = {
    "n_estimators": Integer(200, 800),
    "max_depth": Integer(3, 30),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Real(0.3, 1.0)
}

# --------------------------------------------------
#Define Inputs and Check Data Quality
# --------------------------------------------------
X = df[predictor_cols].to_numpy(dtype=float, copy=True)
X[(X<-1) | np.isin(X, outDat)] = np.nan #Check to clean data
y = df[response].to_numpy(dtype=float, copy=True)
y[(y<-1) | np.isin(y, outDat)] = np.nan

mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y).any(axis=1)) #Check to see if data is consistent
X = X[mask]
y = y[mask]

# --------------------------------------------------
#Split the data into training and testing
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# --------------------------------------------------
#Training Function using Bayesian Optimization
# --------------------------------------------------
def train_model(X_train, y_train):
    base_rf = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1
    )

    opt = BayesSearchCV(
        estimator=base_rf,
        search_spaces=search_space,
        n_iter=n_iter_bayes,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    opt.fit(X_train, y_train)
    return opt.best_estimator_

# --------------------------------------------------
#Train model
# --------------------------------------------------    
model = train_model(X_train, y_train)

pickle_path = os.path.join(PICKLE_DIR, "WaveHeight.pkl")
# pickle_path = os.path.join(PICKLE_DIR, "WavePeriod.pkl")
# pickle_path = os.path.join(PICKLE_DIR, "WindSpeed.pkl")
# pickle_path = os.path.join(PICKLE_DIR, "WaterLevel.pkl")

with open(pickle_path, "wb") as f:
    pickle.dump(model, f)

print(f"Saved model to: {pickle_path}")

# --------------------------------------------------
#Make Predictions
# --------------------------------------------------  
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_full = model.predict(X)

# --------------------------------------------------
#Compute rmse and r2 value metrics
# --------------------------------------------------  
train_rmse = root_mean_squared_error(y_train, y_pred_train)
test_rmse = root_mean_squared_error(y_test, y_pred_test)
full_rmse = root_mean_squared_error(y, y_pred_full)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
full_r2 = r2_score(y, y_pred_full)

# --------------------------------------------------
#Define Summary Table for Accuracy Metrics
# --------------------------------------------------  
def make_summary_table(train_rmse, test_rmse, full_rmse,
                       train_r2, test_r2, full_r2):

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")

    data = [
        ["Train", f"{train_rmse:.3f}", f"{train_r2:.3f}"],
        ["Test",  f"{test_rmse:.3f}", f"{test_r2:.3f}"],
        ["Full",  f"{full_rmse:.3f}", f"{full_r2:.3f}"]
    ]

    table = ax.table(
        cellText=data,
        colLabels=["Dataset", "RMSE", "R²"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    return fig

# --------------------------------------------------
#Define observed vs. predicted response line graph
# --------------------------------------------------  
def plot_observed_vs_predicted(y, y_pred, title="Observed vs Predicted"):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(y, label="Observed", linewidth=2)
    ax.plot(y_pred, label="Predicted", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Target Value")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig

# --------------------------------------------------
#Define observed vs. predicted scatter plot
# --------------------------------------------------
def plot_obs_pred_scatter(y, y_pred, title="Observed vs Predicted (Scatter + Linear Fit"):

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y, y_pred, alpha=0.6, label="Data")

    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Line")

    # Linear regression fit
    lr = LinearRegression()
    lr.fit(y.reshape(-1, 1), y_pred)
    y_fit = lr.predict(y.reshape(-1, 1))
    ax.plot(y, y_fit, color="black", linewidth=2, label="Linear Fit")

    # Labels and formatting
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig

# --------------------------------------------------
#Determine feature importance
# --------------------------------------------------  
def compute_perm_importance(model, X, y, label):
    """
    Computes permutation importance and returns a DataFrame.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    df_imp = pd.DataFrame({
        "feature": predictor_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "model": label
    })

    return df_imp.sort_values("importance_mean", ascending=False)

imp_train = compute_perm_importance(model, X_train, y_train, "train") #Training Set
imp_test = compute_perm_importance(model, X_test, y_test, "test") #Testing Set

# --------------------------------------------------
#Define feature importance plots
# --------------------------------------------------  
def plot_importance(df, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    df = df.sort_values("importance_mean", ascending=True)

    ax.barh(df["feature"], df["importance_mean"], xerr=df["importance_std"])
    ax.set_title(title)
    ax.set_xlabel("Permutation Importance")
    plt.tight_layout()
    return fig

# --------------------------------------------------
#Create and combine figures and export to PDF
# --------------------------------------------------     
summary_fig = make_summary_table(
    train_rmse, test_rmse, full_rmse,
    train_r2, test_r2, full_r2
)

scatter_test = plot_obs_pred_scatter(
    y_test,
    y_pred_test,
    title="Observed vs Predicted (Test Dataset)"
)
scatter_full = plot_obs_pred_scatter(
    y,
    y_pred_full,
    title="Observed vs Predicted (Full Dataset)"
)

line_full = plot_observed_vs_predicted(
    y,
    y_pred_full,
    title="Observed vs Predicted (Full Dataset)"
)

fig_imp_train = plot_importance(imp_train, "Permutation Importance – Train")
fig_imp_test  = plot_importance(imp_test,  "Permutation Importance – Test")

with PdfPages(output_pdf) as pdf:
    pdf.savefig(summary_fig)
    pdf.savefig(scatter_test)
    pdf.savefig(scatter_full)
    pdf.savefig(line_full)
    pdf.savefig(fig_imp_train)
    pdf.savefig(fig_imp_test)
    plt.close(summary_fig)
    plt.close(scatter_test)
    plt.close(scatter_full)
    plt.close(line_full)
    plt.close(fig_imp_train)
    plt.close(fig_imp_test)

print(f"PDF report saved to: {output_pdf}")
