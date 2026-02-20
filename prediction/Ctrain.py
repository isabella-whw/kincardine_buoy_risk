# --------------------------------------------------
#Ctrain creates random forest for directional data,
#using sin and cos components, and re-analysis using
#reconstruction and circular statitistics
# --------------------------------------------------

# --------------------------------------------------
#Import Requirements
# --------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from skopt import BayesSearchCV
from skopt.space import Integer, Real

from matplotlib.backends.backend_pdf import PdfPages

from Cstats import circ_rmse, circ_r2

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
circular_responses = ["Ds", "Dc"] #Include the sin and cos components of the predicted variable
# circular_responses = ["WDs", "WDc"] 

test_size = 0.2 #Split for testing set
random_state = 42
n_iter_bayes = 50 #Number of iterations
output_pdf = "circ_report_WaveDirection.pdf" #Specify output pdf for review
# output_pdf = "circ_report_WindDirection.pdf" 

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
y = df[circular_responses].to_numpy(dtype=float, copy=True)
y[(y<-1) | np.isin(y, outDat)] = np.nan

mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y).any(axis=1)) #Check to see if data is consistent
X = X[mask]
y = y[mask]

# --------------------------------------------------
#Split the data into sin, cos, training, and testing
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# y has two columns: sin (0) and cos (1)
y_train_sin = y_train[:, 0]
y_train_cos = y_train[:, 1]
y_test_sin  = y_test[:, 0]
y_test_cos  = y_test[:, 1]

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

#Train sin and cos models
# --------------------------------------------------    
model_sin = train_model(X_train, y_train_sin)
model_cos = train_model(X_train, y_train_cos)

pickle_path = os.path.join(PICKLE_DIR, "WaveDirection.pkl")
# pickle_path = os.path.join(PICKLE_DIR, "WindDirection.pkl")

with open(pickle_path, "wb") as f:
    pickle.dump(
        {
            "model_sin": model_sin,
            "model_cos": model_cos,
            "predictors": predictor_cols,
            "target_components": circular_responses
        },
        f
    )

print(f"Saved circular model bundle to: {pickle_path}")

# --------------------------------------------------
#Make Predictions
# --------------------------------------------------  
pred_sin_train = model_sin.predict(X_train)
pred_cos_train = model_cos.predict(X_train)

pred_sin_test = model_sin.predict(X_test)
pred_cos_test = model_cos.predict(X_test)

pred_sin_full = model_sin.predict(X)
pred_cos_full = model_cos.predict(X)

# --------------------------------------------------
#Convert sin and cos to degrees
# --------------------------------------------------
def reconstruct_angle(sin_vals, cos_vals):
    ang = np.degrees(np.arctan2(sin_vals, cos_vals))
    return (ang + 360) % 360

train_pred_deg = reconstruct_angle(pred_sin_train, pred_cos_train)
test_pred_deg  = reconstruct_angle(pred_sin_test, pred_cos_test)
full_pred_deg  = reconstruct_angle(pred_sin_full, pred_cos_full)

train_true_deg = reconstruct_angle(y_train_sin, y_train_cos)
test_true_deg  = reconstruct_angle(y_test_sin, y_test_cos)
full_true_deg  = reconstruct_angle(y[:,0], y[:,1])

# --------------------------------------------------
#Compute circular rmse and r2 value metrics
# --------------------------------------------------  
train_rmse = circ_rmse(train_true_deg, train_pred_deg)
test_rmse = circ_rmse(test_true_deg, test_pred_deg)
full_rmse = circ_rmse(full_true_deg, full_pred_deg)

train_r2 = circ_r2(train_true_deg, train_pred_deg)
test_r2 = circ_r2(test_true_deg, test_pred_deg)
full_r2 = circ_r2(full_true_deg, full_pred_deg)

# --------------------------------------------------
#Define polar-bar plots for comparing distributions
# --------------------------------------------------  
def plot_circular_distributions(true_deg, pred_deg, title, bins=36):
    """
    Creates a polar bar plot comparing true vs predicted angle distributions.
    """
    # Convert degrees to radians
    true_rad = np.deg2rad(true_deg)
    pred_rad = np.deg2rad(pred_deg)

    # Histogram bins
    edges = np.linspace(0, 2*np.pi, bins + 1)

    true_hist, _ = np.histogram(true_rad, bins=edges)
    pred_hist, _ = np.histogram(pred_rad, bins=edges)

    # Bar positions (center of each bin)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (2*np.pi) / bins

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Rotate so 0° is North
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Plot bars
    ax.bar(centers, true_hist, width=width, alpha=0.6, label="True", color="tab:blue")
    ax.bar(centers, pred_hist, width=width, alpha=0.6, label="Predicted", color="tab:orange")

    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    return fig

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
        colLabels=["Dataset", "Circular RMSE", "Circular R²"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    return fig

# --------------------------------------------------
#Determine feature importance for the model averages
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

imp_train_sin = compute_perm_importance(model_sin, X_train, y_train_sin, "sin_train") #Training Set
imp_train_cos = compute_perm_importance(model_cos, X_train, y_train_cos, "cos_train")

imp_test_sin = compute_perm_importance(model_sin, X_test, y_test_sin, "sin_test") #Testing Set
imp_test_cos = compute_perm_importance(model_cos, X_test, y_test_cos, "cos_test")

def combine_importances(imp_sin, imp_cos, label):
    df = imp_sin.merge(imp_cos, on="feature", suffixes=("_sin", "_cos"))
    df["importance_mean"] = (df["importance_mean_sin"] + df["importance_mean_cos"]) / 2
    df["importance_std"]  = (df["importance_std_sin"]  + df["importance_std_cos"]) / 2
    df["dataset"] = label
    return df[["feature", "importance_mean", "importance_std", "dataset"]]

imp_train = combine_importances(imp_train_sin, imp_train_cos, "train")
imp_test  = combine_importances(imp_test_sin,  imp_test_cos,  "test")

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

fig_train = plot_circular_distributions(
    train_true_deg, train_pred_deg, "Train Set Direction Distribution"
)
fig_test = plot_circular_distributions(
    test_true_deg, test_pred_deg, "Test Set Direction Distribution"
)
fig_full = plot_circular_distributions(
    full_true_deg, full_pred_deg, "Full Dataset Direction Distribution"
)

fig_imp_train = plot_importance(imp_train, "Permutation Importance – Train")
fig_imp_test  = plot_importance(imp_test,  "Permutation Importance – Test")

with PdfPages(output_pdf) as pdf:
    pdf.savefig(summary_fig)
    pdf.savefig(fig_train)
    pdf.savefig(fig_test)
    pdf.savefig(fig_full)
    pdf.savefig(fig_imp_train)
    pdf.savefig(fig_imp_test)
    plt.close(summary_fig)
    plt.close(fig_train)
    plt.close(fig_test)
    plt.close(fig_full)
    plt.close(fig_imp_train)
    plt.close(fig_imp_test)

print(f"PDF report saved to: {output_pdf}")
