import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from Cstats import circ_diff_deg
from Thumbnail3 import pred_haz


# ------------------------------------------------------------
# Load and prepare data
df = pd.read_csv("NOAA_2019to2024_predictions_12h.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df["year"] = df["datetime"].dt.year
df = df[df['year'].isin([2019, 2020, 2021, 2022, 2023, 2024])]

df["wave_dir_deg"] = abs(circ_diff_deg(315, df["wave_dir_deg"]))
df["wind_dir_deg"] = abs(circ_diff_deg(315, df["wind_dir_deg"]))

# Run hazard model
results = pred_haz(df[[
    "wave_height_m",
    "wave_dir_deg",
    "wave_period_s",
    "max_wave_height_12h_m",
    "wind_speed_ms",
]])

#Add warnings to the database
df_full = pd.concat([df, results], axis=1)


# ------------------------------------------------------------
# Helper: convert a DataFrame into a figure for PDF export
def table_to_figure(df_table, title="", figsize=(10, 2.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)
    ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        rowLabels=df_table.index,
        loc="center"
    )
    return fig


# ------------------------------------------------------------
# Plot function
def plot_wave_and_hazard(df, year):
    df_year = df[df['year'] == year].sort_values('datetime')

    x_min = df_year['datetime'].min()
    x_max = df_year['datetime'].max()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df_year['datetime'], df_year['wave_height_m'],
             color='tab:blue', label='Wave Height (m)', linewidth=0.5)
    ax1.set_xlim([x_min, x_max])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Wave Height (m)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(df_year['datetime'], df_year['total_score'],
             color='tab:red', label='Total Score', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Total Hazard Score', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    for thresh in [3, 7, 11]:
        ax2.hlines(
            y=thresh,
            xmin=x_min,
            xmax=x_max,
            colors='k',
            linestyles='dotted',
            linewidth=2,
            label=f'Threshold {thresh}'
        )

    plt.title(f"Wave Height and Hazard Score — {year}")
    fig.tight_layout()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    return fig


# ------------------------------------------------------------
# Build enhanced tables
df_full = pd.concat([df, results], axis=1)

class_order = ["Low", "Moderate", "High", "Extreme"]
df_full["risk_level"] = pd.Categorical(
    df_full["risk_level"],
    categories=class_order,
    ordered=True
)

# Table 1: total hazard class counts + column percentages
tbl1 = df_full["risk_level"].value_counts().to_frame("count")
tbl1["percent"] = (tbl1["count"] / tbl1["count"].sum() * 100).round(1).astype(str) + "%"

# Table 2: hazard class counts per year + row percentages
tbl2 = df_full.groupby(["year", "risk_level"]).size().unstack(fill_value=0).reindex(columns=class_order, fill_value=0)
row_totals = tbl2.sum(axis=1)
tbl2_percent = (tbl2.div(row_totals, axis=0) * 100).round(1).astype(str) + "%"
tbl2_combined = tbl2.astype(str) + " (" + tbl2_percent + ")"

# Table 3: mean hazard values by class (rounded to 2 decimals)
tbl3 = df_full.groupby("risk_level")[["wave_height_m", "wave_period_s", "wind_speed_ms", "max_wave_height_12h_m"]].mean().round(2).reindex(class_order)


# ------------------------------------------------------------
# Create the PDF
model_name = pred_haz.__module__   # e.g., "GLRCC"
pdf_filename = f"{model_name}_output.pdf"

with PdfPages(pdf_filename) as pdf:

    # Table 1
    fig1 = table_to_figure(tbl1, title="Total Hazard Class Counts")
    pdf.savefig(fig1)
    plt.close(fig1)

    # Table 2
    fig2 = table_to_figure(tbl2_combined, title="Hazard Class Counts per Year", figsize=(10, 4))
    pdf.savefig(fig2)
    plt.close(fig2)

    # Table 3
    fig3 = table_to_figure(tbl3, title="Mean Hazard Values by Class", figsize=(10, 3))
    pdf.savefig(fig3)
    plt.close(fig3)

    # Boxplot
    fig4, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_full, x="year", y="total_score", ax=ax)
    ax.set_title("Distribution of Hazard Scores by Year")
    ymax = max(df_full["total_score"].max(), 12)
    ax.set_ylim(-0.5, ymax)
    ax.axhline(3, color="red", linestyle="--", linewidth=1)
    ax.axhline(7, color="red", linestyle="--", linewidth=1)
    ax.axhline(11, color="red", linestyle="--", linewidth=1)
    pdf.savefig(fig4)
    plt.close(fig4)

    # Table 5
    photo_table = pd.read_csv("photo_hazard_table.csv")
    fig5 = table_to_figure(
        photo_table,
        title="Predicted Hazard vs Photo Response",
        figsize=(12, 4)
    )
    pdf.savefig(fig5)
    plt.close(fig5)

    # Pairplot
    pair = sns.pairplot(
        df_full,
        vars=["wave_height_m", "max_wave_height_12h_m", "wave_period_s", "wind_speed_ms", "total_score"],
        hue="risk_level",
        palette={
            "Low": "green",
            "Moderate": "orange",
            "High": "red",
            "Extreme": "purple"
        }
    )
    pdf.savefig(pair.fig)
    plt.close(pair.fig)

    # Yearly wave + hazard plots
    #for yr in sorted(df_full["year"].unique()):
    for yr in sorted(df_full["year"].dropna().unique()):
        fig_year = plot_wave_and_hazard(df_full, yr)
        if fig_year is not None:
            pdf.savefig(fig_year)
            plt.close(fig_year)
