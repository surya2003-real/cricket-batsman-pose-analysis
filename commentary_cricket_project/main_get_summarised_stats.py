from batsman_shot_analysis.batsman_analysis import BatsmanAnalysis
from batsman_shot_analysis.area_visualisation import plot_shot_distribution_gradient
import pandas as pd

# Initialize BatsmanAnalysis for Virat Kohli
batsman_analysis = BatsmanAnalysis(batsman='Kohli', threshold=0.005)

for i in range(1, 14):
    # Load the summary data for batsman analysis
    filename = f"./cricket_matches/cricket_scores_{i}.xlsx"
    print(f"Loading data from {filename}...")
    df = pd.read_excel(filename)
    # Check if distribution stabilized
    if batsman_analysis.update(df):
        batsman_analysis.export_base_summary(f"{batsman_analysis.batsman}_summary_stable.csv")
        batsman_analysis.export_distribution(f"{batsman_analysis.batsman}_distribution.csv")
        print(f"Stabilized at file {i}.")
        break

    plot_shot_distribution_gradient(batsman_analysis.prev_distribution, f"./plots_consecutive/Kohli_distribution_{i}.png")