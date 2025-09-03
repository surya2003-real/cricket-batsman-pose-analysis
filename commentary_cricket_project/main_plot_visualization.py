import pandas as pd
from batsman_shot_analysis.batsman_shot_summary import BatsmanShotSummary
from batsman_shot_analysis.area_visualisation import plot_shot_distribution_gradient

if __name__ == "__main__":
    batsman = "Kohli"

    for i in range(1, 14):
        print(f"Processing file: cricket_scores_{i}.xlsx")
        filename = f"./cricket_matches/cricket_scores_{i}.xlsx"
        summary = BatsmanShotSummary(batsman)
        summary.update(filename)

        # Compute and save distribution
        dist = summary._calculate_distribution(summary.summary_df)
        print(f"Final distribution for {batsman}:")
        print(dist)
        # Plot and save figure
        plot_shot_distribution_gradient(dist, f"./plots/{batsman}_distribution_{i}.png")

    df = pd.read_csv("Kohli_distribution.csv")
    plot_shot_distribution_gradient(df, "./plots/Kohli_distribution_final.png")