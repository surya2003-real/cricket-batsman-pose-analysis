import pandas as pd
from batsman_shot_analysis.batsman_shot_summary import BatsmanShotSummary

class BatsmanAnalysis(BatsmanShotSummary):
    """
    Extends BatsmanShotSummary to analyze stability of high-intensity shot distribution.
    Inherits data ingestion and summary aggregation; adds distribution stability check.
    """
    def __init__(self, batsman, threshold=0.05):
        super().__init__(batsman)
        self.threshold = threshold
        self.prev_distribution = None
        self._base_summary = None

    def update(self, new_data):
        """
        Ingest new match data and update the shot summary.
        Returns True once the high-shot distribution stabilizes within the threshold.
        """
        # Append new data via parent and build updated summary
        super().update(new_data)
        summary_df = self.summary_df.copy()

        # Initialize base summary if first update
        if self._base_summary is None:
            self._base_summary = summary_df.copy()
        else:
            # Accumulate counts into base summary
            self._base_summary = (
                self._base_summary.set_index('Side')
                .add(summary_df.set_index('Side'), fill_value=0)
                .reset_index()
            )

        # Calculate distributions
        new_dist = self._calculate_distribution(self._base_summary)

        # Check for stabilization
        if self.prev_distribution is not None:
            deviation = self._calculate_deviation(self.prev_distribution, new_dist)
            print(f"Current deviation: {deviation:.4f}")
            if deviation < self.threshold:
                print(f"Distribution stabilized for {self.batsman} with deviation: {deviation:.4f}")
                print(self.base_summary)
                return True

        # Update prev_distribution and report
        self.prev_distribution = new_dist.copy()
        return False

    def _calculate_deviation(self, dist1, dist2):
        """
        Calculate maximum absolute difference between two distributions.
        """
        return (dist1.subtract(dist2).abs().max().max())

    @property
    def base_summary(self):
        """
        Return a copy of the accumulating base summary DataFrame.
        """
        if self._base_summary is None:
            self._base_summary = self.summary_df.copy()
        return self._base_summary.copy()

    def export_base_summary(self, filepath):
        """Export the base summary to a CSV file."""
        self.base_summary.to_csv(filepath, index=False)
        print(f"Base summary exported to {filepath}")

    def export_distribution(self, filepath):
        """Export the latest distribution to a CSV file."""
        if self.prev_distribution is not None:
            self.prev_distribution.to_csv(filepath)
            print(f"Distribution exported to {filepath}")
        else:
            print("No distribution data available for export.")
