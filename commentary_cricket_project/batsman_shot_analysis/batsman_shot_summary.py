import pandas as pd

class BatsmanShotSummary:
    """
    Class to accumulate and summarize shot data for a specific batsman.

    Usage:
        summary_obj = BatsmanShotSummary('Kohli')
        summary_obj.update('match1.xlsx')
        df = summary_obj.summary_df
        summary_obj.export_summary('kohli_summary.csv')
    """
    def __init__(self, batsman):
        self.batsman = batsman
        # DataFrame to store raw shot rows
        self._raw_df = pd.DataFrame(columns=[
            'Over_Ball', 'Bowler', 'Batsman', 'Runs', 'run_str', 'Commentary', 'Side'
        ])
        # Placeholder for processed summary
        self._summary_df = pd.DataFrame(
            columns=['Side', 'Low_Shots', 'High_Shots', 'Total_Runs', 'Runs_by_Low', 'Runs_by_High']
        )

    def update(self, new_data):
        """
        Append new match data and refresh the summary.
        new_data: pandas DataFrame or file path (Excel/CSV).
        """
        # Load incoming data
        if isinstance(new_data, pd.DataFrame):
            df_new = new_data.copy()
        elif isinstance(new_data, str):
            if new_data.lower().endswith(('.xlsx', '.xls')):
                df_new = pd.read_excel(new_data)
            elif new_data.lower().endswith('.csv'):
                df_new = pd.read_csv(new_data)
            else:
                raise ValueError("Unsupported file format: use Excel or CSV")
        else:
            raise TypeError("new_data must be a DataFrame or file path string")
        # print(f"Loaded new data for {self.batsman}: {new_data}")
        # Filter for batsman and append
        df_batsman = df_new[df_new['Batsman'] == self.batsman].copy()
        self._raw_df = pd.concat([self._raw_df, df_batsman], ignore_index=True)
        # Recompute summary
        self._process()

    def _process(self):
        """
        Internal: Clean raw data, classify intensity, and aggregate summary by Side.
        """
        df = self._raw_df.copy()
        # Extract numeric runs; ignore non-numeric entries
        df['Runs_Numeric'] = df['Runs'].astype(str).str.extract(r"(\d+)")
        df = df[df['Runs_Numeric'].notna()].copy()
        df['Runs'] = df['Runs_Numeric'].astype(int)
        df.drop(columns=['Runs_Numeric'], inplace=True)

        # Classify intensity and separate run counts
        df['Intensity'] = df['Runs'].apply(lambda r: 'low' if r <= 2 else 'high')
        df['Runs_Low'] = df['Runs'].where(df['Intensity'] == 'low',  0)
        df['Runs_High'] = df['Runs'].where(df['Intensity'] == 'high', 0)

        # Aggregate metrics by side
        self._summary_df = (
            df.groupby('Side')
              .agg(
                  Low_Shots    = ('Intensity', lambda x: (x == 'low').sum()),
                  High_Shots   = ('Intensity', lambda x: (x == 'high').sum()),
                  Total_Runs   = ('Runs', 'sum'),
                  Runs_by_Low  = ('Runs_Low', 'sum'),
                  Runs_by_High = ('Runs_High', 'sum')
              )
              .reset_index()
        )
        # print(f"Processed summary for {self.batsman}:")
        # print(self._summary_df)

    def _calculate_distribution(self, summary_df):
        """
        Compute distribution of high- and low-intensity shots by side,
        excluding 'out' and 'unknown position'.
        Returns a DataFrame indexed by Side with columns High_Shots and Low_Shots as proportions.
        """
        df = summary_df[~summary_df['Side'].isin(['out', 'unknown position'])].copy()
        df['High_Shots'] = df['High_Shots'].astype(int)
        df['Low_Shots'] = df['Low_Shots'].astype(int)

        high = df.set_index('Side')['High_Shots']
        low = df.set_index('Side')['Low_Shots']
        high_dist = high.div(high.sum() or 1)
        low_dist = low.div(low.sum() or 1)

        dist = pd.DataFrame({
            'High_Shots': high_dist,
            'Low_Shots': low_dist
        }).fillna(0)
        return dist
    
    @property
    def summary_df(self):
        """Public DataFrame of the current aggregated summary."""
        return self._summary_df.copy()

    def get_summary_dict(self):
        """Return summary as a list of dicts (one per side)."""
        return self._summary_df.to_dict(orient='records')

    def export_summary(self, filepath, format='csv', **kwargs):
        """
        Export the summary to a file. Supported formats: 'csv', 'json', 'xlsx'.
        Additional keyword args are passed to the pandas write function.
        """
        fmt = format.lower()
        if fmt == 'csv':
            self._summary_df.to_csv(filepath, index=False, **kwargs)
        elif fmt == 'json':
            self._summary_df.to_json(filepath, orient='records', **kwargs)
        elif fmt in ('xlsx', 'excel'):
            self._summary_df.to_excel(filepath, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
