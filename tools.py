import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import ensure_data_dir
import numpy as np
from sklearn.linear_model import LinearRegression

class DataSummaryTool:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        # try parse any date-like column
        self.date_col = None
        for c in self.df.columns:
            if 'date' in c.lower():
                try:
                    self.df[c] = pd.to_datetime(self.df[c])
                    self.date_col = c
                    break
                except Exception:
                    pass
        # if a date column exists, sort by it
        if self.date_col is not None:
            self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

    def show_head(self, n=5):
        return self.df.head(n).to_markdown()

    def describe(self):
        return self.df.describe(include='all').to_markdown()

    def columns(self):
        return ', '.join(self.df.columns.tolist())

    def generate_plot(self, query_text: str):
        # Very small parser: try to detect two column names in the query and plot them
        cols = self.df.columns.tolist()
        selected = [c for c in cols if c.lower() in query_text.lower()]
        # fallback heuristics
        if len(selected) >= 2:
            x, y = selected[0], selected[1]
        elif len(selected) == 1:
            x, y = selected[0], cols[1] if len(cols)>1 else selected[0]
        else:
            x, y = cols[0], cols[1] if len(cols)>1 else cols[0]
        fig, ax = plt.subplots()
        try:
            ax.scatter(self.df[x], self.df[y])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f'{y} vs {x}')
        except Exception:
            fig, ax = plt.subplots()
            self.df.head(20).plot(kind='bar', ax=ax)
        ensure_data_dir()
        out_path = os.path.join('data', 'last_plot.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path

    def analyze_trends(self):
        """Perform automatic trend analysis on the dataframe.
        Returns a dict with 'text' (summary string) and 'plots' (list of file paths).
        """
        ensure_data_dir()
        df = self.df.copy()
        plots = []
        lines = []

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        if len(numeric_cols) == 0:
            lines.append('No numeric columns found for trend analysis.')
            return {'text': '\n'.join(lines), 'plots': plots}

        # If date_col present, compute temporal trends
        if self.date_col is not None:
            # create numeric index for regression
            df['__t'] = np.arange(len(df)).reshape(-1,1)
            # compute percent change from first to last for numeric cols
            pct_changes = {}
            for c in numeric_cols:
                try:
                    start = float(df[c].iloc[0])
                    end = float(df[c].iloc[-1])
                    pct = None
                    if start != 0:
                        pct = (end - start) / abs(start) * 100.0
                    else:
                        pct = float('nan')
                    pct_changes[c] = pct
                except Exception:
                    pct_changes[c] = float('nan')
            # sort by percent change
            sorted_pct = sorted(pct_changes.items(), key=lambda x: (np.nan_to_num(x[1], nan=-1e9)), reverse=True)
            lines.append('Top changes (first -> last) for numeric columns:')
            for col, pct in sorted_pct[:5]:
                if np.isfinite(pct):
                    lines.append(f'- {col}: {pct:.2f}% change')
                else:
                    lines.append(f'- {col}: change unknown (non-numeric or zero start)')

            # run simple linear regression over time to detect upward/downward trends
            trends = {}
            for c in numeric_cols:
                try:
                    y = df[c].fillna(method='ffill').fillna(0).values.reshape(-1,1)
                    X = df['__t'].values.reshape(-1,1)
                    model = LinearRegression().fit(X, y)
                    slope = float(model.coef_[0])
                    trends[c] = slope
                except Exception:
                    trends[c] = 0.0
            # report strongest slopes
            sorted_slopes = sorted(trends.items(), key=lambda x: x[1], reverse=True)
            lines.append('\nDetected linear trends (slope of regression over time):')
            for col, slope in sorted_slopes[:5]:
                lines.append(f'- {col}: slope={slope:.6f}')

            # If 'sales' or first numeric column exists, plot time series
            preferred = None
            for candidate in ['sales', 'revenue', 'price', 'units']:
                if candidate in [c.lower() for c in numeric_cols]:
                    preferred = next((c for c in numeric_cols if c.lower()==candidate), None)
                    break
            if preferred is None:
                preferred = numeric_cols[0]

            try:
                fig, ax = plt.subplots()
                ax.plot(df[self.date_col], df[preferred])
                ax.set_title(f'Time series of {preferred}')
                ax.set_xlabel(self.date_col)
                ax.set_ylabel(preferred)
                ppath = os.path.join('data', f'trend_{preferred}.png')
                fig.savefig(ppath, bbox_inches='tight')
                plt.close(fig)
                plots.append(ppath)
            except Exception:
                pass

            # If there's a categorical column like 'region' produce regional breakdown over time
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols)>0 and self.date_col is not None:
                cat = cat_cols[0]
                try:
                    g = df.groupby([cat, self.date_col])[numeric_cols[0]].sum().unstack(level=0).fillna(0)
                    fig, ax = plt.subplots()
                    for col in g.columns:
                        ax.plot(g.index, g[col], label=str(col))
                    ax.set_title(f'{numeric_cols[0]} by {cat} over time')
                    ax.set_xlabel(self.date_col)
                    ax.set_ylabel(numeric_cols[0])
                    ax.legend()
                    ppath = os.path.join('data', f'{numeric_cols[0]}_by_{cat}.png')
                    fig.savefig(ppath, bbox_inches='tight')
                    plt.close(fig)
                    plots.append(ppath)
                except Exception:
                    pass
        else:
            # No date column: analyze correlations and top contributors
            corr = df[numeric_cols].corr().abs()
            pairs = []
            for i, c1 in enumerate(numeric_cols):
                for j, c2 in enumerate(numeric_cols):
                    if i>=j: continue
                    pairs.append((c1, c2, corr.loc[c1,c2]))
            pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
            lines.append('Top correlations (abs):')
            for c1, c2, v in pairs_sorted[:5]:
                lines.append(f'- {c1} vs {c2}: r={v:.3f}')
            try:
                col = numeric_cols[0]
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=20)
                ax.set_title(f'Histogram of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('count')
                ppath = os.path.join('data', f'hist_{col}.png')
                fig.savefig(ppath, bbox_inches='tight')
                plt.close(fig)
                plots.append(ppath)
            except Exception:
                pass

        summary = '\n'.join(lines)
        return {'text': summary, 'plots': plots}
