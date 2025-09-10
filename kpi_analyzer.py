# utils/kpi_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Try to import scipy, but provide fallbacks if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def calculate_kpis(df):
    """
    Calculate key performance indicators for the dataset.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
        dict: A dictionary of KPIs.
    """
    kpis = {}
    
    # Basic KPIs
    kpis["row_count"] = len(df)
    kpis["column_count"] = len(df.columns)
    kpis["missing_values"] = df.isnull().sum().sum()
    kpis["duplicate_rows"] = df.duplicated().sum()
    kpis["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # Numeric KPIs
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        kpis["numeric_columns"] = len(numeric_cols)
        
        # Calculate statistics for each numeric column
        for col in numeric_cols:
            col_kpis = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "range": df[col].max() - df[col].min(),
                "variance": df[col].var(),
            }
            
            # Add scipy-specific stats if available
            if SCIPY_AVAILABLE:
                col_kpis["skewness"] = stats.skew(df[col].dropna())
                col_kpis["kurtosis"] = stats.kurtosis(df[col].dropna())
            else:
                # Simple fallbacks for skewness and kurtosis
                col_kpis["skewness"] = "N/A (scipy not available)"
                col_kpis["kurtosis"] = "N/A (scipy not available)"
                
            kpis[f"{col}_stats"] = col_kpis
    
    # Categorical KPIs
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        kpis["categorical_columns"] = len(categorical_cols)
        
        # Calculate statistics for each categorical column
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            col_kpis = {
                "unique_values": df[col].nunique(),
                "most_common": value_counts.index[0] if not value_counts.empty else None,
                "most_common_count": value_counts.iloc[0] if not value_counts.empty else 0,
                "least_common": value_counts.index[-1] if not value_counts.empty else None,
                "least_common_count": value_counts.iloc[-1] if not value_counts.empty else 0,
            }
            
            # Add entropy if scipy is available
            if SCIPY_AVAILABLE:
                col_kpis["entropy"] = stats.entropy(value_counts)
            else:
                col_kpis["entropy"] = "N/A (scipy not available)"
                
            kpis[f"{col}_stats"] = col_kpis
    
    # Date KPIs
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
    if not date_cols.empty:
        kpis["date_columns"] = len(date_cols)
        
        # Calculate statistics for each date column
        for col in date_cols:
            col_kpis = {
                "min_date": df[col].min(),
                "max_date": df[col].max(),
                "date_range_days": (df[col].max() - df[col].min()).days,
                "most_common_year": df[col].dt.year.mode()[0] if not df[col].dt.year.mode().empty else None,
                "most_common_month": df[col].dt.month.mode()[0] if not df[col].dt.month.mode().empty else None,
                "most_common_day": df[col].dt.day.mode()[0] if not df[col].dt.day.mode().empty else None,
            }
            kpis[f"{col}_stats"] = col_kpis
    else:
        # If no datetime columns found, check for object columns that might be dates
        potential_date_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors='raise')
                    potential_date_cols.append(col)
                except:
                    pass
        
        if potential_date_cols:
            kpis["date_columns"] = len(potential_date_cols)
            for col in potential_date_cols:
                kpis[f"{col}_stats"] = {
                    "note": "Detected as potential date column but not converted to datetime"
                }
        else:
            kpis["date_columns"] = 0
    
    return kpis

def detect_anomalies(df, method='iqr', threshold=1.5):
    """
    Detect anomalies in numeric columns using the specified method.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        method (str): The method to use for anomaly detection ('iqr' or 'zscore').
        threshold (float): The threshold for anomaly detection.
    
    Returns:
        dict: A dictionary of anomalies for each column.
    """
    anomalies = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outliers = col_data[outlier_mask]
            
            anomalies[col] = {
                "method": "IQR",
                "count": len(outliers),
                "percentage": (len(outliers) / len(col_data)) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outliers": outliers.tolist()[:10]  # Limit to first 10 outliers
            }
        
        elif method == 'zscore' and SCIPY_AVAILABLE:
            z_scores = np.abs(stats.zscore(col_data))
            outlier_mask = z_scores > threshold
            outliers = col_data[outlier_mask]
            
            anomalies[col] = {
                "method": "Z-score",
                "count": len(outliers),
                "percentage": (len(outliers) / len(col_data)) * 100,
                "threshold": threshold,
                "outliers": outliers.tolist()[:10]  # Limit to first 10 outliers
            }
        elif method == 'zscore' and not SCIPY_AVAILABLE:
            # Simple fallback for z-score without scipy
            mean = col_data.mean()
            std = col_data.std()
            z_scores = np.abs((col_data - mean) / std)
            outlier_mask = z_scores > threshold
            outliers = col_data[outlier_mask]
            
            anomalies[col] = {
                "method": "Z-score (simple)",
                "count": len(outliers),
                "percentage": (len(outliers) / len(col_data)) * 100,
                "threshold": threshold,
                "outliers": outliers.tolist()[:10]  # Limit to first 10 outliers
            }
    
    return anomalies

def generate_executive_insights(df, kpis, anomalies):
    """
    Generate executive insights based on the data analysis.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        kpis (dict): The KPIs calculated from the data.
        anomalies (dict): The anomalies detected in the data.
    
    Returns:
        list: A list of executive insights.
    """
    insights = []
    
    # Basic dataset insights
    insights.append(f"The dataset contains {kpis['row_count']:,} records and {kpis['column_count']} columns.")
    
    if kpis['missing_values'] > 0:
        missing_pct = (kpis['missing_values'] / (kpis['row_count'] * kpis['column_count'])) * 100
        insights.append(f"There are {kpis['missing_values']:,} missing values ({missing_pct:.2f}% of all data points).")
    else:
        insights.append("The dataset has no missing values.")
    
    if kpis['duplicate_rows'] > 0:
        dup_pct = (kpis['duplicate_rows'] / kpis['row_count']) * 100
        insights.append(f"There are {kpis['duplicate_rows']:,} duplicate rows ({dup_pct:.2f}% of all records).")
    else:
        insights.append("The dataset has no duplicate rows.")
    
    # Numeric column insights
    if "numeric_columns" in kpis and kpis["numeric_columns"] > 0:
        insights.append(f"The dataset contains {kpis['numeric_columns']} numeric columns.")
        
        # Find column with highest variance
        max_var_col = None
        max_var = -1
        for col in df.select_dtypes(include=np.number).columns:
            if f"{col}_stats" in kpis and kpis[f"{col}_stats"]["variance"] > max_var:
                max_var = kpis[f"{col}_stats"]["variance"]
                max_var_col = col
        
        if max_var_col:
            insights.append(f"The column '{max_var_col}' shows the highest variability in the data.")
    
    # Categorical column insights
    if "categorical_columns" in kpis and kpis["categorical_columns"] > 0:
        insights.append(f"The dataset contains {kpis['categorical_columns']} categorical columns.")
        
        # Find column with highest entropy (most diverse)
        max_entropy_col = None
        max_entropy = -1
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if f"{col}_stats" in kpis:
                entropy = kpis[f"{col}_stats"]["entropy"]
                if isinstance(entropy, (int, float)) and entropy > max_entropy:
                    max_entropy = entropy
                    max_entropy_col = col
        
        if max_entropy_col:
            insights.append(f"The column '{max_entropy_col}' shows the highest diversity of values.")
    
    # Date column insights
    if "date_columns" in kpis and kpis["date_columns"] > 0:
        insights.append(f"The dataset contains {kpis['date_columns']} date columns.")
        
        for col in df.select_dtypes(include=['datetime']).columns:
            if f"{col}_stats" in kpis:
                date_range_days = kpis[f"{col}_stats"]["date_range_days"]
                min_date = kpis[f"{col}_stats"]["min_date"]
                max_date = kpis[f"{col}_stats"]["max_date"]
                
                # Check if dates are valid (not NaT)
                if pd.isna(min_date) or pd.isna(max_date):
                    insights.append(f"Date column '{col}' contains invalid or missing date values.")
                else:
                    insights.append(f"Date column '{col}' spans {date_range_days} days from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
    
    # Anomaly insights
    total_anomalies = sum(anomaly_info["count"] for anomaly_info in anomalies.values())
    if total_anomalies > 0:
        insights.append(f"Detected {total_anomalies} potential anomalies across numeric columns.")
        
        # Find column with most anomalies
        max_anom_col = None
        max_anom_count = 0
        for col, anomaly_info in anomalies.items():
            if anomaly_info["count"] > max_anom_count:
                max_anom_count = anomaly_info["count"]
                max_anom_col = col
        
        if max_anom_col:
            anom_pct = anomalies[max_anom_col]["percentage"]
            insights.append(f"Column '{max_anom_col}' has the most anomalies ({max_anom_count} records, {anom_pct:.2f}%).")
    
    return insights

def create_kpi_dashboard(kpis, anomalies):
    """
    Create a dashboard with KPI visualizations.
    
    Args:
        kpis (dict): The KPIs calculated from the data.
        anomalies (dict): The anomalies detected in the data.
    
    Returns:
        dict: A dictionary of Plotly figures.
    """
    figures = {}
    
    # Create a summary figure
    fig_summary = go.Figure()
    
    # Add basic KPIs as indicators
    fig_summary.add_trace(go.Indicator(
        mode="number+delta",
        value=kpis["row_count"],
        title={"text": "Total Rows"},
        domain={'row': 0, 'column': 0}
    ))
    
    fig_summary.add_trace(go.Indicator(
        mode="number+delta",
        value=kpis["column_count"],
        title={"text": "Total Columns"},
        domain={'row': 0, 'column': 1}
    ))
    
    fig_summary.add_trace(go.Indicator(
        mode="number+delta",
        value=kpis["missing_values"],
        title={"text": "Missing Values"},
        domain={'row': 1, 'column': 0}
    ))
    
    fig_summary.add_trace(go.Indicator(
        mode="number+delta",
        value=kpis["duplicate_rows"],
        title={"text": "Duplicate Rows"},
        domain={'row': 1, 'column': 1}
    ))
    
    fig_summary.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=400,
        title_text="Dataset Summary KPIs"
    )
    
    figures["summary"] = fig_summary
    
    # Create anomaly figure if anomalies exist
    if anomalies:
        anom_cols = list(anomalies.keys())
        anom_counts = [anomalies[col]["count"] for col in anom_cols]
        
        fig_anomalies = px.bar(
            x=anom_cols,
            y=anom_counts,
            title="Anomaly Counts by Column",
            labels={'x': 'Column', 'y': 'Anomaly Count'}
        )
        
        figures["anomalies"] = fig_anomalies
    
    # Create column type distribution figure
    col_types = {
        "Numeric": kpis.get("numeric_columns", 0),
        "Categorical": kpis.get("categorical_columns", 0),
        "Date": kpis.get("date_columns", 0),
        "Other": kpis["column_count"] - kpis.get("numeric_columns", 0) - kpis.get("categorical_columns", 0) - kpis.get("date_columns", 0)
    }
    
    # Create a DataFrame from the col_types dictionary
    col_types_df = pd.DataFrame({
        "Column Type": list(col_types.keys()),
        "Count": list(col_types.values())
    })
    
    # Create the pie chart using the DataFrame
    fig_col_types = px.pie(
        col_types_df, 
        values="Count", 
        names="Column Type",
        title="Column Type Distribution",
        hover_data=["Count"]
    )
    fig_col_types.update_traces(textposition='inside', textinfo='percent+label')
    
    figures["column_types"] = fig_col_types
    
    return figures