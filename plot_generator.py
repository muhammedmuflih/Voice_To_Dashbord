# utils/plot_generator.py

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import numpy as np
from scipy import stats

# Try to import scipy, but provide fallbacks if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def generate_chart(df: pd.DataFrame, query_details: dict):
    """
    Generates a Plotly chart based on the mentioned columns and desired chart type.
    This version is more flexible and can handle 1, 2, or 3 columns intelligently.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        query_details (dict): A dictionary with chart details (columns, chart_type).
        
    Returns:
        tuple: A tuple containing a Plotly figure and an explanation string.
               Returns (None, error_message) on failure.
    """
    
    columns_to_plot = query_details.get("columns", [])
    chart_type = query_details.get("chart_type", "auto")
    analysis_type = query_details.get("analysis_type", None)

    # Filter out columns not found in the DataFrame
    columns_to_plot = [col for col in columns_to_plot if col in df.columns]

    if not columns_to_plot:
        # Try to suggest columns based on the query
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if numeric_cols:
            columns_to_plot = [numeric_cols[0]]
            if len(numeric_cols) > 1:
                columns_to_plot.append(numeric_cols[1])
            elif categorical_cols:
                columns_to_plot.append(categorical_cols[0])
        elif categorical_cols:
            columns_to_plot = [categorical_cols[0]]
        elif date_cols:
            columns_to_plot = [date_cols[0]]
            if numeric_cols:
                columns_to_plot.append(numeric_cols[0])
        
        if not columns_to_plot:
            return None, "I couldn't find any suitable columns for visualization. Please check your data or try a different query."
            
    try:
        # Handle special analysis types
        if analysis_type == "summary":
            return generate_summary_chart(df, columns_to_plot)
        elif analysis_type == "outlier":
            return generate_outlier_chart(df, columns_to_plot)
        elif analysis_type == "kpi":
            return generate_kpi_chart(df, columns_to_plot)
        
        # Apply Filtering First
        filter_details = query_details.get("filter")
        aggregation = query_details.get("aggregation")

        if filter_details:
            col = filter_details.get("column")
            op = filter_details.get("operator")
            val = filter_details.get("value")
            
            if col in df.columns:
                try:
                    # Convert the value to the column's dtype for comparison
                    if pd.api.types.is_numeric_dtype(df[col]):
                        val = pd.to_numeric(val, errors='ignore')
                    
                    if op == "equals":
                        df = df[df[col] == val]
                    elif op == "greater_than":
                        df = df[df[col] > val]
                    elif op == "less_than":
                        df = df[df[col] < val]
                    elif op == "contains":
                        df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
                    elif op == "not_equal":
                        df = df[df[col] != val]
                    elif op == "top":
                        df = df.nlargest(int(val) if val.isdigit() else 5, col)
                    elif op == "bottom":
                        df = df.nsmallest(int(val) if val.isdigit() else 5, col)
                    
                    # Check if filtering resulted in empty dataframe
                    if df.empty:
                        return None, f"No data found after applying filter: {col} {op} {val}. Please try a different filter."
                except Exception as e:
                    st.warning(f"Could not apply filter: {e}")

        # Apply Aggregation
        if aggregation and len(columns_to_plot) >= 2:
            group_by_col = columns_to_plot[0]
            agg_col = columns_to_plot[1]
            
            if group_by_col in df.columns and agg_col in df.columns:
                if aggregation == "sum":
                    df = df.groupby(group_by_col)[agg_col].sum().reset_index()
                    df.rename(columns={agg_col: f"Total {agg_col}"}, inplace=True)
                    explanation = f"Displaying the total **{agg_col}** by **{group_by_col}**."
                elif aggregation == "average":
                    df = df.groupby(group_by_col)[agg_col].mean().reset_index()
                    df.rename(columns={agg_col: f"Average {agg_col}"}, inplace=True)
                    explanation = f"Displaying the average **{agg_col}** by **{group_by_col}**."
                elif aggregation == "count":
                    df = df.groupby(group_by_col)[agg_col].count().reset_index()
                    df.rename(columns={agg_col: f"Count of {agg_col}"}, inplace=True)
                    explanation = f"Displaying the count of **{agg_col}** by **{group_by_col}**."
                elif aggregation == "max":
                    df = df.groupby(group_by_col)[agg_col].max().reset_index()
                    df.rename(columns={agg_col: f"Max {agg_col}"}, inplace=True)
                    explanation = f"Displaying the maximum **{agg_col}** by **{group_by_col}**."
                elif aggregation == "min":
                    df = df.groupby(group_by_col)[agg_col].min().reset_index()
                    df.rename(columns={agg_col: f"Min {agg_col}"}, inplace=True)
                    explanation = f"Displaying the minimum **{agg_col}** by **{group_by_col}**."
                
                # Update columns to plot for the new aggregated DataFrame
                columns_to_plot = [group_by_col, df.columns[-1]]
        
        # Existing Plotting Logic
        num_cols = [c for c in columns_to_plot if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in columns_to_plot if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))]
        date_cols = [c for c in columns_to_plot if pd.api.types.is_datetime64_any_dtype(df[c])]

        fig = None
        explanation = ""

        if chart_type == "line":
            if len(num_cols) >= 1 and len(date_cols) >= 1:
                x_col, y_col = date_cols[0], num_cols[0]
                fig = px.line(df, x=x_col, y=y_col, title=f"Trend of {y_col} over {x_col}")
                explanation = f"Showing a line chart of **{y_col}** over **{x_col}**."
            else:
                # Try to create a line chart with what we have
                if len(num_cols) >= 1:
                    # Use index as x-axis
                    fig = px.line(df, y=num_cols[0], title=f"Trend of {num_cols[0]}")
                    explanation = f"Showing a line chart of **{num_cols[0]}** over the index."
                else:
                    return None, "Line chart requires at least one numeric column. Try a different chart type."

        elif chart_type == "bar":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                x_col, y_col = cat_cols[0], num_cols[0]
                fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}")
                explanation = f"Showing a bar chart comparing **{y_col}** across different **{x_col}** values."
            elif len(cat_cols) >= 1:
                fig = px.bar(df[cat_cols[0]].value_counts().reset_index(),
                             x=cat_cols[0], y='count',
                             title=f"Count of {cat_cols[0]}",
                             labels={'x': cat_cols[0], 'y': 'Count'})
                explanation = f"Showing a bar chart of the count for each category in **{cat_cols[0]}**."
            else:
                return None, "Bar chart requires at least one categorical column. Try a different chart type."

        elif chart_type == "scatter":
            if len(num_cols) >= 2:
                x_col, y_col = num_cols[0], num_cols[1]
                color_col = cat_cols[0] if len(cat_cols) >= 1 else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot of {y_col} vs {x_col}")
                explanation = f"Showing a scatter plot of **{y_col}** versus **{x_col}**."
                if color_col:
                    explanation += f" The data is colored by **{color_col}**."
            else:
                return None, "Scatter plot requires at least two numeric columns. Try a different chart type."
        
        elif chart_type == "pie":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                names_col, values_col = cat_cols[0], num_cols[0]
                fig = px.pie(df, names=names_col, values=values_col, title=f"Pie Chart of {values_col} by {names_col}")
                explanation = f"Showing a pie chart of the proportion of **{values_col}** for each **{names_col}**."
            elif len(cat_cols) >= 1:
                fig = px.pie(df, names=cat_cols[0], title=f"Distribution of {cat_cols[0]}")
                explanation = f"Showing a pie chart of the distribution of **{cat_cols[0]}**."
            else:
                return None, "Pie chart requires at least one categorical column. Try a different chart type."
        
        elif chart_type == "histogram":
            if len(num_cols) >= 1:
                x_col = num_cols[0]
                color_col = cat_cols[0] if len(cat_cols) >= 1 else None
                fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
                explanation = f"Showing the distribution of **{x_col}**."
            else:
                return None, "Histogram requires at least one numeric column. Try a different chart type."

        elif "map" in chart_type:
            if 'lat' in df.columns or 'lon' in df.columns:
                lat_col = next((c for c in columns_to_plot if 'lat' in c.lower()), None)
                lon_col = next((c for c in columns_to_plot if 'lon' in c.lower()), None)
                
                if lat_col and lon_col:
                    fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, 
                                            zoom=1, height=400,
                                            mapbox_style="carto-positron",
                                            title="Geographical Distribution")
                    explanation = f"Displaying a map showing locations using **{lat_col}** and **{lon_col}**."
                else:
                    return None, "Map visualization requires both latitude and longitude columns. Try a different chart type."
            else:
                return None, "Map visualization requires columns with 'lat' and 'lon' keywords. Try a different chart type."

        else:
            # If no specific chart type is matched, try to determine the best chart type
            if len(num_cols) >= 2 and len(cat_cols) >= 1:
                fig = px.scatter(df, x=num_cols[0], y=num_cols[1], color=cat_cols[0], 
                               title=f"Relationship between {num_cols[0]} and {num_cols[1]}")
                explanation = f"Showing a scatter plot of **{num_cols[1]}** versus **{num_cols[0]}**."
            elif len(num_cols) >= 1 and len(date_cols) >= 1:
                fig = px.line(df, x=date_cols[0], y=num_cols[0], title=f"Trend of {num_cols[0]} over {date_cols[0]}")
                explanation = f"Showing a line chart of **{num_cols[0]}** over **{date_cols[0]}**."
            elif len(cat_cols) >= 1 and len(num_cols) >= 1:
                fig = px.bar(df, x=cat_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {cat_cols[0]}")
                explanation = f"Showing a bar chart of **{num_cols[0]}** by **{cat_cols[0]}**."
            elif len(num_cols) >= 1:
                fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
                explanation = f"Showing the distribution of **{num_cols[0]}**."
            elif len(cat_cols) >= 1:
                fig = px.bar(df[cat_cols[0]].value_counts().reset_index(), x=cat_cols[0], y='count', 
                             title=f"Count of {cat_cols[0]}")
                explanation = f"Showing a bar chart of the count for each category in **{cat_cols[0]}**."
            else:
                return None, "Couldn't determine an appropriate chart type for your data. Try a more specific query."

        if fig:
            return fig, explanation
        else:
            return None, "Chart generation failed for an unknown reason. Please check column types and query."
    
    except Exception as e:
        return None, f"An error occurred while generating the chart: {e}. Please ensure your data is clean and column names are correct."

def generate_summary_chart(df, columns_to_plot):
    """Generate a summary chart for the dataset."""
    try:
        # Create a summary of the dataset
        summary = df.describe(include='all').transpose()
        
        # Create a heatmap of missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        # Create a figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Data Types", "Missing Values", "Numeric Distribution", "Categorical Distribution"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Data types pie chart
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index, values=dtype_counts.values, name="Data Types"),
            row=1, col=1
        )
        
        # Missing values bar chart
        fig.add_trace(
            go.Bar(x=missing.index, y=missing_pct, name="Missing %"),
            row=1, col=2
        )
        
        # Numeric distribution histogram
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            fig.add_trace(
                go.Histogram(x=df[numeric_cols[0]], name=numeric_cols[0]),
                row=2, col=1
            )
        
        # Categorical distribution bar chart
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            value_counts = df[cat_cols[0]].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=cat_cols[0]),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Dataset Summary")
        
        return fig, "Showing a comprehensive summary of the dataset including data types, missing values, and distributions."
    
    except Exception as e:
        return None, f"Error generating summary chart: {e}"

def generate_outlier_chart(df, columns_to_plot):
    """Generate a chart showing outliers in the data."""
    try:
        if not columns_to_plot:
            return None, "Please specify a column for outlier detection."
        
        col = columns_to_plot[0]
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            return None, f"Column '{col}' is not numeric. Outlier detection requires numeric data."
        
        # Calculate outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        # Create a box plot with outliers highlighted
        fig = go.Figure()
        
        # Add box plot
        fig.add_trace(go.Box(
            y=df[col],
            name="Data Distribution",
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        # Add a scatter plot for better visualization of outliers
        fig.add_trace(go.Scatter(
            x=['Outlier'] * len(outliers),
            y=outliers[col],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8)
        ))
        
        fig.update_layout(
            title=f"Outlier Detection for '{col}'",
            yaxis_title=col,
            height=500
        )
        
        explanation = f"Detected {len(outliers)} outliers in '{col}' using the IQR method. Outliers are values below {lower_bound:.2f} or above {upper_bound:.2f}."
        
        return fig, explanation
    
    except Exception as e:
        return None, f"Error generating outlier chart: {e}"

def generate_kpi_chart(df, columns_to_plot):
    """Generate a KPI dashboard."""
    try:
        # Calculate basic KPIs
        kpis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
        }
        
        # Create a figure with indicators
        fig = go.Figure()
        
        # Add KPI indicators
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=kpis["row_count"],
            title={"text": "Total Rows"},
            domain={'row': 0, 'column': 0}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=kpis["column_count"],
            title={"text": "Total Columns"},
            domain={'row': 0, 'column': 1}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=kpis["missing_values"],
            title={"text": "Missing Values"},
            domain={'row': 1, 'column': 0}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=kpis["duplicate_rows"],
            title={"text": "Duplicate Rows"},
            domain={'row': 1, 'column': 1}
        ))
        
        fig.update_layout(
            grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
            height=400,
            title_text="Key Performance Indicators"
        )
        
        return fig, "Showing key performance indicators for the dataset."
    
    except Exception as e:
        return None, f"Error generating KPI chart: {e}"