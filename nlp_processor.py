# utils/nlp_processor.py

import re
import pandas as pd
import numpy as np
import difflib
from collections import defaultdict

def _normalize_column_name(col_name: str):
    """Normalize column names for easier matching."""
    return re.sub(r'[^a-zA-Z0-9]+', '', col_name).lower()

def _find_all_matches(query_text, columns):
    """Finds all columns mentioned in the query using a more robust approach."""
    found_cols = []
    
    # Create a mapping of normalized column names to original column names
    normalized_to_original = {_normalize_column_name(col): col for col in columns}
    
    # Split query into potential phrases (single words, two-word combinations)
    words = re.findall(r'\b\w+\b', query_text.lower())
    phrases = []
    for i in range(len(words)):
        phrases.append(words[i])  # Single words
        if i + 1 < len(words):
            phrases.append(f"{words[i]} {words[i+1]}")  # Two-word phrases
    
    # Sort columns by length to prioritize longer, more specific matches
    sorted_columns = sorted(columns, key=len, reverse=True)
    
    for col in sorted_columns:
        norm_col = _normalize_column_name(col)
        
        # Check for direct inclusion or very close match in the query phrases
        for phrase in phrases:
            if phrase == norm_col or difflib.SequenceMatcher(None, phrase, norm_col).ratio() > 0.8:
                if col not in found_cols:
                    found_cols.append(col)
                    
    return found_cols

def generate_dynamic_suggestions(df):
    """
    Generate dynamic query suggestions based on the DataFrame structure.
    Creates simple, understandable suggestions.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
        list: A list of suggested queries.
    """
    suggestions = []
    
    # Get column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Generate simple, clear suggestions based on available data
    if numeric_cols:
        # Distribution suggestions
        if len(numeric_cols) > 0:
            suggestions.append(f"Show distribution of {numeric_cols[0]}")
        
        # Relationship suggestions
        if len(numeric_cols) >= 2:
            suggestions.append(f"Show relationship between {numeric_cols[0]} and {numeric_cols[1]}")
        
        # Summary statistics
        suggestions.append(f"Show summary statistics for {numeric_cols[0]}")
        
        # Correlation with categorical if available
        if categorical_cols:
            suggestions.append(f"Compare {numeric_cols[0]} across {categorical_cols[0]}")
    
    if categorical_cols:
        # Count suggestions
        if len(categorical_cols) > 0:
            suggestions.append(f"Show count of each {categorical_cols[0]}")
        
        # Top values
        suggestions.append(f"Show top values in {categorical_cols[0]}")
        
        # Comparison with numeric if available
        if numeric_cols:
            suggestions.append(f"Show {numeric_cols[0]} by {categorical_cols[0]}")
    
    if date_cols:
        # Time series suggestions
        if len(date_cols) > 0 and numeric_cols:
            suggestions.append(f"Show trend of {numeric_cols[0]} over time")
        
        # Time-based aggregation
        if categorical_cols:
            suggestions.append(f"Show {categorical_cols[0]} trends over time")
    
    # General suggestions
    if len(df.columns) > 0:
        suggestions.append(f"Show overview of the dataset")
        suggestions.append(f"Show data quality report")
    
    # Limit to 10 suggestions to avoid overwhelming the user
    return suggestions[:10]

def parse_query(query_text: str, df: pd.DataFrame):
    """
    Parses a natural language query to identify columns, chart types, and filter criteria.
    
    Args:
        query_text (str): The user's query.
        df (pd.DataFrame): The DataFrame to analyze.
        
    Returns:
        dict: A dictionary with the parsed data.
    """
    parsed_data = {
        "columns": [],
        "chart_type": "auto",
        "filter": None,
        "aggregation": None,
        "analysis_type": None  # New field to identify the type of analysis
    }
    query_text_lower = query_text.lower()
    columns = df.columns.tolist()
    
    date_cols = [c for c in columns if 'date' in c.lower() or 'time' in c.lower() or 'day' in c.lower()]
    
    # --- 1. Identify analysis type keywords first ---
    if any(keyword in query_text_lower for keyword in ["summary", "overview", "describe", "profile"]):
        parsed_data["analysis_type"] = "summary"
    elif any(keyword in query_text_lower for keyword in ["trend", "over time", "line", "monthly", "yearly"]):
        parsed_data["analysis_type"] = "trend"
        parsed_data["chart_type"] = "line"
    elif any(keyword in query_text_lower for keyword in ["relationship", "correlation", "scatter", "scatter plot"]):
        parsed_data["analysis_type"] = "relationship"
        parsed_data["chart_type"] = "scatter"
    elif any(keyword in query_text_lower for keyword in ["pie", "proportion", "share", "percentage"]):
        parsed_data["analysis_type"] = "proportion"
        parsed_data["chart_type"] = "pie"
    elif any(keyword in query_text_lower for keyword in ["distribution", "histogram", "hist", "frequency"]):
        parsed_data["analysis_type"] = "distribution"
        parsed_data["chart_type"] = "histogram"
    elif any(keyword in query_text_lower for keyword in ["bar", "total", "by", "compare", "comparison"]):
        parsed_data["analysis_type"] = "comparison"
        parsed_data["chart_type"] = "bar"
    elif any(keyword in query_text_lower for keyword in ["outlier", "anomaly", "unusual"]):
        parsed_data["analysis_type"] = "outlier"
    elif any(keyword in query_text_lower for keyword in ["kpi", "metric", "performance"]):
        parsed_data["analysis_type"] = "kpi"
    
    # --- 2. Identify all mentioned columns ---
    parsed_data["columns"] = _find_all_matches(query_text, columns)
    
    # Ensure the primary date column is included for line charts if not explicitly mentioned but needed
    if parsed_data["chart_type"] == "line" and date_cols and date_cols[0] not in parsed_data["columns"]:
        # Only add if there's at least one numeric column already found
        if any(c in df.select_dtypes(include=np.number).columns for c in parsed_data["columns"]):
            parsed_data["columns"].insert(0, date_cols[0]) # Insert date as the first dimension
            
    # Fallback to general if no specific chart type or columns are clear
    if not parsed_data["columns"] and len(columns) > 1:
        numeric_cols_df = df.select_dtypes(include=np.number).columns
        if not numeric_cols_df.empty:
            parsed_data["columns"].append(numeric_cols_df[0])
            if len(numeric_cols_df) > 1:
                parsed_data["columns"].append(numeric_cols_df[1])
            elif not df.select_dtypes(include='object').empty:
                 parsed_data["columns"].append(df.select_dtypes(include='object').columns[0])
    
    # --- 3. Identify Filter and Aggregation Keywords ---
    filter_keywords = {
        "equals": ["equal to", "is", "of", "in"],
        "greater_than": ["greater than", "more than", ">", "above"],
        "less_than": ["less than", "fewer than", "<", "below"],
        "contains": ["contains", "including"],
        "not_equal": ["not equal to", "not"],
        "top": ["top", "most", "highest"],
        "bottom": ["bottom", "least", "lowest"],
    }
    
    aggregation_keywords = {
        "sum": ["total", "sum"],
        "average": ["average", "avg", "mean"],
        "count": ["count", "number of"],
        "max": ["max", "highest", "largest"],
        "min": ["min", "lowest", "smallest"],
    }
    
    # Simple keyword-based filtering
    for col in columns:
        if col.lower() in query_text_lower:
            # Check for filter values
            for op, ops_list in filter_keywords.items():
                for op_word in ops_list:
                    regex = re.compile(f'{op_word}\\s+([a-zA-Z0-9\\s_]+)', re.IGNORECASE)
                    match = regex.search(query_text_lower)
                    if match:
                        value = match.group(1).strip()
                        parsed_data["filter"] = {"column": col, "operator": op, "value": value}
                        break
                if parsed_data["filter"]:
                    break
    
    # Check for aggregation functions
    for agg_type, agg_words in aggregation_keywords.items():
        if any(word in query_text_lower for word in agg_words):
            parsed_data["aggregation"] = agg_type
            break
            
    return parsed_data