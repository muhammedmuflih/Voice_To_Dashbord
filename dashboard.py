# dashboard.py
# ✅ EXECUTIVE DASHBOARD v3
# Features: Enhanced UI • KPI Dashboard • Anomaly Detection • Dynamic Suggestions • Replace Values Feature • Data Quality Score
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import base64
from io import BytesIO
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE SETUP (MUST BE FIRST) ---
st.set_page_config(
    page_title="DataSense AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure utils directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# --- Import our robust utility functions ---
from utils.data_loader import load_data
from utils.Preprosseing import advanced_preprocess_data, get_data_profile, download_cleaned_data
from utils.nlp_processor import parse_query, generate_dynamic_suggestions
from utils.speech_input import get_voice_query
from utils.plot_generator import generate_chart
from utils.text_to_speech import speak_text
from utils.kpi_analyzer import calculate_kpis, detect_anomalies, generate_executive_insights, create_kpi_dashboard

# --- CONFIG ---
MAX_ROWS = 500000

# --- SESSION STATE ---
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    if 'last_chart_query' not in st.session_state:
        st.session_state.last_chart_query = ""
    if 'last_chart' not in st.session_state:
        st.session_state.last_chart = None
    if 'show_preprocessing_options' not in st.session_state:
        st.session_state.show_preprocessing_options = False
    if 'preprocessing_changes' not in st.session_state:
        st.session_state.preprocessing_changes = []
    if 'kpis' not in st.session_state:
        st.session_state.kpis = None
    if 'anomalies' not in st.session_state:
        st.session_state.anomalies = None
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'dynamic_suggestions' not in st.session_state:
        st.session_state.dynamic_suggestions = []
    if 'preprocessing_applied' not in st.session_state:
        st.session_state.preprocessing_applied = False
    if 'columns_to_drop' not in st.session_state:
        st.session_state.columns_to_drop = []
    if 'voice_query_result' not in st.session_state:
        st.session_state.voice_query_result = None
    if 'show_all_columns' not in st.session_state:
        st.session_state.show_all_columns = False

# Initialize session state
initialize_session_state()

# --- MAIN APP LAYOUT ---
def main():
    # Dark Mode Toggle
    st.sidebar.title("🤖 DataSense AI")
    st.sidebar.markdown("Intelligent Data Explorer")
    st.sidebar.divider()
    st.sidebar.header("Settings")
    if "is_dark_mode" not in st.session_state:
        st.session_state.is_dark_mode = True
    if st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.is_dark_mode):
        st.session_state.is_dark_mode = True
    else:
        st.session_state.is_dark_mode = False
    if st.session_state.is_dark_mode:
        st.markdown(
            """
            <style>
            .st-emotion-cache-1cypcdb {
                color: white;
                background-color: #0e1117;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
    st.sidebar.divider()
    
    # --- FILE UPLOAD SECTION ---
    st.sidebar.header("📁 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file (CSV, Excel, JSON, PDF, DOCX, etc.)",
        type=["csv", "xlsx", "json", "yaml", "yml", "pdf", "docx", "txt", "md"]
    )
    
    if uploaded_file and st.session_state.raw_df is None:
        st.session_state.file_name = uploaded_file.name
        with st.spinner("Loading data..."):
            st.session_state.raw_df = load_data(uploaded_file)
            if st.session_state.raw_df is not None:
                # We also make a clean copy to work with
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.show_preprocessing_options = True
                
                # Generate initial profile, KPIs, and insights
                st.session_state.profile = get_data_profile(st.session_state.df)
                st.session_state.kpis = calculate_kpis(st.session_state.df)
                st.session_state.anomalies = detect_anomalies(st.session_state.df)
                st.session_state.insights = generate_executive_insights(
                    st.session_state.df, 
                    st.session_state.kpis, 
                    st.session_state.anomalies
                )
                st.session_state.dynamic_suggestions = generate_dynamic_suggestions(st.session_state.df)
                
                # Initialize columns_to_drop as empty list
                st.session_state.columns_to_drop = []
                
                st.success("File loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load the file. Please check the file format.")
                st.session_state.raw_df = None
    
    if st.session_state.raw_df is not None:
        # --- PREPROCESSING OPTIONS ---
        st.sidebar.subheader("Preprocessing Options")
        with st.sidebar.expander("✨ Preprocess Data", expanded=False):
            st.write("Select the cleaning steps you want to apply.")
            
            # Enhanced column dropping section
            st.subheader("Drop Columns")
            
            # Calculate missing percentages for all columns
            missing_pct = (st.session_state.df.isnull().sum() / len(st.session_state.df)) * 100
            
            # Create a DataFrame with column info
            col_info_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Missing %': missing_pct.round(2),
                'Data Type': st.session_state.df.dtypes.astype(str),
                'Unique Values': st.session_state.df.nunique().values
            })
            
            # Sort by missing percentage (descending)
            col_info_df = col_info_df.sort_values('Missing %', ascending=False)
            
            # Option to show all columns
            show_all = st.checkbox("Show all columns", value=st.session_state.show_all_columns)
            st.session_state.show_all_columns = show_all
            
            # Filter columns based on selection
            if show_all:
                display_df = col_info_df
                st.write("All columns:")
            else:
                display_df = col_info_df[col_info_df['Missing %'] >= 50]
                if len(display_df) > 0:
                    st.write(f"Columns with 50% or more missing values ({len(display_df)} columns):")
                else:
                    st.write("No columns with 50% or more missing values found.")
            
            # Display column selection table
            with st.container():
                # Add a select all/none option
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select All"):
                        st.session_state.columns_to_drop = display_df['Column'].tolist()
                        st.rerun()
                with col2:
                    if st.button("Select None"):
                        st.session_state.columns_to_drop = []
                        st.rerun()
                
                # Display columns with checkboxes
                for _, row in display_df.iterrows():
                    col = row['Column']
                    missing_pct = row['Missing %']
                    dtype = row['Data Type']
                    unique_vals = row['Unique Values']
                    
                    # Determine if column is already selected
                    is_selected = col in st.session_state.columns_to_drop
                    
                    # Create a more informative checkbox label
                    label = f"{col} ({dtype}) - {missing_pct}% missing, {unique_vals} unique values"
                    
                    # Show checkbox with color coding for missing percentage
                    if missing_pct >= 50:
                        checkbox_color = "🔴"  # Red for high missing
                    elif missing_pct >= 20:
                        checkbox_color = "🟡"  # Yellow for medium missing
                    else:
                        checkbox_color = "🟢"  # Green for low missing
                        
                    selected = st.checkbox(
                        f"{checkbox_color} {label}", 
                        value=is_selected, 
                        key=f"drop_{col}"
                    )
                    
                    # Update the columns_to_drop list
                    if selected and col not in st.session_state.columns_to_drop:
                        st.session_state.columns_to_drop.append(col)
                    elif not selected and col in st.session_state.columns_to_drop:
                        st.session_state.columns_to_drop.remove(col)
            
            # Show summary of selected columns
            if st.session_state.columns_to_drop:
                st.write(f"**Selected {len(st.session_state.columns_to_drop)} columns to drop:**")
                st.write(", ".join(st.session_state.columns_to_drop))
            else:
                st.write("No columns selected for dropping.")
            
            st.write("---")
            
            # Other preprocessing options with explanations
            st.subheader("Data Cleaning")
            
            handle_missing = st.checkbox(
                "Handle Missing Values", 
                value=True,
                help="Fill missing values in numeric columns with the median and in categorical columns with the mode."
            )
            
            remove_duplicates = st.checkbox(
                "Remove Duplicate Rows", 
                value=True,
                help="Remove identical rows to avoid redundancy in analysis."
            )
            
            remove_constants = st.checkbox(
                "Remove Constant Columns", 
                value=True,
                help="Remove columns that have the same value in all rows as they don't provide useful information."
            )
            
            convert_dates = st.checkbox(
                "Convert Date Columns", 
                value=True,
                help="Automatically detect and convert columns with date information to proper datetime format."
            )
            
            # New Replace Values feature
            replace_values = st.checkbox(
                "Replace Values in Selected Column", 
                value=False,
                help="Replace specific patterns in a selected column with a replacement value."
            )
            
            if replace_values:
                st.write("**Replace Values Options:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    replace_column = st.selectbox(
                        "Select Column",
                        options=st.session_state.df.columns,
                        help="Choose the column where you want to replace values."
                    )
                
                with col2:
                    # Add example patterns in a dropdown
                    pattern_options = [
                        ".00:00:00",  # Default for removing time from dates
                        "\\$",  # Remove dollar signs
                        ",",  # Remove commas
                        "[^0-9.]",  # Keep only numbers and decimal points
                        "USD",  # Remove currency codes
                        " ",  # Remove spaces
                    ]
                    replace_pattern = st.selectbox(
                        "Pattern to Replace",
                        options=pattern_options,
                        index=0,  # Default to first option
                        help="Enter the pattern to search for (supports regex)."
                    )
                    # Allow custom pattern input
                    custom_pattern = st.text_input(
                        "Or Enter Custom Pattern",
                        value="",
                        help="Enter a custom regex pattern if the options don't match your needs."
                    )
                    if custom_pattern:
                        replace_pattern = custom_pattern
                
                with col3:
                    replace_with = st.text_input(
                        "Replace With",
                        value="",
                        help="Enter the replacement value. Leave empty to remove the pattern."
                    )
                
                # Add a preview section
                if replace_column and replace_pattern:
                    st.write("**Preview of Changes:**")
                    sample_data = st.session_state.df[replace_column].head(5).copy()
                    
                    # Convert to string for preview
                    sample_data_str = sample_data.astype(str)
                    
                    # Show original values
                    st.write("Original values:")
                    st.dataframe(pd.DataFrame({"Original": sample_data}))
                    
                    # Show what would be replaced
                    if sample_data_str.str.contains(replace_pattern, regex=True).any():
                        # Apply replacement to preview
                        preview_data = sample_data_str.str.replace(replace_pattern, replace_with, regex=True)
                        st.write("After replacement:")
                        st.dataframe(pd.DataFrame({"After": preview_data}))
                        
                        # Count how many would be changed
                        count_to_change = sample_data_str.str.contains(replace_pattern, regex=True).sum()
                        st.write(f"**{count_to_change} out of 5 sample values would be changed**")
                    else:
                        st.write("No matches found in the sample data")
            
            handle_outliers = st.checkbox(
                "Handle Outliers (numeric)", 
                value=False,
                help="Identify and cap extreme values in numeric columns using the IQR method to reduce their impact on analysis."
            )
            
            optimize_dtypes = st.checkbox(
                "Optimize Data Types", 
                value=True,
                help="Convert data types to more memory-efficient formats (e.g., object to category, float64 to float32)."
            )
            
            if st.button("🚀 Apply Cleaning"):
                with st.spinner("Preprocessing data... This might take a moment."):
                    options = {
                        "handle_missing": handle_missing,
                        "remove_duplicates": remove_duplicates,
                        "remove_constants": remove_constants,
                        "convert_dates": convert_dates,
                        "handle_outliers": handle_outliers,
                        "optimize_dtypes": optimize_dtypes,
                        "columns_to_drop": st.session_state.columns_to_drop,
                    }
                    
                    # Add replace values options if enabled
                    if replace_values:
                        options.update({
                            "replace_values": replace_values,
                            "replace_column": replace_column,
                            "replace_pattern": replace_pattern,
                            "replace_with": replace_with,
                        })
                    
                    st.session_state.df, changes = advanced_preprocess_data(st.session_state.raw_df.copy(), options)
                    st.session_state.show_preprocessing_options = False
                    st.session_state.preprocessing_applied = True
                    
                    # Update profile, KPIs, and insights after preprocessing
                    st.session_state.profile = get_data_profile(st.session_state.df)
                    st.session_state.kpis = calculate_kpis(st.session_state.df)
                    st.session_state.anomalies = detect_anomalies(st.session_state.df)
                    st.session_state.insights = generate_executive_insights(
                        st.session_state.df, 
                        st.session_state.kpis, 
                        st.session_state.anomalies
                    )
                    
                    st.success("Preprocessing completed!")
                    st.session_state.preprocessing_changes = changes
                    st.rerun()
        
        # --- MAIN DASHBOARD AREA ---
        st.title(f"Dashboard for `{st.session_state.file_name}`")
        st.markdown("---")
        df = st.session_state.df
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Summary", "💡 Smart Query", "📈 KPIs & Anomalies", "📖 Full Dataset"])
        
        with tab1:
            st.header("📊 Executive Summary")
            
            # Show preprocessing status
            if st.session_state.preprocessing_applied:
                st.success("✅ Data preprocessing has been applied to this dataset.")
                with st.expander("📝 Summary of Preprocessing Changes", expanded=True):
                    for change in st.session_state.preprocessing_changes:
                        st.markdown(f"- {change}")
                
                # Download button for preprocessed data
                st.markdown(
                    f'<a href="data:file/csv;base64,{base64.b64encode(download_cleaned_data(df).encode()).decode()}" download="preprocessed_data.csv">Download Preprocessed Data (CSV)</a>',
                    unsafe_allow_html=True
                )
                st.write("---")
            
            # Display key insights
            st.subheader("Key Insights")
            for insight in st.session_state.insights:
                st.markdown(f"• {insight}")
            
            st.write("---")
            
            # Display data profile summary
            st.subheader("Data Profile")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{st.session_state.profile['shape'][0]:,}")
            with col2:
                st.metric("Columns", st.session_state.profile['shape'][1])
            with col3:
                # Fix: Calculate total missing values from the dictionary
                total_missing = sum(st.session_state.profile['missing_values'].values())
                st.metric("Missing Values", f"{total_missing:,}")
            with col4:
                st.metric("Duplicates", st.session_state.profile['duplicate_rows'])
            
            st.write("---")
            
            # NEW: Data Quality Indicators
            st.subheader("🚨 Data Quality Indicators")
            
            # Calculate data quality metrics
            total_cells = st.session_state.profile['shape'][0] * st.session_state.profile['shape'][1]
            total_missing = sum(st.session_state.profile['missing_values'].values())
            missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
            duplicate_percentage = (st.session_state.profile['duplicate_rows'] / st.session_state.profile['shape'][0]) * 100 if st.session_state.profile['shape'][0] > 0 else 0
            
            # Check for outliers
            outlier_count = 0
            if st.session_state.anomalies:
                outlier_count = sum(anomaly_info["count"] for anomaly_info in st.session_state.anomalies.values())
            outlier_percentage = (outlier_count / st.session_state.profile['shape'][0]) * 100 if st.session_state.profile['shape'][0] > 0 else 0
            
            # Check for constant columns
            constant_columns = 0
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_columns += 1
            
            # Check for high cardinality columns
            high_cardinality_columns = 0
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if df[col].nunique() > len(df) * 0.8:
                    high_cardinality_columns += 1
            
            # Create indicator cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Missing Values Indicator
                if missing_percentage > 5:
                    st.error("❌ Missing Values")
                    st.metric("% Missing", f"{missing_percentage:.1f}%")
                elif missing_percentage > 0:
                    st.warning("⚠️ Missing Values")
                    st.metric("% Missing", f"{missing_percentage:.1f}%")
                else:
                    st.success("✅ No Missing Values")
                    st.metric("% Missing", "0%")
            
            with col2:
                # Duplicates Indicator
                if duplicate_percentage > 5:
                    st.error("❌ Duplicates")
                    st.metric("% Duplicates", f"{duplicate_percentage:.1f}%")
                elif duplicate_percentage > 0:
                    st.warning("⚠️ Duplicates")
                    st.metric("% Duplicates", f"{duplicate_percentage:.1f}%")
                else:
                    st.success("✅ No Duplicates")
                    st.metric("% Duplicates", "0%")
            
            with col3:
                # Outliers Indicator
                if outlier_percentage > 5:
                    st.error("❌ Outliers")
                    st.metric("% Outliers", f"{outlier_percentage:.1f}%")
                elif outlier_percentage > 0:
                    st.warning("⚠️ Outliers")
                    st.metric("% Outliers", f"{outlier_percentage:.1f}%")
                else:
                    st.success("✅ No Outliers")
                    st.metric("% Outliers", "0%")
            
            with col4:
                # Constant Columns Indicator
                if constant_columns > 0:
                    st.error("❌ Constant Columns")
                    st.metric("Count", f"{constant_columns}")
                else:
                    st.success("✅ No Constant Columns")
                    st.metric("Count", "0")
            
            with col5:
                # High Cardinality Indicator
                if high_cardinality_columns > 0:
                    st.warning("⚠️ High Cardinality")
                    st.metric("Count", f"{high_cardinality_columns}")
                else:
                    st.success("✅ Normal Cardinality")
                    st.metric("Count", "0")
            
            st.write("---")
            
            # NEW: Overall Data Quality Score
            st.subheader("📊 Overall Data Quality Score")
            
            # Calculate overall data quality score
            quality_score = 100  # Start with perfect score
            
            # Deduct points for each issue
            if missing_percentage > 0:
                quality_score -= min(20, missing_percentage * 2)  # Max 20 points deduction
            
            if duplicate_percentage > 0:
                quality_score -= min(15, duplicate_percentage * 3)  # Max 15 points deduction
            
            if outlier_percentage > 0:
                quality_score -= min(15, outlier_percentage * 3)  # Max 15 points deduction
            
            if constant_columns > 0:
                quality_score -= min(10, constant_columns * 5)  # Max 10 points deduction
            
            if high_cardinality_columns > 0:
                quality_score -= min(10, high_cardinality_columns * 5)  # Max 10 points deduction
            
            # Additional deductions for extreme cases
            if missing_percentage > 30:
                quality_score -= 20  # Severe missing data penalty
            if duplicate_percentage > 20:
                quality_score -= 15  # Severe duplication penalty
            
            # Ensure score doesn't go below 0
            quality_score = max(0, quality_score)
            
            # Create columns for the score display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a gauge chart for the quality score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=quality_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Data Quality Score"},
                    delta={'reference': 100},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},
                            {'range': [50, 80], 'color': "gold"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display score breakdown
                st.subheader("Score Breakdown")
                st.write(f"**Base Score:** 100")
                
                if missing_percentage > 0:
                    deduction = min(20, missing_percentage * 2)
                    if missing_percentage > 30:
                        deduction += 20
                    st.write(f"- Missing Values: -{deduction:.1f}")
                
                if duplicate_percentage > 0:
                    deduction = min(15, duplicate_percentage * 3)
                    if duplicate_percentage > 20:
                        deduction += 15
                    st.write(f"- Duplicates: -{deduction:.1f}")
                
                if outlier_percentage > 0:
                    deduction = min(15, outlier_percentage * 3)
                    st.write(f"- Outliers: -{deduction:.1f}")
                
                if constant_columns > 0:
                    deduction = min(10, constant_columns * 5)
                    st.write(f"- Constant Columns: -{deduction:.1f}")
                
                if high_cardinality_columns > 0:
                    deduction = min(10, high_cardinality_columns * 5)
                    st.write(f"- High Cardinality: -{deduction:.1f}")
                
                st.write("---")
                st.write(f"**Final Score:** {quality_score:.1f}")
            
            # Add quality interpretation
            if quality_score >= 90:
                st.success("🎉 **Excellent data quality!** Your dataset is clean and ready for analysis.")
                st.write("✅ No significant issues detected")
                st.write("✅ Data can be used for reliable analysis")
            elif quality_score >= 80:
                st.info("👍 **Good data quality.** Minor issues detected but shouldn't affect analysis significantly.")
                st.write("✅ Dataset is mostly clean")
                st.write("⚠️ Consider addressing minor issues for optimal results")
            elif quality_score >= 50:
                st.warning("⚠️ **Moderate data quality.** Consider addressing the highlighted issues before analysis.")
                st.write("⚠️ Some data quality issues present")
                st.write("🔧 Preprocessing recommended before analysis")
                st.write("📊 Results may be affected by data quality issues")
            else:
                st.error("🚨 **Poor data quality.** Significant issues detected that may affect your analysis results.")
                st.write("❌ Multiple data quality issues detected")
                st.write("🔧 Extensive preprocessing required")
                st.write("⚠️ Analysis results may be unreliable")
                st.write("💡 Consider data cleaning or collecting better data")
            
            st.write("---")
            
            # Display KPI dashboard
            st.subheader("KPI Dashboard")
            kpi_figures = create_kpi_dashboard(st.session_state.kpis, st.session_state.anomalies)
            
            # Display summary KPIs
            if "summary" in kpi_figures:
                st.plotly_chart(kpi_figures["summary"], use_container_width=True, key="summary_kpi_chart")
            
            # Display anomaly chart if anomalies exist
            if "anomalies" in kpi_figures:
                st.plotly_chart(kpi_figures["anomalies"], use_container_width=True, key="anomaly_chart")
            
            # Display column type distribution with detailed breakdown
            st.subheader("Column Type Distribution")
            col_types = {
                "Numeric": st.session_state.kpis.get('numeric_columns', 0),
                "Categorical": st.session_state.kpis.get('categorical_columns', 0),
                "Date": st.session_state.kpis.get('date_columns', 0),
                "Other": st.session_state.kpis['column_count'] - 
                        st.session_state.kpis.get('numeric_columns', 0) - 
                        st.session_state.kpis.get('categorical_columns', 0) - 
                        st.session_state.kpis.get('date_columns', 0)
            }
            
            # Create a more informative pie chart
            if "column_types" in kpi_figures:
                st.plotly_chart(kpi_figures["column_types"], use_container_width=True, key="column_types_chart")
            
            # Add detailed breakdown
            st.write("**Column Breakdown:**")
            for col_type, count in col_types.items():
                if count > 0:
                    st.write(f"- {col_type}: {count} columns ({count/st.session_state.kpis['column_count']:.1%})")
            
            # Add Date Columns Summary section if date columns exist
            if st.session_state.kpis.get('date_columns', 0) > 0:
                st.write("---")
                st.subheader("Date Columns Summary")
                date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                
                for col in date_cols:
                    col_info = st.session_state.profile['columns'][col]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min Date", col_info['min_date'].strftime('%Y-%m-%d') if not pd.isna(col_info['min_date']) else "N/A")
                    with col2:
                        st.metric("Max Date", col_info['max_date'].strftime('%Y-%m-%d') if not pd.isna(col_info['max_date']) else "N/A")
                    with col3:
                        st.metric("Date Range (Days)", col_info['date_range_days'])
        
        with tab2:
            st.header("💡 Ask a Question")
            st.write("Use natural language to get insights from your data. Try a voice or text query!")
            
            # Display simple, understandable suggestions
            st.subheader("Try These Simple Queries:")
            
            # Get column types for better suggestions
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Create simple suggestions based on available columns
            suggestions = []
            
            if numeric_cols:
                if len(numeric_cols) >= 2:
                    suggestions.append(f"Show relationship between {numeric_cols[0]} and {numeric_cols[1]}")
                suggestions.append(f"Show distribution of {numeric_cols[0]}")
                
            if categorical_cols:
                suggestions.append(f"Show count of each {categorical_cols[0]}")
                if numeric_cols:
                    suggestions.append(f"Show {numeric_cols[0]} by {categorical_cols[0]}")
                    
            if date_cols:
                if numeric_cols:
                    suggestions.append(f"Show trend of {numeric_cols[0]} over time")
                    
            # Display suggestions in a clean layout
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions[:8]):  # Limit to 8 suggestions
                with cols[i % 2]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        # Use a callback to handle the button click
                        handle_suggestion_click(suggestion)
            
            st.write("---")
            
            # Custom query input
            st.subheader("Ask Your Own Question:")
            
            # Initialize session state for voice query result if not exists
            if 'voice_query_result' not in st.session_state:
                st.session_state.voice_query_result = None
            
            # Check if we have a voice query result
            query_value = st.session_state.voice_query_result if st.session_state.voice_query_result else ""
            
            # Create the text input widget
            query = st.text_input(
                "Your query:",
                placeholder="e.g., 'Show me a bar chart of sales by product category' or 'What is the average age?'",
                key="text_query",
                value=query_value
            )
            
            # Voice input button
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🎤 Voice Input"):
                    with st.spinner("Listening..."):
                        voice_query_text = get_voice_query()
                        if voice_query_text:
                            # Set the voice query result in session state
                            st.session_state.voice_query_result = voice_query_text
                            speak_text("I heard: " + voice_query_text)
                            # Rerun to update the text input with the voice result
                            st.rerun()
                        else:
                            st.info("Could not recognize speech. Please try again.")
            
            # Process the query if it's not empty
            if query:
                st.info(f"Processing query: **'{query}'**")
                
                # 1. Parse the query to find columns and chart type
                query_details = parse_query(query, df)
                
                # 2. Generate the chart
                chart, explanation = generate_chart(df, query_details)
                
                if chart:
                    st.session_state.last_chart = chart
                    st.session_state.last_chart_query = query
                    st.plotly_chart(chart, use_container_width=True, key="generated_chart")
                    speak_text(explanation)
                    st.markdown(f"**Explanation:** {explanation}")
                else:
                    st.error(explanation)
                
                # Clear the voice query result after processing
                if st.session_state.voice_query_result:
                    st.session_state.voice_query_result = None
            
            # Display last successful chart
            if st.session_state.get('last_chart_query') and st.session_state.get('last_chart'):
                st.write("---")
                st.subheader("📊 Last Generated Chart")
                st.write(f"Query: *{st.session_state.last_chart_query}*")
                # Fixed: Added unique key to prevent duplicate ID error
                st.plotly_chart(st.session_state.last_chart, use_container_width=True, key="last_chart_display")
                
                if st.button("🗑️ Clear Last Chart"):
                    st.session_state.last_chart_query = ""
                    st.session_state.last_chart = None
                    st.rerun()
                    
        with tab3:
            st.header("📈 KPIs & Anomalies")
            
            # Display detailed KPIs
            st.subheader("Key Performance Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Rows", f"{st.session_state.kpis['row_count']:,}")
                st.metric("Total Columns", st.session_state.kpis['column_count'])
                st.metric("Numeric Columns", st.session_state.kpis.get('numeric_columns', 0))
                st.metric("Categorical Columns", st.session_state.kpis.get('categorical_columns', 0))
            
            with col2:
                st.metric("Date Columns", st.session_state.kpis.get('date_columns', 0))
                # Fix: Calculate total missing values from the KPIs
                st.metric("Missing Values", f"{st.session_state.kpis['missing_values']:,}")
                st.metric("Duplicate Rows", f"{st.session_state.kpis['duplicate_rows']:,}")
                st.metric("Memory Usage (MB)", f"{st.session_state.kpis['memory_usage_mb']:.2f}")
            
            st.write("---")
            
            # Display anomalies
            st.subheader("Anomaly Detection")
            if st.session_state.anomalies:
                for col, anomaly_info in st.session_state.anomalies.items():
                    with st.expander(f"Anomalies in '{col}' ({anomaly_info['count']} detected, {anomaly_info['percentage']:.2f}%)"):
                        st.write(f"Method: {anomaly_info['method']}")
                        
                        if anomaly_info['method'] == 'IQR':
                            st.write(f"Lower Bound: {anomaly_info['lower_bound']:.2f}")
                            st.write(f"Upper Bound: {anomaly_info['upper_bound']:.2f}")
                        else:
                            st.write(f"Threshold: {anomaly_info['threshold']}")
                        
                        st.write("Sample Outliers:")
                        st.dataframe(pd.DataFrame(anomaly_info['outliers'], columns=['Value']))
            else:
                st.info("No anomalies detected in the dataset.")
            
            st.write("---")
            
            # Display detailed column statistics
            st.subheader("Column Statistics")
            selected_column = st.selectbox(
                "Select a Column for Detailed Analysis",
                options=df.columns
            )
            
            if selected_column:
                col_info = st.session_state.profile['columns'][selected_column]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Data Type", col_info['dtype'])
                with col2:
                    st.metric("Unique Values", col_info['unique_count'])
                with col3:
                    st.metric("Missing Count", col_info['missing_count'])
                with col4:
                    st.metric("Missing %", f"{col_info['missing_percent']:.2f}%")
                
                # Type-specific statistics
                if pd.api.types.is_numeric_dtype(df[selected_column]):
                    st.write("---")
                    st.subheader("Numeric Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Min", f"{col_info['min']:.2f}")
                        st.metric("Max", f"{col_info['max']:.2f}")
                        st.metric("Mean", f"{col_info['mean']:.2f}")
                    
                    with col2:
                        st.metric("Median", f"{col_info['median']:.2f}")
                        st.metric("Std Dev", f"{col_info['std']:.2f}")
                    
                    # Distribution chart
                    fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                    st.plotly_chart(fig, use_container_width=True, key=f"distribution_{selected_column}")
                    
                elif pd.api.types.is_datetime64_any_dtype(df[selected_column]):
                    st.write("---")
                    st.subheader("Date Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Min Date", col_info['min_date'].strftime('%Y-%m-%d') if not pd.isna(col_info['min_date']) else "N/A")
                        st.metric("Max Date", col_info['max_date'].strftime('%Y-%m-%d') if not pd.isna(col_info['max_date']) else "N/A")
                    
                    with col2:
                        st.metric("Date Range (Days)", col_info['date_range_days'])
                        st.metric("Most Common Year", col_info['most_common_year'])
                    
                    # Time series chart
                    fig = px.line(
                        df.groupby(selected_column).size().reset_index(name='count'),
                        x=selected_column,
                        y='count',
                        title=f"Records Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{selected_column}")
                    
                elif pd.api.types.is_object_dtype(df[selected_column]) or pd.api.types.is_categorical_dtype(df[selected_column]):
                    st.write("---")
                    st.subheader("Categorical Statistics")
                    st.metric("Cardinality", col_info['cardinality'])
                    
                    # Value counts chart
                    value_counts = df[selected_column].value_counts().head(20)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top 20 Values in {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"valuecounts_{selected_column}")
                    
                    with st.expander("All Values"):
                        st.dataframe(df[selected_column].value_counts())
                    
        with tab4:
            st.header("Full Dataset")
            st.write("View the complete cleaned dataset below. You can sort columns and filter rows directly in the table.")
            st.dataframe(df)
        
        st.sidebar.divider()
        if st.sidebar.button("🗑️ Clear Data"):
            st.session_state.raw_df = None
            st.session_state.df = None
            st.session_state.file_name = None
            st.session_state.last_chart_query = ""
            st.session_state.last_chart = None
            st.session_state.show_preprocessing_options = False
            st.session_state.preprocessing_changes = []
            st.session_state.preprocessing_applied = False
            st.session_state.kpis = None
            st.session_state.anomalies = None
            st.session_state.insights = None
            st.session_state.profile = None
            st.session_state.dynamic_suggestions = []
            st.session_state.columns_to_drop = []
            st.session_state.voice_query_result = None
            st.session_state.show_all_columns = False
            st.info("Dataset cleared. Please upload a new file.")
            st.rerun()
    
    else:
        st.title("Welcome to DataSense AI")
        st.markdown("Your intelligent data exploration assistant.")
        st.info("👈 Please upload a file from the sidebar to begin.")

# Callback function for handling suggestion button clicks
def handle_suggestion_click(suggestion):
    """Handle the click event on a suggestion button."""
    st.session_state.voice_query_result = suggestion
    st.rerun()

# Run the main function
if __name__ == "__main__":
    main()