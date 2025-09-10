# utils/Preprosseing.py
import pandas as pd
import numpy as np
import streamlit as st
import io
import re
import base64

def explain_dataset(df):
    """
    Analyzes and displays key information about the uploaded DataFrame.
    """
    st.subheader("📊 Dataset Overview")
    
    # 1. Dataset Shape
    st.write("---")
    st.markdown("**Shape of the Dataset:**")
    st.info(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    # 2. Data Types & Non-Null Counts
    st.write("---")
    st.markdown("**Data Types and Non-Null Counts:**")
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True)
    st.code(buffer.getvalue(), language='text')
    
    # 3. Missing Values Summary
    st.write("---")
    st.markdown("**Missing Values Summary:**")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if not null_counts.empty:
        null_df = pd.DataFrame(
            {'Missing Values': null_counts, 
             'Percentage': (null_counts / df.shape[0] * 100).round(2)}
        )
        st.dataframe(null_df)
    else:
        st.success("🎉 No missing values found in the dataset!")

    # 4. Duplicate Rows
    st.write("---")
    st.markdown("**Duplicate Rows:**")
    duplicated_rows = df.duplicated().sum()
    if duplicated_rows > 0:
        st.warning(f"There are **{duplicated_rows}** duplicate rows in the dataset.")
    else:
        st.success("🎉 No duplicate rows found in the dataset.")

    # 5. Categorical vs. Numerical Columns
    st.write("---")
    st.markdown("**Column Types:**")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Numerical Columns", len(numeric_cols))
        if numeric_cols:
            st.write("📊", ", ".join(numeric_cols[:3]))
            if len(numeric_cols) > 3:
                st.write(f"... and {len(numeric_cols) - 3} more")
    
    with col2:
        st.metric("Categorical Columns", len(categorical_cols))
        if categorical_cols:
            st.write("📝", ", ".join(categorical_cols[:3]))
            if len(categorical_cols) > 3:
                st.write(f"... and {len(categorical_cols) - 3} more")
    
    with col3:
        st.metric("Date/Time Columns", len(datetime_cols))
        if datetime_cols:
            st.write("📅", ", ".join(datetime_cols[:3]))
            if len(datetime_cols) > 3:
                st.write(f"... and {len(datetime_cols) - 3} more")

    # 6. Detailed Statistics
    if numeric_cols:
        with st.expander("📈 Numerical Columns Analysis", expanded=False):
            st.dataframe(df[numeric_cols].describe().T)
            
            # Check for potential outliers
            st.subheader("Potential Outliers Detection")
            outlier_info = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    outlier_info.append({
                        'Column': col,
                        'Outliers': len(outliers),
                        'Percentage': f"{(len(outliers)/len(df)*100):.2f}%"
                    })
            
            if outlier_info:
                outlier_df = pd.DataFrame(outlier_info)
                st.dataframe(outlier_df)
            else:
                st.success("No significant outliers detected using IQR method.")

    if categorical_cols:
        with st.expander("📝 Categorical Columns Analysis", expanded=False):
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                st.subheader(f"Distribution of '{col}'")
                value_counts = df[col].value_counts().head(10)
                st.bar_chart(value_counts)
                
                # Show unique values count
                unique_count = df[col].nunique()
                st.write(f"Unique values: {unique_count}")
                
                if unique_count <= 20:
                    st.write("All unique values:", df[col].unique().tolist())
                else:
                    st.write("Top 10 most frequent values:", value_counts.index.tolist())

    # 7. Data Quality Issues
    st.write("---")
    st.subheader("⚠️ Data Quality Issues Found:")
    
    issues = []
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        issues.append("❌ Missing values detected")
    
    # Check for duplicates
    if df.duplicated().sum() > 0:
        issues.append("❌ Duplicate rows detected")
    
    # Check for columns with high cardinality
    for col in categorical_cols:
        if df[col].nunique() > len(df) * 0.8:
            issues.append(f"❌ High cardinality in '{col}' (might be ID column)")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues.append(f"❌ Constant columns detected: {constant_cols}")
    
    if issues:
        for issue in issues:
            st.warning(issue)
        st.info("Consider using the preprocessing option to automatically fix some of these issues.")
    else:
        st.success("✅ No major data quality issues detected!")


def preprocess_data(df):
    """
    Performs comprehensive data cleaning steps:
    1. Drops duplicate rows
    2. Handles missing values intelligently
    3. Removes constant columns
    4. Detects and converts date columns
    5. Handles high cardinality categorical columns
    
    Returns the cleaned DataFrame with a summary of changes.
    """
    if df is None or df.empty:
        return df, []

    original_shape = df.shape
    changes_made = []
    
    st.subheader("🔄 Data Preprocessing Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Drop duplicate rows
    status_text.text("Step 1/6: Removing duplicate rows...")
    progress_bar.progress(1/6)
    
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    rows_dropped = initial_rows - df.shape[0]
    if rows_dropped > 0:
        changes_made.append(f"✅ Removed {rows_dropped} duplicate rows")
    
    # Step 2: Remove constant columns
    status_text.text("Step 2/6: Removing constant columns...")
    progress_bar.progress(2/6)
    
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        changes_made.append(f"✅ Removed {len(constant_cols)} constant columns: {constant_cols}")
    
    # Step 3: Smart date detection and conversion
    status_text.text("Step 3/6: Converting date columns...")
    progress_bar.progress(3/6)
    
    potential_date_cols = [col for col in df.columns if 
                          any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp'])]
    
    # Also check for columns that might contain dates based on their content
    for col in df.columns:
        if col not in potential_date_cols and df[col].dtype == 'object':
            # Check if column values match date patterns
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                # Check for common date patterns
                date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}'
                has_date_pattern = sample_values.astype(str).str.match(date_pattern).any()
                if has_date_pattern:
                    potential_date_cols.append(col)
    
    for col in potential_date_cols:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                changes_made.append(f"✅ Converted '{col}' to datetime")
            except:
                pass
    
    # Step 4: Handle missing values intelligently
    status_text.text("Step 4/6: Handling missing values...")
    progress_bar.progress(4/6)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / len(df)) * 100
            
            if missing_percentage > 50:
                # Drop columns with too many missing values
                df = df.drop(columns=[col])
                changes_made.append(f"⚠️ Dropped '{col}' (>{missing_percentage:.1f}% missing)")
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Fill numerical missing values with median (more robust than mean)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Fill datetime missing values with mode or forward fill
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with mode")
                else:
                    df[col] = df[col].fillna(method='ffill')
                    changes_made.append(f"✅ Forward-filled {missing_count} missing values in '{col}'")
            else:
                # Fill categorical missing values with mode or 'Unknown'
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with mode ('{mode_val}')")
                else:
                    df[col] = df[col].fillna('Unknown')
                    changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with 'Unknown'")
    
    # Step 5: Handle high cardinality categorical columns
    status_text.text("Step 5/6: Optimizing high cardinality columns...")
    progress_bar.progress(5/6)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.8:  # If more than 80% unique values
            # This might be an ID column or high cardinality category
            # Keep only top categories, group others as 'Other'
            top_categories = df[col].value_counts().head(10).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
            changes_made.append(f"✅ Grouped rare categories in '{col}' into 'Other'")
    
    # Step 6: Final optimization
    status_text.text("Step 6/6: Final optimizations...")
    progress_bar.progress(1.0)
    
    # Optimize data types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category if it has few unique values
            if df[col].nunique() < len(df) * 0.5:
                df[col] = df[col].astype('category')
                changes_made.append(f"✅ Optimized '{col}' to category type")
    
    status_text.text("Preprocessing completed!")
    
    # Display summary
    st.success("🎉 Preprocessing completed successfully!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Shape", f"{original_shape[0]} × {original_shape[1]}")
    with col2:
        st.metric("Final Shape", f"{df.shape[0]} × {df.shape[1]}")
    
    if changes_made:
        st.subheader("📋 Changes Made:")
        for change in changes_made:
            st.write(change)
    else:
        st.info("No preprocessing changes were needed - your data is already clean!")
    
    return df, changes_made


def advanced_preprocess_data(df, options):
    """
    Advanced preprocessing with customizable options.
    
    Args:
        df: DataFrame to preprocess
        options: Dictionary of preprocessing options
        
    Returns:
        Tuple of (cleaned DataFrame, list of changes made)
    """
    if df is None or df.empty:
        return df, []
    
    changes_made = []
    
    # Step 1: Drop selected columns
    if "columns_to_drop" in options and options["columns_to_drop"]:
        columns_to_drop = [col for col in options["columns_to_drop"] if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            changes_made.append(f"✅ Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
    
    # Step 2: Remove duplicate rows
    if options.get("remove_duplicates", True):
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        rows_dropped = initial_rows - df.shape[0]
        if rows_dropped > 0:
            changes_made.append(f"✅ Removed {rows_dropped} duplicate rows")
    
    # Step 3: Remove constant columns
    if options.get("remove_constants", True):
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            changes_made.append(f"✅ Removed {len(constant_cols)} constant columns: {constant_cols}")
    
    # Step 4: Convert date columns
    if options.get("convert_dates", True):
        potential_date_cols = [col for col in df.columns if 
                              any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp'])]
        
        # Also check for columns that might contain dates based on their content
        for col in df.columns:
            if col not in potential_date_cols and df[col].dtype == 'object':
                # Check if column values match date patterns
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Check for common date patterns
                    date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}'
                    has_date_pattern = sample_values.astype(str).str.match(date_pattern).any()
                    if has_date_pattern:
                        potential_date_cols.append(col)
        
        for col in potential_date_cols:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                    changes_made.append(f"✅ Converted '{col}' to datetime")
                except:
                    pass
    
    # Step 5: Handle missing values
    if options.get("handle_missing", True):
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                
                if missing_percentage > 50:
                    # Drop columns with too many missing values
                    df = df.drop(columns=[col])
                    changes_made.append(f"⚠️ Dropped '{col}' (>{missing_percentage:.1f}% missing)")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # Fill numerical missing values with median (more robust than mean)
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Fill datetime missing values with mode or forward fill
                    if not df[col].mode().empty:
                        mode_val = df[col].mode()[0]
                        df[col] = df[col].fillna(mode_val)
                        changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with mode")
                    else:
                        df[col] = df[col].fillna(method='ffill')
                        changes_made.append(f"✅ Forward-filled {missing_count} missing values in '{col}'")
                else:
                    # Fill categorical missing values with mode or 'Unknown'
                    if not df[col].mode().empty:
                        mode_val = df[col].mode()[0]
                        df[col] = df[col].fillna(mode_val)
                        changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with mode ('{mode_val}')")
                    else:
                        df[col] = df[col].fillna('Unknown')
                        changes_made.append(f"✅ Filled {missing_count} missing values in '{col}' with 'Unknown'")
    
    # Step 6: Handle outliers
    if options.get("handle_outliers", False):
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                # Cap outliers instead of removing them
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                changes_made.append(f"✅ Capped {len(outliers)} outliers in '{col}' using IQR method")
    
# In the advanced_preprocess_data function, update Step 7: Replace values

    # Step 7: Replace values in selected column (NEW FEATURE)
    if options.get("replace_values", False):
        replace_column = options.get("replace_column", None)
        replace_pattern = options.get("replace_pattern", "")
        replace_with = options.get("replace_with", "")
        
        if replace_column and replace_column in df.columns and replace_pattern:
            st.write(f"Debug: Attempting to replace '{replace_pattern}' with '{replace_with}' in column '{replace_column}'")
            
            # Count how many values will be changed
            if df[replace_column].dtype == 'object':
                # For string columns, use string methods
                mask = df[replace_column].astype(str).str.contains(replace_pattern, regex=True, na=False)
                count_changed = mask.sum()
                st.write(f"Debug: Found {count_changed} values to change")
                
                if count_changed > 0:
                    # Apply the replacement
                    original_values = df[replace_column].copy()
                    df[replace_column] = df[replace_column].astype(str).str.replace(
                        replace_pattern, replace_with, regex=True
                    )
                    
                    # Show before and after examples
                    st.write("Debug: Sample changes:")
                    for i in range(min(5, len(df))):
                        if mask.iloc[i]:
                            st.write(f"Before: '{original_values.iloc[i]}' → After: '{df[replace_column].iloc[i]}'")
                    
                    # Convert back to original dtype if possible
                    try:
                        if original_values.dtype != 'object':
                            df[replace_column] = pd.to_numeric(df[replace_column], errors='ignore')
                    except:
                        pass
                    
                    changes_made.append(
                        f"✅ Replaced '{replace_pattern}' with '{replace_with}' in {count_changed} values of '{replace_column}'"
                    )
            else:
                # For non-string columns, convert to string, replace, then convert back
                original_dtype = df[replace_column].dtype
                original_values = df[replace_column].copy()
                
                # Convert to string for replacement
                df[replace_column] = df[replace_column].astype(str)
                
                # Apply replacement
                mask = df[replace_column].str.contains(replace_pattern, regex=True, na=False)
                count_changed = mask.sum()
                st.write(f"Debug: Found {count_changed} values to change in non-string column")
                
                if count_changed > 0:
                    # Apply the replacement
                    df[replace_column] = df[replace_column].str.replace(
                        replace_pattern, replace_with, regex=True
                    )
                    
                    # Show before and after examples
                    st.write("Debug: Sample changes:")
                    for i in range(min(5, len(df))):
                        if mask.iloc[i]:
                            st.write(f"Before: '{original_values.iloc[i]}' → After: '{df[replace_column].iloc[i]}'")
                    
                    # Try to convert back to original dtype
                    try:
                        if original_dtype == 'int64':
                            df[replace_column] = pd.to_numeric(df[replace_column], errors='coerce').astype('Int64')
                        elif original_dtype == 'float64':
                            df[replace_column] = pd.to_numeric(df[replace_column], errors='coerce')
                        elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                            df[replace_column] = pd.to_datetime(df[replace_column], errors='coerce')
                    except Exception as e:
                        st.write(f"Debug: Could not convert back to original dtype: {e}")
                        # Keep as string if conversion fails
                        pass
                    
                    changes_made.append(
                        f"✅ Replaced '{replace_pattern}' with '{replace_with}' in {count_changed} values of '{replace_column}'"
                    )
        else:
            st.write("Debug: Replace values option selected but invalid parameters")
            if not replace_column:
                st.write("Debug: No column selected")
            if replace_column not in df.columns:
                st.write(f"Debug: Column '{replace_column}' not found in dataframe")
            if not replace_pattern:
                st.write("Debug: No pattern specified")
    # Step 8: Optimize data types
    if options.get("optimize_dtypes", True):
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to category if it has few unique values
            if df[col].nunique() < len(df) * 0.5:
                df[col] = df[col].astype('category')
                changes_made.append(f"✅ Optimized '{col}' to category type")
    
    return df, changes_made


def get_data_profile(df):
    """
    Generate a comprehensive profile of the dataset.
    
    Args:
        df: DataFrame to profile
        
    Returns:
        Dictionary containing dataset profile information
    """
    if df is None or df.empty:
        return {}
    
    profile = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'columns': {}
    }
    
    # Profile each column
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'missing_count': df[col].isnull().sum(),
            'missing_percent': round((df[col].isnull().sum() / len(df)) * 100, 2)
        }
        
        # Type-specific information
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            min_date = df[col].min()
            max_date = df[col].max()
            col_info.update({
                'min_date': min_date,
                'max_date': max_date,
                'date_range_days': (max_date - min_date).days if not pd.isna(min_date) and not pd.isna(max_date) else 0,
                'most_common_year': df[col].dt.year.mode()[0] if not df[col].dt.year.isna().all() else None
            })
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            col_info.update({
                'cardinality': df[col].nunique() / len(df)
            })
        
        profile['columns'][col] = col_info
    
    return profile


def download_cleaned_data(df):
    """
    Convert DataFrame to CSV string for download.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV string representation of the DataFrame
    """
    return df.to_csv(index=False)