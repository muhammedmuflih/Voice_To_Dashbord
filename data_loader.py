# utils/data_loader.py

import pandas as pd
import json
import yaml
from io import BytesIO
import streamlit as st
import os
import time
import base64
import numpy as np
from plotly.subplots import make_subplots

def flatten_json(y):
    """Flattens a nested JSON object."""
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def get_file_size(file_obj):
    """Get file size in MB."""
    file_obj.seek(0, os.SEEK_END)
    file_size = file_obj.tell() / (1024 * 1024)  # Size in MB
    file_obj.seek(0)
    return file_size

def load_data_in_chunks(uploaded_file, file_extension, chunk_size=100000):
    """Load large data files in chunks to handle memory efficiently."""
    if file_extension in ["csv"]:
        chunks = pd.read_csv(uploaded_file, chunksize=chunk_size)
        df = pd.concat(chunks)
    elif file_extension in ["xlsx"]:
        # For Excel, we'll use openpyxl with read_only mode for large files
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        # For other formats, we'll load normally
        return None, False
    
    return df, True

def extract_text_from_txt(uploaded_file):
    """Extract text from a TXT file."""
    try:
        # Read the file as text
        text = uploaded_file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return None

def extract_text_from_docx(uploaded_file):
    """Extract text from a DOCX file using python-docx."""
    try:
        # Import here to avoid errors if docx is not installed
        from docx import Document
        
        # Save the uploaded file temporarily
        with open("temp_doc.docx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Open the document
        doc = Document("temp_doc.docx")
        
        # Extract text
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Clean up temporary file
        os.remove("temp_doc.docx")
        
        return text
    except ImportError:
        st.error("Processing DOCX files requires the 'python-docx' package. Install with: pip install python-docx")
        return None
    except Exception as e:
        st.error(f"Error processing DOCX file: {e}")
        return None

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file using PyPDF2."""
    try:
        # Import here to avoid errors if PyPDF2 is not installed
        import PyPDF2
        
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        # Extract text from each page
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text
    except ImportError:
        st.error("Processing PDF files requires the 'PyPDF2' package. Install with: pip install PyPDF2")
        return None
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return None

def load_data(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    df = None
    extracted_text = None
    large_file_handled = False
    
    file_size = get_file_size(uploaded_file)
    st.info(f"Detected file type: .{file_extension}, Size: {file_size:.2f} MB")

    try:
        # Handle large files (over 500MB)
        if file_size > 500:
            st.warning(f"Large file detected ({file_size:.2f} MB). Loading in optimized mode...")
            if file_extension in ["csv", "xlsx"]:
                df, large_file_handled = load_data_in_chunks(uploaded_file, file_extension)
                if large_file_handled:
                    st.success(f"Large file loaded successfully. Shape: {df.shape}")
                else:
                    st.error("Failed to load large file. Please try a smaller sample or different format.")
                    return None
            else:
                st.warning("Large file handling is optimized for CSV and Excel files. Loading normally...")
        
        # If not a large file or large file handling failed, continue with normal loading
        if not large_file_handled:
            if file_extension in ["csv"]:
                df = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx"]:
                df = pd.read_excel(uploaded_file)
            elif file_extension in ["json"]:
                data = json.load(uploaded_file)
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    flattened_data = [flatten_json(item) for item in data]
                    df = pd.DataFrame(flattened_data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([flatten_json(data)])
                else:
                    st.warning("JSON file is not in a clear tabular format. Attempting to extract text.")
                    extracted_text = json.dumps(data, indent=2)
            elif file_extension in ["yaml", "yml"]:
                data = yaml.safe_load(uploaded_file)
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    flattened_data = [flatten_json(item) for item in data]
                    df = pd.DataFrame(flattened_data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([flatten_json(data)])
                else:
                    st.warning("YAML file is not in a clear tabular format. Attempting to extract text.")
                    extracted_text = yaml.dump(data, indent=2)
            elif file_extension in ["txt"]:
                extracted_text = extract_text_from_txt(uploaded_file)
            elif file_extension in ["md"]:
                extracted_text = extract_text_from_txt(uploaded_file)  # Markdown is processed as text
            elif file_extension in ["docx"]:
                extracted_text = extract_text_from_docx(uploaded_file)
            elif file_extension in ["pdf"]:
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                st.error(f"Unsupported file type: .{file_extension}")
                return None

        # If we have a DataFrame, check if it's too large to display fully
        if df is not None and len(df) > 10000:
            st.warning(f"Dataset contains {len(df):,} rows. Only showing a sample of 10,000 rows in the preview.")
            # Create a sample for preview but keep the full data for analysis
            preview_df = df.sample(n=10000, random_state=42)
            # Store the full df in session state for later use
            st.session_state.full_df = df
            df = preview_df

        # Detect and convert date columns
        if df is not None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains date-like values
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        # Try to convert to datetime
                        try:
                            pd.to_datetime(sample, errors='raise')
                            # If successful, convert the whole column
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            st.info(f"Automatically converted '{col}' to datetime format")
                        except:
                            pass

    except Exception as e:
        st.error(f"Failed to load or process {uploaded_file.name}: {e}")
        return None

    if df is not None:
        return df
    elif extracted_text is not None:
        st.warning("No structured data detected. Appending extracted text as a 'Document_Content' column.")
        return pd.DataFrame({"Document_Content": [extracted_text]})
    else:
        return None