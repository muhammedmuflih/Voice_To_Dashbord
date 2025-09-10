# utils/pdf_export.py

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Try to import from fpdf (not fpdf2)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    st.warning("⚠️ FPDF package not found. PDF export functionality will be limited. Install with: pip install fpdf")

# Alternative PDF generation using reportlab if fpdf is not available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def create_data_summary_pdf(df, profile, charts=None):
    """
    Create a PDF report with data summary and visualizations.
    
    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        profile (dict): The data profile.
        charts (list, optional): List of chart figures to include.
    
    Returns:
        bytes: The PDF data as bytes.
    """
    if FPDF_AVAILABLE:
        return create_data_summary_pdf_fpdf(df, profile, charts)
    elif REPORTLAB_AVAILABLE:
        return create_data_summary_pdf_reportlab(df, profile, charts)
    else:
        st.error("Neither FPDF nor ReportLab is available. Please install one of these packages to generate PDF reports.")
        return None

def create_data_summary_pdf_fpdf(df, profile, charts=None):
    """
    Create a PDF report using FPDF.
    """
    pdf = FPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'DataSense AI Executive Summary', 0, 1, 'C')
    pdf.ln(10)
    
    # Add generation timestamp
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    pdf.ln(10)
    
    # Dataset Overview
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Dataset Overview', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f'Shape: {profile["shape"][0]} rows x {profile["shape"][1]} columns', 0, 1, 'L')
    pdf.cell(0, 10, f'Memory Usage: {profile["memory_usage"]:.2f} MB', 0, 1, 'L')
    pdf.cell(0, 10, f'Duplicate Rows: {profile["duplicate_rows"]}', 0, 1, 'L')
    pdf.ln(10)
    
    # Missing Values
    missing_cols = [col for col, count in profile["missing_values"].items() if count > 0]
    if missing_cols:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Missing Values', 0, 1, 'L')
        pdf.set_font('Arial', '', 11)
        for col in missing_cols:
            pdf.cell(0, 10, f'{col}: {profile["missing_values"][col]} missing ({profile["columns"][col]["missing_percent"]:.2f}%)', 0, 1, 'L')
        pdf.ln(10)
    
    # Column Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Column Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    for col, info in profile["columns"].items():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'{col} ({info["dtype"]})', 0, 1, 'L')
        pdf.set_font('Arial', '', 11)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            pdf.cell(0, 10, f'Range: {info["min"]:.2f} to {info["max"]:.2f}', 0, 1, 'L')
            pdf.cell(0, 10, f'Mean: {info["mean"]:.2f}, Median: {info["median"]:.2f}', 0, 1, 'L')
            pdf.cell(0, 10, f'Standard Deviation: {info["std"]:.2f}', 0, 1, 'L')
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            pdf.cell(0, 10, f'Date Range: {info["min_date"].strftime("%Y-%m-%d")} to {info["max_date"].strftime("%Y-%m-%d")}', 0, 1, 'L')
            pdf.cell(0, 10, f'Days in Range: {info["date_range_days"]}', 0, 1, 'L')
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            pdf.cell(0, 10, f'Unique Values: {info["unique_count"]}, Cardinality: {info["cardinality"]}', 0, 1, 'L')
            pdf.cell(0, 10, 'Top Values:', 0, 1, 'L')
            for val, count in list(info["top_values"].items())[:3]:
                pdf.cell(0, 10, f'  {val}: {count} ({count/len(df)*100:.1f}%)', 0, 1, 'L')
        
        pdf.ln(5)
    
    # Add charts if provided
    if charts:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Data Visualizations', 0, 1, 'C')
        pdf.ln(10)
        
        for i, (title, fig) in enumerate(charts):
            # Save figure to a temporary buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Add to PDF
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, title, 0, 1, 'C')
            pdf.image(buf, x=10, y=None, w=180)
            pdf.ln(10)
    
    # Get PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes

def create_data_summary_pdf_reportlab(df, profile, charts=None):
    """
    Create a PDF report using ReportLab.
    """
    # Create a buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center alignment
    story.append(Paragraph("DataSense AI Executive Summary", title_style))
    story.append(Spacer(1, 12))
    
    # Timestamp
    timestamp_style = styles['Normal']
    timestamp_style.alignment = 2  # Right alignment
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style))
    story.append(Spacer(1, 12))
    
    # Dataset Overview
    story.append(Paragraph("Dataset Overview", styles['Heading2']))
    data = [
        ['Rows', str(profile["shape"][0])],
        ['Columns', str(profile["shape"][1])],
        ['Memory Usage (MB)', f"{profile['memory_usage']:.2f}"],
        ['Duplicate Rows', str(profile["duplicate_rows"])]
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Missing Values
    missing_cols = [col for col, count in profile["missing_values"].items() if count > 0]
    if missing_cols:
        story.append(Paragraph("Missing Values", styles['Heading2']))
        missing_data = [['Column', 'Missing Count', 'Missing %']]
        for col in missing_cols:
            missing_data.append([
                col,
                str(profile["missing_values"][col]),
                f"{profile['columns'][col]['missing_percent']:.2f}%"
            ])
        t = Table(missing_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
    
    # Column Summary
    story.append(Paragraph("Column Summary", styles['Heading2']))
    
    for col, info in profile["columns"].items():
        story.append(Paragraph(f"{col} ({info['dtype']})", styles['Heading3']))
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = [
                ['Statistic', 'Value'],
                ['Min', f"{info['min']:.2f}"],
                ['Max', f"{info['max']:.2f}"],
                ['Mean', f"{info['mean']:.2f}"],
                ['Median', f"{info['median']:.2f}"],
                ['Standard Deviation', f"{info['std']:.2f}"]
            ]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_data = [
                ['Statistic', 'Value'],
                ['Min Date', info['min_date'].strftime('%Y-%m-%d')],
                ['Max Date', info['max_date'].strftime('%Y-%m-%d')],
                ['Days in Range', str(info['date_range_days'])]
            ]
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            col_data = [['Statistic', 'Value'],
                       ['Unique Values', str(info['unique_count'])],
                       ['Cardinality', info['cardinality']],
                       ['Top Values', '']]
            
            for val, count in list(info["top_values"].items())[:3]:
                col_data.append([f"  {val}", f"{count} ({count/len(df)*100:.1f}%)"])  # Fixed the missing parenthesis here
        
        t = Table(col_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
    
    # Add charts if provided
    if charts:
        story.append(Paragraph("Data Visualizations", styles['Heading2']))
        
        for i, (title, fig) in enumerate(charts):
            story.append(Paragraph(title, styles['Heading3']))
            
            # Save figure to a temporary buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Add image to PDF
            img_path = f"temp_chart_{i}.png"
            with open(img_path, "wb") as f:
                f.write(buf.getvalue())
            
            story.append(Image(img_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Clean up temporary image files
    if charts:
        for i in range(len(charts)):
            img_path = f"temp_chart_{i}.png"
            if os.path.exists(img_path):
                os.remove(img_path)
    
    return pdf_data

def get_pdf_download_link(df, profile, charts=None, filename="data_summary.pdf"):
    """
    Generate a download link for the PDF report.
    
    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        profile (dict): The data profile.
        charts (list, optional): List of chart figures to include.
        filename (str): The filename for the PDF.
    
    Returns:
        str: HTML download link.
    """
    pdf_bytes = create_data_summary_pdf(df, profile, charts)
    if pdf_bytes:
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Report</a>'
        return href
    else:
        return "PDF generation failed. Required packages not available."