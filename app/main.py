import html
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.services import incident_service, ai_service, rag_service
from app.database import db
from app.utils import helpers
from app.config import (
    DEPARTMENTS,
    INCIDENT_STATUSES,
    SEVERITY_LEVELS,
)


_safe_html_text = helpers.safe_html_text
_clean_html_block = helpers.clean_html_block
_format_remediation_html = helpers.format_remediation_html
_format_fallback_guidance = helpers.format_fallback_guidance


st.set_page_config(
    page_title="OpsPilot AI | Operations Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F8FAFC;
    }

    [data-testid="stSidebar"] {
        background: #101B32 !important;
        color: #E7ECF5;
    }

    .brand {
        font-size: 21px;
        font-weight: 800;
        color: #FFFFFF !important;
        padding: 6px 0 2px 0;
        letter-spacing: -0.2px;
    }

    .brand-sub {
        color: #64748B !important;
        font-size: 10.5px !important;
        font-weight: 600 !important;
        letter-spacing: 1.2px !important;
        margin-bottom: 24px;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] h3 {
        color: #64748B !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        margin-top: 0 !important;
        margin-bottom: 8px !important;
        padding-left: 4px !important;
    }

    /* 1. Hide default circular radio indicator controls */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label > div > div:first-child,
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioControl"],
    [data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] + div > div:first-child {
        display: none !important;
    }

    /* 2. Navigation group container */
    [data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"],
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] {
        gap: 3px !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* 3. Navigation item row */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 9px 12px !important;
        margin: 1px 0 !important;
        border-radius: 7px !important;
        background: transparent !important;
        cursor: pointer !important;
        border: 1px solid transparent !important;
        border-left: 3px solid transparent !important;
        transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label > div {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        gap: 10px !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] {
        width: 100% !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
        color: #94A3B8 !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        letter-spacing: 0.15px !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        transition: color 0.15s ease !important;
    }

    /* 4. Shared SVG icons via CSS mask */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(1) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(2) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(3) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(4) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(5) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(6) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(7) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(1) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(2) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(3) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(4) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(5) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(6) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(7) [data-testid="stMarkdownContainer"] p::before {
        content: "" !important;
        display: inline-block !important;
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        margin-right: 10px !important;
        background-color: currentColor !important;
        -webkit-mask-size: contain !important;
        mask-size: contain !important;
        -webkit-mask-repeat: no-repeat !important;
        mask-repeat: no-repeat !important;
        -webkit-mask-position: center !important;
        mask-position: center !important;
        opacity: 0.8 !important;
        transition: opacity 0.15s ease, background-color 0.15s ease !important;
    }

    /* 1. Incident Journey: Compass navigation icon */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(1) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(1) [data-testid="stMarkdownContainer"] p::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolygon points='16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76'/%3E%3C/svg%3E") !important;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolygon points='16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76'/%3E%3C/svg%3E") !important;
    }

    /* 2. Overview: Grid dashboard */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(2) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(2) [data-testid="stMarkdownContainer"] p::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='9' rx='1'/%3E%3Crect x='14' y='3' width='7' height='5' rx='1'/%3E%3Crect x='14' y='12' width='7' height='9' rx='1'/%3E%3Crect x='3' y='16' width='7' height='5' rx='1'/%3E%3C/svg%3E") !important;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='9' rx='1'/%3E%3Crect x='14' y='3' width='7' height='5' rx='1'/%3E%3Crect x='14' y='12' width='7' height='9' rx='1'/%3E%3Crect x='3' y='16' width='7' height='5' rx='1'/%3E%3C/svg%3E") !important;
    }

    /* 3. Incident Center: Alert triangle / Shield */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(3) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(3) [data-testid="stMarkdownContainer"] p::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z'/%3E%3Cline x1='12' y1='9' x2='12' y2='13'/%3E%3Cline x1='12' y1='17' x2='12.01' y2='17'/%3E%3C/svg%3E") !important;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z'/%3E%3Cline x1='12' y1='9' x2='12' y2='13'/%3E%3Cline x1='12' y1='17' x2='12.01' y2='17'/%3E%3C/svg%3E") !important;
    }

    /* 4. AI Investigation: Search analysis */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(4) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(4) [data-testid="stMarkdownContainer"] p::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3Cpath d='M11 8v6M8 11h6'/%3E%3C/svg%3E") !important;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3Cpath d='M11 8v6M8 11h6'/%3E%3C/svg%3E") !important;
    }

    /* 5. Knowledge Base: Book repository */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(5) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(5) [data-testid="stMarkdownContainer"] p::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z'/%3E%3Cpath d='M8 6h8'/%3E%3Cpath d='M8 10h8'/%3E%3C/svg%3E") !important;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z'/%3E%3Cpath d='M8 6h8'/%3E%3Cpath d='M8 10h8'/%3E%3C/svg%3E") !important;
    }

    /* 6. Simulation Lab: Flask / beaker */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(6) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(6) [data-testid="stMarkdownContainer"] p::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 2v7.31L4.15 19.1A2 2 0 0 0 5.86 22h12.28a2 2 0 0 0 1.71-2.9L14 9.31V2'/%3E%3Cline x1='8' y1='2' x2='16' y2='2'/%3E%3Cline x1='9' y1='12' x2='15' y2='12'/%3E%3C/svg%3E") !important;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 2v7.31L4.15 19.1A2 2 0 0 0 5.86 22h12.28a2 2 0 0 0 1.71-2.9L14 9.31V2'/%3E%3Cline x1='8' y1='2' x2='16' y2='2'/%3E%3Cline x1='9' y1='12' x2='15' y2='12'/%3E%3C/svg%3E") !important;
    }

    /* 7. Evaluation: Bar chart */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > *:nth-child(7) [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > *:nth-child(7) [data-testid="stMarkdownContainer"] p::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E") !important;
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E") !important;
    }

    /* 5. Active / Selected Navigation Item */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected],
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[data-selected] label,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: rgba(49, 91, 234, 0.18) !important;
        border-left: 3px solid #3B82F6 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[data-selected] label [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected] [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[data-selected] label [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) [data-testid="stMarkdownContainer"] p::before {
        opacity: 1 !important;
        background-color: #60A5FA !important;
    }

    /* 6. Tasteful Hover State on Inactive Rows */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover:not([data-selected]):not(:has(input:checked)),
    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-hovered]:not([data-selected]):not(:has(input:checked)) {
        background: rgba(255, 255, 255, 0.04) !important;
        border-left: 3px solid rgba(148, 163, 184, 0.4) !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-hovered] [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover [data-testid="stMarkdownContainer"] p::before,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-hovered] [data-testid="stMarkdownContainer"] p::before {
        opacity: 1 !important;
    }

    /* 7. Keyboard Focus Accessibility */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:focus-within,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-focus-visible] {
        outline: 2px solid #3B82F6 !important;
        outline-offset: 1px !important;
    }

    /* 8. Sidebar dividers and captions */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 20px 0 16px 0 !important;
    }

    [data-testid="stSidebar"] .stCaption, 
    [data-testid="stSidebar"] small, 
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #64748B !important;
        font-size: 11px !important;
        letter-spacing: 0.3px !important;
    }

    .eyebrow {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    .main-title {
        color: #0F172A;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #315BEA !important;
        border-radius: 10px !important;
        padding: 18px 20px !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04) !important;
        margin-top: 14px;
        margin-bottom: 16px;
    }

    .metric-label {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #0F172A;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #64748B;
        font-size: 11px;
        margin-top: 5px;
    }

    .section-heading {
        color: #0F172A;
        font-size: 19px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: #315BEA;
        color: white !important;
        border: 1px solid #315BEA !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 8px 18px !important;
        box-shadow: 0 1px 2px rgba(49, 91, 234, 0.1) !important;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover {
        background: #2448C5 !important;
        border-color: #2448C5 !important;
        color: white !important;
        box-shadow: 0 3px 8px rgba(49, 91, 234, 0.25) !important;
    }

    .stButton > button:active {
        background: #1D3DB5 !important;
        transform: translateY(1px);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(49, 91, 234, 0.3) !important;
        outline: none !important;
    }

    /* Forms & Inputs */
    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 22px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }

    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        font-size: 13.5px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
        border-color: #315BEA !important;
        box-shadow: 0 0 0 3px rgba(49, 91, 234, 0.15) !important;
        outline: none !important;
    }

    /* Selectboxes and Dropdowns */
    div[data-baseweb="select"] {
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        min-height: 42px !important;
    }

    div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {
        color: #0F172A !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #64748B !important;
    }

    div[data-baseweb="select"]:focus-within > div {
        border-color: #315BEA !important;
        box-shadow: 0 0 0 3px rgba(49, 91, 234, 0.15) !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08) !important;
    }

    li[data-baseweb="menu-item"] {
        color: #0F172A !important;
        font-size: 13.5px !important;
        padding: 8px 14px !important;
    }

    li[data-baseweb="menu-item"]:hover,
    li[data-baseweb="menu-item"][aria-selected="true"] {
        background-color: #EEF2FF !important;
        color: #315BEA !important;
        font-weight: 600 !important;
    }

    /* Form Labels & Captions */
    label[data-testid="stWidgetLabel"] p {
        color: #1E293B !important;
        font-weight: 650 !important;
        font-size: 13px !important;
        letter-spacing: 0.2px;
        margin-bottom: 2px !important;
    }

    div[data-testid="stWidgetLabel"] label {
        color: #1E293B !important;
    }

    .stCaption, small, [data-testid="stCaptionContainer"] {
        color: #64748B !important;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E2E8F0 !important;
        background: transparent !important;
    }

    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 9px 16px !important;
        border-radius: 6px 6px 0 0 !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease;
    }

    button[data-baseweb="tab"]:hover {
        color: #315BEA !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #315BEA !important;
        border-bottom: 2px solid #315BEA !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #315BEA !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 9px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 650 !important;
        color: #1E293B !important;
        font-size: 13.5px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #315BEA !important;
    }

    div[data-testid="stExpander"] summary svg {
        fill: #64748B !important;
    }

    /* Checkboxes */
    div[data-testid="stCheckbox"] label p {
        color: #1E293B !important;
        font-size: 13px !important;
        font-weight: 550 !important;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* Citations and Badges */
    .citation-badge {
        background: #F1F5F9;
        color: #1E293B;
        border: 1px solid #CBD5E1;
        font-family: monospace;
        font-size: 11px;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
        margin: 2px 3px 2px 0;
    }

    .badge-high {
        background: #FFF1F2;
        color: #BE123C;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #FECDD3;
    }

    .badge-med {
        background: #EFF6FF;
        color: #1D4ED8;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #BFDBFE;
    }

    .badge-low {
        background: #F8FAFC;
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
    }

    .diag-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    /* Guided Incident Journey Stepper */
    .journey-stepper-container {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 16px;
        margin-top: 10px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        gap: 6px;
        overflow-x: auto;
    }

    .journey-step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        flex: 1;
        min-width: 110px;
    }

    .journey-step-badge {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .journey-step-badge.completed {
        background: #315BEA;
        color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(49, 91, 234, 0.25);
    }

    .journey-step-badge.active, .journey-step-badge.ready {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 2px solid #3B82F6;
    }

    .journey-step-badge.pending {
        background: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
    }

    .journey-step-badge.attention {
        background: #FFFBEB;
        color: #B45309;
        border: 1.5px solid #F59E0B;
    }

    .journey-step-badge.fallback {
        background: #FEF3C7;
        color: #92400E;
        border: 1.5px solid #D97706;
    }

    .journey-step-title {
        font-size: 12px;
        font-weight: 650;
        color: #0F172A;
        line-height: 1.25;
    }

    .journey-step-subtext {
        font-size: 10.5px;
        color: #64748B;
        margin-top: 3px;
        line-height: 1.2;
    }

    .journey-step-pill {
        font-size: 9.5px;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 10px;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        display: inline-block;
    }

    .journey-step-pill.completed {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }

    .journey-step-pill.active, .journey-step-pill.ready {
        background: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }

    .journey-step-pill.pending {
        background: #F1F5F9;
        color: #64748B;
        border: 1px solid #E2E8F0;
    }

    .journey-step-pill.attention {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    .journey-step-pill.fallback {
        background: #FFEDD5;
        color: #C2410C;
        border: 1px solid #FED7AA;
    }

    /* Stage Inspection Card */
    .journey-stage-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #315BEA;
        border-radius: 10px;
        padding: 22px 24px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        margin-top: 14px;
        margin-bottom: 20px;
    }

    .journey-narrative-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 14px;
        margin-bottom: 16px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="brand">🛡️ OpsPilot AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">OPERATIONS INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)

    st.markdown("### WORKSPACE")

    nav_options = [
        "Incident Journey",
        "Overview",
        "Incident Center",
        "AI Investigation",
        "Knowledge Base",
        "Simulation Lab",
        "Evaluation"
    ]
    target_nav = st.session_state.pop("nav_page", None)
    if target_nav in nav_options:
        st.session_state["main_nav_radio"] = target_nav

    if "main_nav_radio" in st.session_state:
        page = st.radio(
            "Navigation",
            nav_options,
            key="main_nav_radio",
            label_visibility="collapsed"
        )
    else:
        page = st.radio(
            "Navigation",
            nav_options,
            index=0,
            key="main_nav_radio",
            label_visibility="collapsed"
        )

    st.divider()
    st.caption("AI-POWERED DECISION SUPPORT")
    st.caption("Version 1.0 | Development")

# ==============================================================================
# GUIDED INCIDENT JOURNEY: STAGE STATUS DERIVATION (STEP 8.1)
# ==============================================================================

def compute_journey_stage_statuses(
    incident: Optional[Dict[str, Any]],
    investigations: List[Dict[str, Any]],
    audit_logs: List[Dict[str, Any]],
    total_docs: int = 0
) -> Dict[int, Dict[str, Any]]:
    """Derive real status for all 7 stages from actual persisted records and available evidence.

    Epistemic & Governance Constraints:
    1. Never mark all seven journey stages completed merely because an investigation was approved.
    2. Derive stage statuses from actual persisted records and available evidence.
    3. Distinguish recommendation approval from actual remediation execution and verified incident resolution.
    4. Do not invent retrieval results, evidence, classification outcomes or AI analysis.
    5. Display missing Knowledge Base documents and unavailable Gemini analysis honestly.
    6. Keep technical terms understandable through simple user-facing labels and explanations.
    """
    stages: Dict[int, Dict[str, Any]] = {}

    # Stage 1: Incident Received
    if not incident:
        stages[1] = {
            "name": "Incident Received",
            "short_title": "1. Received",
            "state": "pending",
            "badge": "1",
            "pill": "Pending",
            "label": "Awaiting Incident",
            "subtext": "No incident selected or recorded",
            "is_completed": False,
            "narrative": "An operational disruption or service anomaly is reported and officially recorded with an immutable timestamp and unique tracking identifier.",
            "evidence_summary": "No incident record currently selected.",
        }
    else:
        stages[1] = {
            "name": "Incident Received",
            "short_title": "1. Received",
            "state": "completed",
            "badge": "✓",
            "pill": "Completed",
            "label": "Incident Logged",
            "subtext": f"Tracking ID: {incident.get('formatted_id', 'INC-????')}",
            "is_completed": True,
            "narrative": "The operational disruption was captured and registered into the system with an immutable creation timestamp.",
            "evidence_summary": f"Incident {incident.get('formatted_id')} registered: \"{incident.get('title')}\" reported at {helpers.format_timestamp(incident.get('created_at'))}.",
        }

    # Stage 2: Impact & Classification
    if not incident:
        stages[2] = {
            "name": "Impact & Classification",
            "short_title": "2. Impact",
            "state": "pending",
            "badge": "2",
            "pill": "Pending",
            "label": "Awaiting Intake",
            "subtext": "Requires recorded incident",
            "is_completed": False,
            "narrative": "Assessing business blast radius, operational department, and technical category to establish response urgency and priority.",
            "evidence_summary": "Pending incident selection.",
        }
    else:
        cat = incident.get("category")
        sev = incident.get("severity")
        dept = incident.get("department")
        impact = incident.get("impact")
        is_classified = bool(
            cat and sev
            and cat != "Pending AI Classification"
            and sev != "Pending AI Assessment"
        )
        if is_classified:
            stages[2] = {
                "name": "Impact & Classification",
                "short_title": "2. Impact",
                "state": "completed",
                "badge": "✓",
                "pill": "Classified",
                "label": f"{sev} Severity",
                "subtext": f"{cat} ({dept or 'General'})",
                "is_completed": True,
                "narrative": "Business impact and domain categorization have been established and confirmed against standard operational taxonomies.",
                "evidence_summary": f"Severity: {sev} | Category: {cat} | Department: {dept or 'Unassigned'} | Business Impact: {impact or 'Not set'}.",
            }
        else:
            stages[2] = {
                "name": "Impact & Classification",
                "short_title": "2. Impact",
                "state": "attention",
                "badge": "⚠️",
                "pill": "Action Needed",
                "label": "Unclassified",
                "subtext": "Pending category & severity assessment",
                "is_completed": False,
                "narrative": "The incident requires category and severity assessment before diagnostic procedures can be prioritized.",
                "evidence_summary": "Department, category, and severity assessment have not yet been completed.",
            }

    # Stage 3: Knowledge Retrieval (RAG)
    if not incident:
        stages[3] = {
            "name": "Knowledge Retrieval",
            "short_title": "3. Retrieval",
            "state": "pending",
            "badge": "3",
            "pill": "Pending",
            "label": "Awaiting Intake",
            "subtext": "Requires recorded incident",
            "is_completed": False,
            "narrative": "Full-text search (BM25) across internal runbooks and standard operating procedures (SOPs) to find matching resolution guidance.",
            "evidence_summary": "Pending incident selection.",
        }
    elif total_docs == 0:
        stages[3] = {
            "name": "Knowledge Retrieval",
            "short_title": "3. Retrieval",
            "state": "attention",
            "badge": "ℹ️",
            "pill": "0 Docs Indexed",
            "label": "0 Runbooks Indexed",
            "subtext": "No documentation ingested in Knowledge Base",
            "is_completed": False,
            "narrative": "OpsPilot scans enterprise runbooks and standard operating procedures (SOPs). Currently, 0 documents are indexed in the Knowledge Base.",
            "evidence_summary": "Knowledge Base contains 0 indexed documents. Retrieval returned 0 reference chunks. Investigation will proceed with general diagnostic reasoning.",
        }
    else:
        chunks = []
        if investigations:
            chunks = investigations[0].get("retrieved_evidence") or []
        if chunks:
            stages[3] = {
                "name": "Knowledge Retrieval",
                "short_title": "3. Retrieval",
                "state": "completed",
                "badge": "✓",
                "pill": "Retrieved",
                "label": f"{len(chunks)} Runbook Chunks",
                "subtext": "Relevant operational guidance matched",
                "is_completed": True,
                "narrative": "Internal documentation was scanned and relevant procedure chunks were retrieved to ground the diagnostic investigation.",
                "evidence_summary": f"Retrieved {len(chunks)} relevant runbook chunks from {total_docs} indexed documents via BM25 full-text search.",
            }
        else:
            stages[3] = {
                "name": "Knowledge Retrieval",
                "short_title": "3. Retrieval",
                "state": "ready",
                "badge": "3",
                "pill": "Ready",
                "label": f"{total_docs} Runbooks Available",
                "subtext": "Ready for automated search",
                "is_completed": False,
                "narrative": "Internal documents are indexed and ready for automated search once diagnostic analysis is initiated.",
                "evidence_summary": f"{total_docs} standard runbooks indexed in Knowledge Base. Ready for automated query extraction.",
            }

    # Stage 4: AI Root Cause Investigation
    if not incident or not investigations:
        stages[4] = {
            "name": "AI Root Cause Investigation",
            "short_title": "4. AI Analysis",
            "state": "pending",
            "badge": "4",
            "pill": "Pending",
            "label": "Not Started",
            "subtext": "Awaiting diagnostic analysis",
            "is_completed": False,
            "narrative": "AI diagnostic engine analyzes reported symptoms against technical runbooks while enforcing strict verification rules (separating unverified symptoms from confirmed telemetry).",
            "evidence_summary": "No AI diagnostic investigation has been executed for this incident yet.",
        }
    else:
        latest_inv = investigations[0]
        is_fallback = bool(
            latest_inv.get("is_fallback")
            or latest_inv.get("error_type")
            or "fallback" in (latest_inv.get("mode") or "").lower()
        )
        if is_fallback:
            err_type = latest_inv.get("error_type", "offline")
            stages[4] = {
                "name": "AI Root Cause Investigation",
                "short_title": "4. AI Analysis",
                "state": "fallback",
                "badge": "⚡",
                "pill": "Model Offline",
                "label": "Fallback Active",
                "subtext": f"Operator Manual Mode ({err_type})",
                "is_completed": True,
                "narrative": "Automated AI diagnostic synthesis was unavailable (due to service timeout or unconfigured credentials). OpsPilot is operating in manual operator mode with strict evidence controls.",
                "evidence_summary": f"Model Offline ({err_type}). Automated AI analysis was not generated. Operating under human-directed investigation.",
            }
        else:
            conf = latest_inv.get("confidence")
            conf_str = f"{int(conf * 100)}%" if conf is not None else "Model Confirmed"
            stages[4] = {
                "name": "AI Root Cause Investigation",
                "short_title": "4. AI Analysis",
                "state": "completed",
                "badge": "✓",
                "pill": "Diagnosed",
                "label": f"AI Diagnosed ({conf_str})",
                "subtext": "Root-cause hypotheses formulated",
                "is_completed": True,
                "narrative": "AI diagnostic synthesis completed successfully. Hypotheses were formulated and strictly grounded against retrieved runbook evidence.",
                "evidence_summary": f"Diagnostic analysis complete. Model reported confidence: {conf_str}. Hypotheses grounded in verified runbooks.",
            }

    # Stage 5: Remediation Plan
    if not incident or not investigations:
        stages[5] = {
            "name": "Remediation Plan",
            "short_title": "5. Remediation",
            "state": "pending",
            "badge": "5",
            "pill": "Pending",
            "label": "Not Started",
            "subtext": "Awaiting diagnostic findings",
            "is_completed": False,
            "narrative": "Formulating actionable step-by-step technical procedures to remediate root causes and restore normal service operations.",
            "evidence_summary": "No remediation plan formulated yet.",
        }
    else:
        latest_inv = investigations[0]
        final_act = latest_inv.get("final_action")
        rec = latest_inv.get("ai_recommendation")
        is_fallback = bool(
            latest_inv.get("is_fallback")
            or latest_inv.get("error_type")
            or "fallback" in (latest_inv.get("mode") or "").lower()
        )
        if final_act:
            tag = "Operator Plan" if is_fallback else "Approved Action"
            snippet = helpers.clean_text(final_act)
            if len(snippet) > 120:
                snippet = snippet[:120].rstrip() + "..."
            stages[5] = {
                "name": "Remediation Plan",
                "short_title": "5. Remediation",
                "state": "completed",
                "badge": "✓",
                "pill": "Formulated",
                "label": tag,
                "subtext": "Concrete resolution steps formulated",
                "is_completed": True,
                "narrative": "A concrete technical remediation plan has been authored and approved to resolve the operational fault.",
                "evidence_summary": f"Remediation plan established: {snippet}",
            }
        elif rec:
            has_cites = bool(latest_inv.get("evidence_citations"))
            if is_fallback and not has_cites:
                rec_clean = helpers.format_fallback_guidance(rec, citations=[], total_docs=total_docs)
            else:
                rec_clean = rec
            snippet = helpers.clean_text(rec_clean)
            if len(snippet) > 120:
                snippet = snippet[:120].rstrip() + "..."
            stages[5] = {
                "name": "Remediation Plan",
                "short_title": "5. Remediation",
                "state": "completed",
                "badge": "✓",
                "pill": "Proposed",
                "label": "Proposed Plan Ready",
                "subtext": "Awaiting human review & approval",
                "is_completed": True,
                "narrative": "A remediation checklist has been proposed based on retrieved operational runbooks.",
                "evidence_summary": f"Proposed procedure ready for review: {snippet}",
            }
        else:
            stages[5] = {
                "name": "Remediation Plan",
                "short_title": "5. Remediation",
                "state": "attention",
                "badge": "⚠️",
                "pill": "Action Needed",
                "label": "Action Needed",
                "subtext": "Remediation procedure required",
                "is_completed": False,
                "narrative": "A specific remediation procedure must be authored or generated before human sign-off.",
                "evidence_summary": "No remediation actions currently drafted.",
            }

    # Stage 6: Human Review & Decision
    if not incident or not investigations:
        stages[6] = {
            "name": "Human Review & Decision",
            "short_title": "6. Governance",
            "state": "pending",
            "badge": "6",
            "pill": "Pending",
            "label": "Not Started",
            "subtext": "Awaiting remediation plan",
            "is_completed": False,
            "narrative": "AI advises while human operators decide. An authorized operator must review, customize if necessary, and approve or reject the proposed actions.",
            "evidence_summary": "Pending remediation plan.",
        }
    else:
        latest_inv = investigations[0]
        decision = latest_inv.get("human_decision", "Pending")
        actor = latest_inv.get("actor") or "Operator"
        if decision == "Approved":
            rev_notes = helpers.clean_text(latest_inv.get("reviewer_notes") or "None recorded")
            stages[6] = {
                "name": "Human Review & Decision",
                "short_title": "6. Governance",
                "state": "completed",
                "badge": "✓",
                "pill": "Approved",
                "label": "Plan Approved",
                "subtext": f"Approved by {actor}",
                "is_completed": True,
                "narrative": "Human-in-the-loop governance complete: The remediation plan was reviewed and formally approved by an authorized operator.",
                "evidence_summary": f"Decision: Approved by {actor}. Notes: \"{rev_notes}\".",
            }
        elif decision == "Rejected":
            stages[6] = {
                "name": "Human Review & Decision",
                "short_title": "6. Governance",
                "state": "completed",
                "badge": "✓",
                "pill": "Rejected",
                "label": "Proposal Rejected",
                "subtext": f"Rejected by {actor}",
                "is_completed": True,
                "narrative": "Human governance complete: The proposed recommendation was rejected by the operator. Incident status reset to Open.",
                "evidence_summary": f"Decision: Rejected by {actor}. Reason: \"{latest_inv.get('reviewer_notes') or 'Operator rejected'}\".",
            }
        else:
            stages[6] = {
                "name": "Human Review & Decision",
                "short_title": "6. Governance",
                "state": "attention",
                "badge": "⚠️",
                "pill": "Review Needed",
                "label": "Awaiting Decision",
                "subtext": "Human operator sign-off required",
                "is_completed": False,
                "narrative": "A remediation recommendation is pending human review. Critical system modifications require explicit operator approval before execution.",
                "evidence_summary": "Awaiting human review decision ('Approve As-Is', 'Edit & Approve', or 'Confirm Rejection').",
            }

    # Stage 7: Outcome & Audit Trail
    # MANDATORY CORRECTIONS 1, 2, 3:
    # Never mark all seven journey stages completed merely because an investigation was approved.
    # Distinguish recommendation approval from actual remediation execution and verified incident resolution.
    if not incident:
        stages[7] = {
            "name": "Outcome & Audit Trail",
            "short_title": "7. Outcome",
            "state": "pending",
            "badge": "7",
            "pill": "Pending",
            "label": "Not Started",
            "subtext": "Awaiting incident lifecycle",
            "is_completed": False,
            "narrative": "Verification of operational execution, system health recovery, incident closure, and immutable audit logging.",
            "evidence_summary": "No incident active.",
        }
    else:
        inc_status = str(incident.get("status") or "Open")
        latest_audit = audit_logs[-1] if audit_logs else None
        audit_count = len(audit_logs)
        latest_act = latest_audit.get("action", "None") if latest_audit else "None"

        if inc_status in ["Resolved", "Closed"]:
            stages[7] = {
                "name": "Outcome & Audit Trail",
                "short_title": "7. Outcome",
                "state": "completed",
                "badge": "✓",
                "pill": "Resolved",
                "label": f"Incident {inc_status}",
                "subtext": "Remediation verified & incident closed",
                "is_completed": True,
                "narrative": "Remediation actions were executed in production, service health was verified, and the incident has been closed with an immutable governance audit trail.",
                "evidence_summary": f"Incident officially {inc_status}. {audit_count} immutable audit log entries recorded. Latest event: {latest_act}.",
            }
        elif inc_status == "Approved":
            # CRITICAL DISTINCTION: Recommendation is approved, but execution and verified resolution are pending!
            stages[7] = {
                "name": "Outcome & Audit Trail",
                "short_title": "7. Outcome",
                "state": "attention",
                "badge": "⏳",
                "pill": "Awaiting Execution",
                "label": "Awaiting Execution",
                "subtext": "Remediation approved — Resolution pending",
                "is_completed": False,
                "narrative": "Operational Distinction: The remediation plan has been approved by the human operator, but actual technical execution and verified incident resolution are pending. The incident remains active in the system until service recovery is confirmed.",
                "evidence_summary": f"Current Status: Approved (Plan approved; execution and resolution pending). {audit_count} audit logs recorded. Latest governance event: {latest_act}.",
            }
        elif inc_status == "Mitigated":
            stages[7] = {
                "name": "Outcome & Audit Trail",
                "short_title": "7. Outcome",
                "state": "attention",
                "badge": "⏳",
                "pill": "Mitigated",
                "label": "Mitigated (Verification Pending)",
                "subtext": "Temporary fix active — Monitoring recovery",
                "is_completed": False,
                "narrative": "A temporary mitigation has been applied to reduce customer impact. Full incident closure requires permanent resolution verification.",
                "evidence_summary": f"Current Status: Mitigated. Permanent resolution verification pending. {audit_count} audit logs recorded.",
            }
        elif inc_status == "Investigating":
            stages[7] = {
                "name": "Outcome & Audit Trail",
                "short_title": "7. Outcome",
                "state": "pending",
                "badge": "7",
                "pill": "Investigating",
                "label": "Under Investigation",
                "subtext": "Diagnostic cycle in progress",
                "is_completed": False,
                "narrative": "The incident is actively under investigation. Resolution outcome cannot be determined until diagnostic procedures conclude.",
                "evidence_summary": f"Current Status: Investigating. {audit_count} audit logs recorded.",
            }
        else: # "Open"
            stages[7] = {
                "name": "Outcome & Audit Trail",
                "short_title": "7. Outcome",
                "state": "pending",
                "badge": "7",
                "pill": "Open",
                "label": "Incident Open",
                "subtext": "Unresolved — Lifecycle beginning",
                "is_completed": False,
                "narrative": "The incident is open and unresolved. No approved remediation or resolution has taken place yet.",
                "evidence_summary": f"Current Status: Open. {audit_count} audit logs recorded. Latest event: {latest_act}.",
            }

    return stages


# ROUTING & PAGES
if page == "Incident Journey":
    st.markdown('<div class="eyebrow">GUIDED WORKFLOW / INCIDENT LIFECYCLE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Guided Incident Journey</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Experience an end-to-end, evidence-grounded incident investigation from initial report to audit closure.</div>',
        unsafe_allow_html=True
    )

    try:
        all_incidents = incident_service.get_all_incidents()
    except Exception as err:
        st.error(f"Error loading incidents: {err}")
        all_incidents = []

    if not all_incidents:
        st.info("No incidents have been logged yet. Capture an incident from the **Overview** dashboard to begin the Incident Journey.")
    else:
        inc_options = [
            f"{inc.get('formatted_id')} • {inc.get('title')}"
            for inc in all_incidents
        ]
        target_id = st.session_state.get("journey_incident_id") or st.session_state.get("selected_investigation_incident_id")
        default_idx = 0
        if target_id:
            for idx, inc in enumerate(all_incidents):
                if inc["id"] == target_id:
                    default_idx = idx
                    break

        sel_col1, sel_col2 = st.columns([3, 1])
        with sel_col1:
            selected_label = st.selectbox(
                "Select Incident for Guided Journey",
                options=inc_options,
                index=default_idx,
                key="journey_incident_selector"
            )

        selected_idx = inc_options.index(selected_label)
        selected_incident_meta = all_incidents[selected_idx]
        selected_id = selected_incident_meta["id"]
        st.session_state["journey_incident_id"] = selected_id

        try:
            selected_incident = incident_service.get_incident_by_id(selected_id) or selected_incident_meta
        except Exception as err:
            st.error(f"Error fetching incident details: {err}")
            selected_incident = selected_incident_meta

        try:
            existing_inv_rows = incident_service.get_investigations_for_incident(selected_id)
            deserialized_invs = [incident_service.deserialize_investigation(r) for r in existing_inv_rows]
        except Exception:
            deserialized_invs = []

        try:
            audit_logs = incident_service.get_audit_history_for_incident(selected_id)
        except Exception:
            audit_logs = []

        try:
            total_docs = len(db.get_all_knowledge_documents())
        except Exception:
            total_docs = 0

        # Compute statuses across all 7 stages
        stage_statuses = compute_journey_stage_statuses(
            incident=selected_incident,
            investigations=deserialized_invs,
            audit_logs=audit_logs,
            total_docs=total_docs
        )

        # Overview Card for Selected Incident
        esc_id = _safe_html_text(selected_incident.get("formatted_id", "N/A"))
        esc_status = _safe_html_text(str(selected_incident.get("status", "Open")).upper())
        esc_created = _safe_html_text(helpers.format_timestamp(selected_incident.get("created_at")))
        esc_title = _safe_html_text(selected_incident.get("title", ""))
        esc_desc = _safe_html_text(selected_incident.get("description", ""), preserve_newlines=True)
        esc_dept = _safe_html_text(selected_incident.get("department") or "Unassigned")
        esc_sev = _safe_html_text(selected_incident.get("severity") or "Pending Assessment")
        esc_cat = _safe_html_text(selected_incident.get("category") or "Pending Classification")

        st.markdown(_clean_html_block(f"""
        <div class="metric-card" style="margin-top: 10px; margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="eyebrow">{esc_id} • STATUS: {esc_status}</span>
                <span class="metric-note">Reported: {esc_created}</span>
            </div>
            <div class="main-title" style="font-size: 20px; margin-top: 6px;">{esc_title}</div>
            <div style="color: #4A5568; font-size: 13.5px; margin-top: 8px; line-height: 1.5; word-break: break-word;">
                {esc_desc}
            </div>
            <div style="margin-top: 12px; display: flex; gap: 24px; font-size: 12px;">
                <div><span style="color: #64748B; margin-right: 4px;">Department:</span><strong style="color: #14213D;">{esc_dept}</strong></div>
                <div><span style="color: #64748B; margin-right: 4px;">Category:</span><strong style="color: #14213D;">{esc_cat}</strong></div>
                <div><span style="color: #64748B; margin-right: 4px;">Severity:</span><strong style="color: #14213D;">{esc_sev}</strong></div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Interactive Seven-Stage Journey Navigation (Single cohesive interactive experience)
        tab_names = [
            f"{stage_statuses[i]['badge']} {stage_statuses[i]['short_title']}"
            for i in range(1, 8)
        ]
        tabs = st.tabs(tab_names)

        for i in range(1, 8):
            stg = stage_statuses[i]
            with tabs[i - 1]:
                s_name = _safe_html_text(stg["name"])
                s_badge = _safe_html_text(stg["badge"])
                s_pill = _safe_html_text(stg["pill"])
                s_state = _safe_html_text(stg["state"])
                s_sub = _safe_html_text(stg["subtext"])
                s_narr = _safe_html_text(stg["narrative"])
                s_ev = _safe_html_text(stg["evidence_summary"])

                st.markdown(_clean_html_block(f"""
                <div class="journey-stage-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="eyebrow">STAGE {i} OF 7 • STATUS: {s_pill.upper()}</span>
                        <span class="journey-step-pill {s_state}">{s_pill}</span>
                    </div>
                    <div class="main-title" style="font-size: 21px; margin-top: 8px;">{s_badge} {s_name}</div>
                    <div style="color: #64748B; font-size: 13.5px; margin-top: 2px;">{s_sub}</div>

                    <div class="journey-narrative-box">
                        <strong style="color: #0F172A; font-size: 13px;">📖 OPERATIONAL CONTEXT</strong>
                        <div style="color: #334155; font-size: 13.5px; margin-top: 6px; line-height: 1.5;">
                            {s_narr}
                        </div>
                    </div>

                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-top: 10px;">
                        <strong style="color: #0F172A; font-size: 13px;">🔍 PERSISTED RECORD &amp; EVIDENCE</strong>
                        <div style="color: #475569; font-size: 13px; margin-top: 6px; line-height: 1.5;">
                            {s_ev}
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

                # Navigation Deep-link buttons to specialized workspaces
                if i in (1, 2):
                    if st.button("Open Incident Center Workspace ➔", key=f"journey_btn_ic_{i}"):
                        st.session_state["nav_page"] = "Incident Center"
                        st.rerun()
                elif i == 3:
                    if st.button("Open Knowledge Base Workspace ➔", key="journey_btn_kb_3"):
                        st.session_state["nav_page"] = "Knowledge Base"
                        st.rerun()
                elif i in (4, 5, 6):
                    if st.button("Open AI Investigation Workspace ➔", key=f"journey_btn_ai_{i}"):
                        st.session_state["nav_page"] = "AI Investigation"
                        st.session_state["selected_investigation_incident_id"] = selected_id
                        st.rerun()
                elif i == 7:
                    if st.button("Inspect Governance Audit Ledger in Incident Center ➔", key="journey_btn_audit_7"):
                        st.session_state["nav_page"] = "Incident Center"
                        st.rerun()

elif page == "Overview":
    st.markdown('<div class="eyebrow">OPERATIONS / DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Operations Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Monitor incidents, understand business impact and make evidence-based decisions.</div>',
        unsafe_allow_html=True
    )

    # Live dashboard metrics fetched from the database service layer
    try:
        stats = incident_service.get_incident_stats()
    except Exception as err:
        st.error(f"Unable to load live incident statistics: {err}")
        stats = {"total": 0, "open": 0, "critical": 0, "resolved": 0}

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("TOTAL INCIDENTS", str(stats.get("total", 0)), "All reported incidents"),
        ("OPEN INCIDENTS", str(stats.get("open", 0)), "Awaiting investigation"),
        ("CRITICAL INCIDENTS", str(stats.get("critical", 0)), "Require urgent attention"),
        ("RESOLVED INCIDENTS", str(stats.get("resolved", 0)), "Successfully closed")
    ]

    for col, metric in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{_safe_html_text(metric[0])}</div>
                <div class="metric-value">{_safe_html_text(metric[1])}</div>
                <div class="metric-note">{_safe_html_text(metric[2])}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Incident Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Capture a business incident to begin an AI-assisted investigation.</div>',
        unsafe_allow_html=True
    )

    with st.form("incident_form"):
        title = st.text_input(
            "Incident title",
            placeholder="e.g. Payment deducted but order not confirmed"
        )

        description = st.text_area(
            "Incident description",
            placeholder="Describe what happened, when it happened and who is affected...",
            height=120
        )

        col1, col2 = st.columns(2)

        with col1:
            department = st.selectbox(
                "Affected department",
                ["Not sure", "IT & Security", "Payments & Billing", "Customer Operations", "Engineering & DevOps", "Data Platform", "General Operations", "Other"]
            )

        with col2:
            impact = st.selectbox(
                "Business impact",
                ["Not sure", "Low", "Medium", "High", "Critical"]
            )

        submitted = st.form_submit_button("＋ Submit Incident", use_container_width=True)

        if submitted:
            try:
                created = incident_service.create_incident(
                    title=title,
                    description=description,
                    department=department,
                    impact=impact,
                    actor="Employee",
                )
                st.session_state["last_created_incident"] = created
                st.rerun()
            except ValueError as ve:
                st.error(f"Validation Error: {ve}")
            except Exception as err:
                st.error(f"Failed to record incident: {err}")

    if "last_created_incident" in st.session_state:
        incident = st.session_state["last_created_incident"]

        st.markdown('<div class="section-heading">Recently Logged Incident</div>', unsafe_allow_html=True)
        st.success(
            f"✅ **Incident {incident['formatted_id']} recorded successfully.**\n\n"
            f"**{incident['title']}**\n\n"
            f"{incident['description']}\n\n"
            f"Department: {incident['department']}  |  Impact: {incident['impact']}  |  Status: {incident['status']}"
        )

elif page == "Incident Center":
    st.markdown('<div class="eyebrow">INCIDENTS / WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Incident Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Search, filter, and inspect detailed incident records, investigations, and audit trails.</div>',
        unsafe_allow_html=True
    )

    try:
        all_incidents = incident_service.get_all_incidents()
    except Exception as err:
        st.error(f"Unable to load incidents: {err}")
        all_incidents = []

    if not all_incidents:
        st.info("No incidents have been logged yet. Capture a new incident from the **Overview** dashboard to begin.")
    else:
        # Search and Filter Controls
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1, 1, 1])

        with f_col1:
            search_query = st.text_input("Search", placeholder="Search by incident ID or title...", label_visibility="collapsed")
        with f_col2:
            status_filter = st.selectbox("Status", ["All Statuses"] + INCIDENT_STATUSES, label_visibility="collapsed")
        with f_col3:
            dept_choices = ["All Departments"] + sorted(list(set(DEPARTMENTS) | {"Other", "Not sure"}))
            dept_filter = st.selectbox("Department", dept_choices, label_visibility="collapsed")
        with f_col4:
            sev_filter = st.selectbox("Severity", ["All Severities"] + SEVERITY_LEVELS, label_visibility="collapsed")

        # Apply search and filters in Python (no direct SQL in UI)
        filtered = all_incidents

        if search_query.strip():
            q = search_query.strip().lower()
            filtered = [
                i for i in filtered
                if q in i.get("formatted_id", "").lower() or q in i.get("title", "").lower()
            ]

        if status_filter != "All Statuses":
            filtered = [i for i in filtered if i.get("status") == status_filter]

        if dept_filter != "All Departments":
            filtered = [i for i in filtered if i.get("department") == dept_filter]

        if sev_filter != "All Severities":
            filtered = [i for i in filtered if i.get("severity") == sev_filter]

        # Incident Count Badge
        st.caption(f"Showing {len(filtered)} of {len(all_incidents)} total incidents")

        if not filtered:
            st.warning("No incidents match the selected search and filter criteria.")
        else:
            # Display Clean Tabular Listing
            table_rows = [
                {
                    "Incident ID": item.get("formatted_id", "N/A"),
                    "Title": item.get("title", "Untitled"),
                    "Department": item.get("department") or "Unassigned",
                    "Impact": item.get("impact") or "Not set",
                    "Severity": item.get("severity") or "Pending AI",
                    "Status": item.get("status", "Open"),
                    "Created": helpers.format_timestamp(item.get("created_at")),
                }
                for item in filtered
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

            # Detail Inspector Section
            st.markdown('<div class="section-heading">Incident Detail Inspector</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="subtitle">Select an incident to view full symptoms, diagnostic metadata, investigation history, and governance audit trail.</div>',
                unsafe_allow_html=True
            )

            option_labels = [
                f"{item.get('formatted_id')} • {item.get('title')}"
                for item in filtered
            ]
            selected_label = st.selectbox(
                "Select Incident to Inspect",
                options=option_labels,
                label_visibility="collapsed"
            )

            selected_idx = option_labels.index(selected_label)
            selected_incident_meta = filtered[selected_idx]
            selected_id = selected_incident_meta["id"]

            try:
                selected_incident = incident_service.get_incident_by_id(selected_id)
            except Exception as err:
                st.error(f"Error fetching incident details: {err}")
                selected_incident = selected_incident_meta

            if selected_incident:
                # Persistent Flash Notification (survives rerun for governance decisions)
                if "classification_flash" in st.session_state:
                    flash = st.session_state.pop("classification_flash", None)
                    if flash and isinstance(flash, dict):
                        flash_type = flash.get("type", "info")
                        flash_msg = flash.get("msg", "")
                        if flash_type == "success":
                            st.success(flash_msg)
                        elif flash_type == "info":
                            st.info(flash_msg)
                        elif flash_type == "warning":
                            st.warning(flash_msg)
                        elif flash_type == "error":
                            st.error(flash_msg)

                # Detail Overview Card
                esc_id = _safe_html_text(selected_incident.get("formatted_id", "N/A"))
                esc_status = _safe_html_text(str(selected_incident.get("status", "Open")).upper())
                esc_created = _safe_html_text(helpers.format_timestamp(selected_incident.get("created_at")))
                esc_title = _safe_html_text(selected_incident.get("title", ""))
                esc_desc = _safe_html_text(selected_incident.get("description", ""), preserve_newlines=True)

                st.markdown(_clean_html_block(f"""
                <div class="metric-card" style="margin-top: 10px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="eyebrow">{esc_id} • STATUS: {esc_status}</span>
                        <span class="metric-note">Reported: {esc_created}</span>
                    </div>
                    <div class="main-title" style="font-size: 22px; margin-top: 6px;">{esc_title}</div>
                    <div style="color: #4A5568; font-size: 14px; margin-top: 10px; line-height: 1.6; word-break: break-word;">
                        {esc_desc}
                    </div>
                </div>
                """), unsafe_allow_html=True)

                # Metadata Cards
                m1, m2, m3, m4 = st.columns(4)
                esc_dept = _safe_html_text(selected_incident.get("department") or "Unassigned")
                esc_impact = _safe_html_text(selected_incident.get("impact") or "Not set")
                esc_cat = _safe_html_text(selected_incident.get("category") or "Pending AI Classification")
                esc_sev = _safe_html_text(selected_incident.get("severity") or "Pending AI Assessment")

                with m1:
                    st.markdown(_clean_html_block(f"""
                    <div class="metric-card" style="padding: 14px;">
                        <div class="metric-label">DEPARTMENT</div>
                        <div style="font-weight: 700; color: #14213D; font-size: 15px; margin-top: 4px;">
                            {esc_dept}
                        </div>
                    </div>
                    """), unsafe_allow_html=True)
                with m2:
                    st.markdown(_clean_html_block(f"""
                    <div class="metric-card" style="padding: 14px;">
                        <div class="metric-label">BUSINESS IMPACT</div>
                        <div style="font-weight: 700; color: #14213D; font-size: 15px; margin-top: 4px;">
                            {esc_impact}
                        </div>
                    </div>
                    """), unsafe_allow_html=True)
                with m3:
                    st.markdown(_clean_html_block(f"""
                    <div class="metric-card" style="padding: 14px;">
                        <div class="metric-label">AI CATEGORY</div>
                        <div style="font-weight: 700; color: #14213D; font-size: 15px; margin-top: 4px;">
                            {esc_cat}
                        </div>
                    </div>
                    """), unsafe_allow_html=True)
                with m4:
                    st.markdown(_clean_html_block(f"""
                    <div class="metric-card" style="padding: 14px;">
                        <div class="metric-label">SEVERITY LEVEL</div>
                        <div style="font-weight: 700; color: #14213D; font-size: 15px; margin-top: 4px;">
                            {esc_sev}
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

                # --------------------------------------------------------------
                # AI Classification & Human Governance Section
                # --------------------------------------------------------------
                proposal_key = f"ai_proposal_{selected_id}"
                has_proposal = proposal_key in st.session_state
                is_classified = bool(selected_incident.get("category") and selected_incident.get("severity"))

                st.markdown('<div class="section-heading" style="font-size: 16px; margin-top: 18px; margin-bottom: 8px;">AI Classification & Human Governance</div>', unsafe_allow_html=True)

                def _execute_ai_classification(spinner_msg: str = "Analyzing incident symptoms with Gemini AI..."):
                    try:
                        with st.spinner(spinner_msg):
                            proposal_result = ai_service.classify_incident(
                                title=selected_incident.get("title", ""),
                                description=selected_incident.get("description", ""),
                                timeout=45,
                            )
                        st.session_state[proposal_key] = proposal_result
                        st.rerun()
                    except ai_service.GeminiNotConfiguredError:
                        st.error("⚠️ Gemini API is not configured. Please set GEMINI_API_KEY in the root `.env` file.")
                    except ai_service.GeminiNetworkError as ne:
                        st.error(f"🌐 AI Network Error: {ne}. The incident was not modified.")
                    except ai_service.GeminiAPIError as ae:
                        st.error(f"⚠️ Gemini API Error (HTTP {ae.status_code}): {ae}. The incident was not modified.")
                    except ai_service.GeminiParseError as pe:
                        st.error(f"⚠️ AI Formatting Error: {pe}. The incident was not modified.")
                    except ai_service.GeminiValidationError as ve:
                        st.error(f"⚠️ AI Validation Error: {ve}. The incident was not modified.")
                    except Exception as e:
                        st.error(f"⚠️ Unexpected Error during classification: {e}. The incident was not modified.")

                # P2: Context-Aware Trigger - only shown when no proposal is staged
                if not has_proposal:
                    action_col, note_col = st.columns([1, 3])
                    with action_col:
                        btn_label = "🤖 Re-classify with AI" if is_classified else "🤖 Classify with AI"
                        if st.button(btn_label, key=f"btn_classify_{selected_id}", use_container_width=True):
                            _execute_ai_classification("Analyzing incident symptoms with Gemini AI...")

                    with note_col:
                        if is_classified:
                            st.caption("Re-evaluates symptoms using Gemini AI. Existing classification will not be changed until an operator explicitly approves the proposal.")
                        else:
                            st.caption("Analyze symptoms using Gemini AI to propose category, severity, department, and business impact for human review.")

                # If proposal is staged, render the Unified Governance Panel (P3)
                if has_proposal:
                    proposal = st.session_state[proposal_key]

                    esc_prop_cat = _safe_html_text(proposal.get("category", "N/A"))
                    esc_prop_sev = _safe_html_text(proposal.get("severity", "N/A"))
                    esc_prop_dept = _safe_html_text(proposal.get("department", "N/A"))
                    esc_prop_imp = _safe_html_text(proposal.get("impact", "N/A"))
                    esc_prop_reason = _safe_html_text(proposal.get("reasoning", ""), preserve_newlines=True)

                    esc_curr_cat = _safe_html_text(selected_incident.get("category") or "Pending AI Classification")
                    esc_curr_sev = _safe_html_text(selected_incident.get("severity") or "Pending AI Assessment")
                    esc_curr_dept = _safe_html_text(selected_incident.get("department") or "Unassigned")
                    esc_curr_imp = _safe_html_text(selected_incident.get("impact") or "Not set")

                    raw_conf = proposal.get("confidence", 0.0)
                    conf_pct = f"{raw_conf * 100:.0f}" if isinstance(raw_conf, (int, float)) else "N/A"

                    cat_match = selected_incident.get("category") == proposal.get("category")
                    sev_match = selected_incident.get("severity") == proposal.get("severity")
                    dept_match = selected_incident.get("department") == proposal.get("department")
                    imp_match = selected_incident.get("impact") == proposal.get("impact")

                    def _render_badge(matches: bool) -> str:
                        if matches:
                            return '<span style="color: #64748B; font-size: 11px; font-weight: 600;">MATCHES</span>'
                        return '<span style="color: #315BEA; font-size: 11px; font-weight: 700; background: #EEF2FF; padding: 2px 7px; border-radius: 4px;">UPDATE PROPOSED</span>'

                    # P3: Unified Governance Panel Container
                    with st.container(border=True):
                        # Governance Panel Header with compact re-run button (P2)
                        hdr_col1, hdr_col2, hdr_col3 = st.columns([5, 3, 2])
                        with hdr_col1:
                            st.markdown(
                                _clean_html_block("""
                                <span class="eyebrow" style="color: #315BEA;">AI CLASSIFICATION PROPOSAL • AWAITING HUMAN REVIEW</span>
                                <div style="font-weight: 800; font-size: 17px; color: #14213D; margin-top: 2px;">Comparison &amp; Decision Staging</div>
                                """),
                                unsafe_allow_html=True,
                            )
                        with hdr_col2:
                            st.markdown(
                                _clean_html_block(f"""
                                <div style="text-align: right; padding-top: 2px;">
                                    <span style="font-size: 11px; font-weight: 700; color: #64748B;">MODEL CERTAINTY:</span>
                                    <span style="background: #E2E8F0; color: #1E293B; padding: 2px 8px; border-radius: 10px; font-weight: 700; font-size: 12px; margin-left: 4px;">{conf_pct}%</span>
                                    <div style="font-size: 10px; color: #8290A5; margin-top: 2px;">(Uncalibrated model self-estimate)</div>
                                </div>
                                """),
                                unsafe_allow_html=True,
                            )
                        with hdr_col3:
                            if st.button("🔄 Re-run Analysis", key=f"btn_rerun_{selected_id}", use_container_width=True, help="Re-query Gemini AI to produce a fresh classification proposal"):
                                _execute_ai_classification("Re-analyzing incident symptoms with Gemini AI...")

                        # Side-by-Side Comparison Table
                        st.markdown(_clean_html_block(f"""
                        <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px;">
                            <thead>
                                <tr style="border-bottom: 2px solid #E7ECF3; text-align: left; color: #718096; font-size: 11px; letter-spacing: 0.5px;">
                                    <th style="padding: 6px 10px;">ATTRIBUTE</th>
                                    <th style="padding: 6px 10px;">CURRENT VALUE</th>
                                    <th style="padding: 6px 10px;">AI PROPOSAL</th>
                                    <th style="padding: 6px 10px;">STATUS</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="border-bottom: 1px solid #F1F5F9;">
                                    <td style="padding: 8px 10px; font-weight: 600; color: #64748B;">Category</td>
                                    <td style="padding: 8px 10px; color: #14213D;">{esc_curr_cat}</td>
                                    <td style="padding: 8px 10px; font-weight: 700; color: #14213D;">{esc_prop_cat}</td>
                                    <td style="padding: 8px 10px;">{_render_badge(cat_match)}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #F1F5F9;">
                                    <td style="padding: 8px 10px; font-weight: 600; color: #64748B;">Severity</td>
                                    <td style="padding: 8px 10px; color: #14213D;">{esc_curr_sev}</td>
                                    <td style="padding: 8px 10px; font-weight: 700; color: #14213D;">{esc_prop_sev}</td>
                                    <td style="padding: 8px 10px;">{_render_badge(sev_match)}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #F1F5F9;">
                                    <td style="padding: 8px 10px; font-weight: 600; color: #64748B;">Department</td>
                                    <td style="padding: 8px 10px; color: #14213D;">{esc_curr_dept}</td>
                                    <td style="padding: 8px 10px; font-weight: 700; color: #14213D;">{esc_prop_dept}</td>
                                    <td style="padding: 8px 10px;">{_render_badge(dept_match)}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 10px; font-weight: 600; color: #64748B;">Business Impact</td>
                                    <td style="padding: 8px 10px; color: #14213D;">{esc_curr_imp}</td>
                                    <td style="padding: 8px 10px; font-weight: 700; color: #14213D;">{esc_prop_imp}</td>
                                    <td style="padding: 8px 10px;">{_render_badge(imp_match)}</td>
                                </tr>
                            </tbody>
                        </table>

                        <div style="margin-top: 14px; margin-bottom: 12px; padding: 12px 16px; background: #F8FAFC; border-left: 3px solid #315BEA; border-radius: 6px;">
                            <div style="font-weight: 700; color: #14213D; font-size: 12px; margin-bottom: 4px; letter-spacing: 0.5px;">EVIDENCE-GROUNDED REASONING</div>
                            <div style="color: #4A5568; font-size: 13.5px; line-height: 1.5; word-break: break-word;">{esc_prop_reason}</div>
                        </div>
                        """), unsafe_allow_html=True)

                        # Human Review Flag Warning
                        if proposal.get("requires_human_review"):
                            st.warning("⚠️ **Human Review Flagged by AI**: The model detected partial, ambiguous, or borderline diagnostic facts. Please verify evidence carefully before approving.")

                        # P2: Missing Information List with Diagnostic Transparency (auto-expanded if requires_human_review)
                        missing_info = proposal.get("missing_information", [])
                        if missing_info:
                            auto_expand = bool(proposal.get("requires_human_review"))
                            with st.expander("🔍 Diagnostic Information Gaps Flagged by AI", expanded=auto_expand):
                                for item in missing_info:
                                    st.markdown(f"• {_safe_html_text(item)}")

                        # P1: Decision Controls Alignment & Optional Rejection Notes Expander
                        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #14213D; margin-top: 14px; margin-bottom: 6px;'>OPERATOR GOVERNANCE DECISION</div>", unsafe_allow_html=True)

                        with st.expander("📝 Optional rejection notes / reasoning", expanded=False):
                            rejection_notes = st.text_input(
                                "Rejection reason",
                                placeholder="Specify justification for rejecting this proposal...",
                                key=f"rej_notes_{selected_id}",
                                label_visibility="collapsed",
                            )

                        dec_col1, dec_col2 = st.columns([1, 1])

                        with dec_col1:
                            if st.button("✅ Approve & Save Classification", key=f"btn_approve_{selected_id}", use_container_width=True):
                                try:
                                    summary_note = f"Approved AI classification: category='{proposal['category']}', severity='{proposal['severity']}', department='{proposal['department']}', impact='{proposal['impact']}'."
                                    success = incident_service.update_incident_classification(
                                        incident_id=selected_id,
                                        category=proposal["category"],
                                        severity=proposal["severity"],
                                        department=proposal["department"],
                                        impact=proposal["impact"],
                                        actor="Operator",
                                        notes=summary_note,
                                    )
                                    if success:
                                        st.session_state["classification_flash"] = {
                                            "type": "success",
                                            "msg": f"✅ Classification approved and saved for {esc_id}: Category '{proposal['category']}', Severity '{proposal['severity']}', Department '{proposal['department']}'.",
                                        }
                                        st.session_state.pop(proposal_key, None)
                                        try:
                                            st.toast("✅ Classification approved and saved.")
                                        except Exception:
                                            pass
                                        st.rerun()
                                    else:
                                        st.error("Failed to save approved classification.")
                                except Exception as err:
                                    st.error(f"Error saving classification: {err}")

                        with dec_col2:
                            if st.button("❌ Reject Proposal", key=f"btn_reject_{selected_id}", use_container_width=True):
                                try:
                                    clean_reason = helpers.clean_text(rejection_notes) if hasattr(helpers, 'clean_text') else rejection_notes.strip()
                                    success = incident_service.reject_incident_classification(
                                        incident_id=selected_id,
                                        proposed_category=proposal.get("category"),
                                        proposed_severity=proposal.get("severity"),
                                        proposed_department=proposal.get("department"),
                                        proposed_impact=proposal.get("impact"),
                                        actor="Operator",
                                        reason=clean_reason or "Operator rejected proposed AI classification.",
                                    )
                                    if success:
                                        st.session_state["classification_flash"] = {
                                            "type": "info",
                                            "msg": f"ℹ️ AI classification proposal for {esc_id} was rejected. Incident record was not modified.",
                                        }
                                        st.session_state.pop(proposal_key, None)
                                        try:
                                            st.toast("AI classification proposal rejected.")
                                        except Exception:
                                            pass
                                        st.rerun()
                                    else:
                                        st.error("Failed to record rejection.")
                                except Exception as err:
                                    st.error(f"Error recording rejection: {err}")

                st.markdown("<br>", unsafe_allow_html=True)

                # Tabs for Investigation History and Audit History
                tab_inv, tab_audit = st.tabs(["🔍 Investigation History", "📜 Governance Audit Trail"])

                with tab_inv:
                    try:
                        investigations_raw = incident_service.get_investigations_for_incident(selected_id)
                    except Exception as err:
                        st.error(f"Error fetching investigations: {err}")
                        investigations_raw = []

                    if investigations_raw:
                        btn_c1, btn_c2 = st.columns([1, 2])
                        with btn_c1:
                            if st.button("🔬 Open in AI Investigation Workspace", key=f"btn_nav_ai_inv_{selected_id}", use_container_width=True):
                                st.session_state["nav_page"] = "AI Investigation"
                                st.session_state["selected_investigation_incident_id"] = selected_id
                                st.rerun()

                        for raw_inv in investigations_raw:
                            inv = incident_service.deserialize_investigation(raw_inv)
                            inv_id = _safe_html_text(inv.get("id", "N/A"))
                            inv_time = _safe_html_text(helpers.format_timestamp(inv.get("created_at")))
                            conf_val = inv.get("confidence")
                            is_inv_fallback = (
                                inv.get("is_fallback") is True
                                or inv.get("mode") == "evidence_only_fallback"
                                or conf_val is None
                                or "Automated AI diagnostic synthesis is currently unavailable" in (inv.get("ai_recommendation") or "")
                            )
                            inv_rec_raw = inv.get("ai_recommendation") or "N/A"
                            inv_cites = inv.get("evidence_citations") or []
                            if is_inv_fallback and not inv_cites:
                                inv_rec_raw = _format_fallback_guidance(inv_rec_raw, citations=inv_cites, total_docs=0)
                            inv_rec = _safe_html_text(inv_rec_raw, preserve_newlines=True)
                            if is_inv_fallback or conf_val is None:
                                inv_conf = "N/A — Model Offline"
                            else:
                                inv_conf = f"{conf_val:.0%} (Model-Reported)"
                            inv_decision = _safe_html_text(inv.get("human_decision") or "Pending")
                            final_act = _format_remediation_html(inv.get("final_action") or "")

                            decision_badge = '<span style="background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">PENDING APPROVAL</span>'
                            if inv.get("human_decision") == "Approved":
                                decision_badge = '<span style="background: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">APPROVED</span>'
                            elif inv.get("human_decision") == "Rejected":
                                decision_badge = '<span style="background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">REJECTED</span>'

                            final_act_html = f'<div style="margin-top: 8px; padding: 8px 12px; background: #F0FDF4; border-left: 3px solid #16A34A; border-radius: 4px; font-size: 13px;"><strong>Approved Action:</strong><div style="margin-top: 4px; line-height: 1.5;">{final_act}</div></div>' if final_act else ''

                            st.markdown(_clean_html_block(f"""
                            <div class="metric-card" style="margin-bottom: 12px; padding: 16px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div><strong>Investigation #{inv_id}</strong> <span style="margin: 0 6px;"></span> {decision_badge}</div>
                                    <span class="metric-note">{inv_time}</span>
                                </div>
                                <div style="margin-top: 8px; font-size: 13.5px; line-height: 1.5;">
                                    <strong style="margin-right: 4px;">AI Recommendation:</strong>{inv_rec}
                                </div>
                                {final_act_html}
                                <div style="margin-top: 8px; color: #64748B; font-size: 12px;">
                                    <span style="margin-right: 4px;">Model Confidence:</span><strong>{inv_conf}</strong> <span style="margin: 0 6px; color: #CBD5E1;">|</span> <span style="margin-right: 4px;">Citations:</span><strong>{len(inv.get('evidence_citations', []))}</strong> <span style="margin: 0 6px; color: #CBD5E1;">|</span> <span style="margin-right: 4px;">Hypotheses:</span><strong>{len(inv.get('root_cause_hypotheses', []))}</strong>
                                </div>
                            </div>
                            """), unsafe_allow_html=True)
                    else:
                        st.info("No investigations have been conducted on this incident yet.")
                        if st.button("🔬 Launch AI Investigation", key=f"btn_start_ai_inv_{selected_id}"):
                            st.session_state["nav_page"] = "AI Investigation"
                            st.session_state["selected_investigation_incident_id"] = selected_id
                            st.rerun()

                with tab_audit:
                    try:
                        audit_logs = incident_service.get_audit_history_for_incident(selected_id)
                    except Exception as err:
                        st.error(f"Error fetching audit trail: {err}")
                        audit_logs = []

                    if audit_logs:
                        latest_log = audit_logs[-1]
                        latest_act = latest_log.get("action", "")
                        latest_actor = latest_log.get("actor", "Operator")
                        latest_ts = helpers.format_timestamp(latest_log.get("created_at"))
                        if latest_act == "GOVERNANCE_AUDIT_CORRECTION" or "CORRECTION" in latest_act:
                            st.caption(f"⚖️ **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")
                        elif "APPROVED" in latest_act:
                            st.caption(f"🔒 **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")
                        elif "REJECTED" in latest_act:
                            st.caption(f"⚠️ **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")
                        elif "EDITED" in latest_act:
                            st.caption(f"✏️ **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")

                        audit_records = [
                            {
                                "Timestamp": helpers.format_timestamp(entry.get("created_at")),
                                "Action": entry.get("action", ""),
                                "Actor": entry.get("actor", "System"),
                                "Details": entry.get("notes") or "",
                            }
                            for entry in audit_logs
                        ]
                        st.dataframe(audit_records, use_container_width=True, hide_index=True)
                    else:
                        st.info("No audit logs recorded for this incident.")


elif page == "AI Investigation":
    st.markdown('<div class="eyebrow">DIAGNOSTICS / AI INVESTIGATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">AI Investigation Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Evidence-grounded root-cause analysis, runbook retrieval, and supervised remediation governance.</div>',
        unsafe_allow_html=True
    )

    try:
        all_incidents = incident_service.get_all_incidents()
    except Exception as err:
        st.error(f"Error loading incidents: {err}")
        all_incidents = []

    if not all_incidents:
        st.info("No incidents have been logged yet. Capture an incident from the **Overview** dashboard to begin an AI investigation.")
    else:
        # Pre-selection handling
        inc_options = [
            f"{inc.get('formatted_id')} • {inc.get('title')}"
            for inc in all_incidents
        ]
        target_id = st.session_state.pop("selected_investigation_incident_id", None)
        default_idx = 0
        if target_id:
            for idx, inc in enumerate(all_incidents):
                if inc["id"] == target_id:
                    default_idx = idx
                    st.session_state["investigation_incident_selector"] = inc_options[idx]
                    break

        sel_col1, sel_col2 = st.columns([3, 1])
        with sel_col1:
            if "investigation_incident_selector" in st.session_state:
                selected_label = st.selectbox(
                    "Select Incident to Investigate",
                    options=inc_options,
                    key="investigation_incident_selector",
                )
            else:
                selected_label = st.selectbox(
                    "Select Incident to Investigate",
                    options=inc_options,
                    index=0,
                    key="investigation_incident_selector",
                )

        selected_idx = inc_options.index(selected_label)
        selected_incident_meta = all_incidents[selected_idx]
        selected_id = selected_incident_meta["id"]

        try:
            selected_incident = incident_service.get_incident_by_id(selected_id) or selected_incident_meta
        except Exception as err:
            st.error(f"Error fetching incident: {err}")
            selected_incident = selected_incident_meta

        # Flash Notification
        if "investigation_flash" in st.session_state:
            flash = st.session_state.pop("investigation_flash", None)
            if flash and isinstance(flash, dict):
                f_type = flash.get("type", "info")
                f_msg = flash.get("msg", "")
                if f_type == "success":
                    st.success(f_msg)
                elif f_type == "info":
                    st.info(f_msg)
                elif f_type == "warning":
                    st.warning(f_msg)
                elif f_type == "error":
                    st.error(f_msg)

        # Context Card
        esc_id = _safe_html_text(selected_incident.get("formatted_id", "N/A"))
        esc_title = _safe_html_text(selected_incident.get("title", ""))
        esc_desc = _safe_html_text(selected_incident.get("description", ""), preserve_newlines=True)
        esc_status = _safe_html_text(str(selected_incident.get("status", "Open")).upper())
        esc_dept = _safe_html_text(selected_incident.get("department") or "Unassigned")
        esc_sev = _safe_html_text(selected_incident.get("severity") or "Pending Assessment")
        esc_imp = _safe_html_text(selected_incident.get("impact") or "Not set")
        esc_created = _safe_html_text(helpers.format_timestamp(selected_incident.get("created_at")))

        st.markdown(_clean_html_block(f"""
        <div class="metric-card" style="margin-top: 10px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="eyebrow">{esc_id} • STATUS: {esc_status}</span>
                <span class="metric-note">Reported: {esc_created}</span>
            </div>
            <div class="main-title" style="font-size: 20px; margin-top: 6px;">{esc_title}</div>
            <div style="color: #4A5568; font-size: 13.5px; margin-top: 8px; line-height: 1.5; word-break: break-word;">
                {esc_desc}
            </div>
            <div style="margin-top: 12px; display: flex; gap: 20px; font-size: 12px; color: #64748B;">
                <div><span style="color: #64748B; margin-right: 4px;">Department:</span><strong style="color: #14213D;">{esc_dept}</strong></div>
                <div><span style="color: #64748B; margin-right: 4px;">Severity:</span><strong style="color: #14213D;">{esc_sev}</strong></div>
                <div><span style="color: #64748B; margin-right: 4px;">Impact:</span><strong style="color: #14213D;">{esc_imp}</strong></div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Fetch investigations for this incident
        try:
            existing_inv_rows = incident_service.get_investigations_for_incident(selected_id)
        except Exception:
            existing_inv_rows = []

        active_inv = None
        if existing_inv_rows:
            active_inv = incident_service.deserialize_investigation(existing_inv_rows[0])

        def _execute_investigation():
            try:
                with st.spinner("Retrieving knowledge runbooks and conducting AI root-cause investigation with Gemini..."):
                    # Strict refinement 1: default timeout is 45s
                    inv_data = ai_service.investigate_incident(
                        incident_data=selected_incident,
                        timeout=45,
                    )

                is_new_fallback = (
                    inv_data.get("is_fallback") is True
                    or inv_data.get("mode") == "evidence_only_fallback"
                    or "Automated AI diagnostic synthesis is currently unavailable" in (inv_data.get("ai_recommendation") or "")
                )

                # Strengthened duplicate prevention:
                # Suppress saving a duplicate proposal and audit log ONLY when:
                # active_inv exists for this incident, is still 'Pending', and is already a fallback.
                if (
                    is_new_fallback
                    and active_inv
                    and active_inv.get("incident_id") == selected_id
                    and active_inv.get("human_decision") == "Pending"
                    and (
                        active_inv.get("is_fallback") is True
                        or active_inv.get("mode") == "evidence_only_fallback"
                        or "Automated AI diagnostic synthesis is currently unavailable" in (active_inv.get("ai_recommendation") or "")
                    )
                ):
                    reason = inv_data.get("fallback_reason") or "provider unavailable"
                    st.session_state["investigation_flash"] = {
                        "type": "warning",
                        "msg": f"⚠️ Automated AI synthesis remains unavailable ({reason}). Existing unreviewed fallback proposal #{active_inv['id']} was retained without creating duplicate records or audit logs.",
                    }
                    st.rerun()
                    return

                new_id = incident_service.save_investigation_proposal(
                    incident_id=selected_id,
                    investigation_data=inv_data,
                    evidence_packets=inv_data.get("retrieved_evidence"),
                    actor="AI Diagnostic Engine",
                )

                if is_new_fallback:
                    reason = inv_data.get("fallback_reason") or "provider unavailable"
                    st.session_state["investigation_flash"] = {
                        "type": "warning",
                        "msg": f"⚠️ Evidence retrieved, but automated AI synthesis is unavailable ({reason}). Diagnostic fallback proposal #{new_id} recorded for operator review.",
                    }
                else:
                    st.session_state["investigation_flash"] = {
                        "type": "success",
                        "msg": f"✅ AI Investigation proposal #{new_id} recorded and awaiting operator review.",
                    }
                st.rerun()
            except ai_service.GeminiNotConfiguredError:
                st.error("⚠️ Gemini API is not configured. Please set GEMINI_API_KEY in the `.env` file.")
            except ai_service.GeminiNetworkError as ne:
                st.error(f"🌐 AI Network Error: {ne}. No changes were made.")
            except ai_service.GeminiAPIError as ae:
                st.error(f"⚠️ Gemini API Error (HTTP {ae.status_code}): {ae}.")
            except ai_service.GeminiParseError as pe:
                st.error(f"⚠️ AI Formatting Error: {pe}. The model output could not be parsed as valid diagnostic JSON.")
            except ai_service.GeminiValidationError as ve:
                st.error(f"⚠️ AI Diagnostic Validation Error: {ve}.")
            except Exception as e:
                st.error(f"⚠️ Error during investigation: {e}")

        # Action / Status Bar
        if not active_inv:
            st.markdown("""
            <div class="metric-card" style="text-align: center; padding: 30px; margin-bottom: 24px;">
                <div style="font-size: 28px; margin-bottom: 8px;">🔬</div>
                <div style="font-weight: 750; font-size: 17px; color: #14213D;">No Investigation Conducted Yet</div>
                <div style="color: #64748B; font-size: 13.5px; margin-top: 4px; margin-bottom: 18px;">
                    Retrieve relevant enterprise runbooks from the knowledge base and perform an evidence-grounded root cause analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🤖 Run AI Investigation", key=f"btn_run_inv_{selected_id}", use_container_width=True):
                _execute_investigation()
        else:
            # Active investigation status strip
            inv_id = active_inv["id"]
            decision = active_inv.get("human_decision", "Pending")
            is_fallback = (
                active_inv.get("is_fallback") is True
                or active_inv.get("mode") == "evidence_only_fallback"
                or "Automated AI diagnostic synthesis is currently unavailable" in (active_inv.get("ai_recommendation") or "")
            )
            conf_val = active_inv.get("confidence")
            if is_fallback or conf_val is None:
                conf_display = "N/A — Model Offline"
                conf_subtext = "No automated AI analysis generated"
            else:
                conf_display = f"{conf_val * 100:.0f}%"
                conf_subtext = "Model-reported self-assessment — not an independently calibrated probability"
            created_ts = _safe_html_text(helpers.format_timestamp(active_inv.get("created_at")))

            badge_html = '<span style="background: #FEF3C7; color: #92400E; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 12px;">PENDING OPERATOR REVIEW</span>'
            if decision == "Approved":
                badge_html = '<span style="background: #DCFCE7; color: #166534; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 12px;">APPROVED</span>'
            elif decision == "Rejected":
                badge_html = '<span style="background: #FEE2E2; color: #991B1B; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 12px;">REJECTED</span>'

            hdr_c1, hdr_c2 = st.columns([3, 1])
            with hdr_c1:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <span style="font-size: 18px; font-weight: 800; color: #14213D;">Investigation Proposal #{inv_id}</span>
                    {badge_html}
                    <span class="metric-note">Generated: {created_ts}</span>
                </div>
                """, unsafe_allow_html=True)
            with hdr_c2:
                if st.button("🔄 Re-run AI Investigation", key=f"btn_rerun_inv_{selected_id}", use_container_width=True, help="Re-queries Gemini AI and retrieves fresh knowledge chunks"):
                    _execute_investigation()

            has_retrieved_runbooks = bool(
                active_inv.get("provenance")
                or active_inv.get("evidence_citations")
                or active_inv.get("retrieved_evidence")
            )

            if is_fallback:
                err_type = active_inv.get("error_type") or "general"
                category_titles = {
                    "missing_configuration": "Gemini API Configuration Missing (GEMINI_API_KEY not configured)",
                    "authentication_error": "Gemini API Authentication Failed (HTTP 401 — Invalid Credentials)",
                    "permission_denied": "Gemini API Permission Denied (HTTP 403 — Insufficient Permissions)",
                    "timeout": "Gemini API Request Timed Out (>45s)",
                    "connection_error": "Network Connection Failure to Gemini API",
                    "transport_error": "Transport Failure During Gemini API Call",
                    "provider_unavailable": "Gemini AI Service Temporarily Unavailable",
                }
                err_title = category_titles.get(err_type, "Automated AI Diagnostic Synthesis Unavailable")
                err_desc = active_inv.get("fallback_reason") or "Automated AI diagnostic synthesis is currently unavailable."
                if not has_retrieved_runbooks:
                    offline_rec_action = "No runbooks are indexed in the Knowledge Base. Inspect live system logs and author an operator remediation procedure in the Governance Console."
                else:
                    offline_rec_action = "Review the retrieved runbook guidance below and author an operator remediation procedure in the Governance Console."

                st.markdown(_clean_html_block(f"""
                <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">⚠️</span>
                        <div>
                            <div style="font-weight: 800; font-size: 14px; color: #92400E;">OFFLINE DIAGNOSTIC MODE: {err_title}</div>
                            <div style="font-size: 13px; color: #78350F; margin-top: 3px; line-height: 1.4;">{err_desc}</div>
                            <div style="font-size: 12px; color: #B45309; margin-top: 6px; font-weight: 600;">
                                <span style="margin-right: 4px;">Recommended Action:</span>{offline_rec_action}
                            </div>
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

            # TWO-COLUMN WORKSPACE: Findings (Left) & Evidence Explorer (Right)
            col_findings, col_evidence = st.columns([1.6, 1.0], gap="large")

            with col_findings:
                st.markdown('<div class="section-heading" style="font-size: 17px; margin-top: 0; margin-bottom: 8px;">Diagnostic Findings & Boundaries</div>', unsafe_allow_html=True)

                # 1. Reported Symptoms (User Claims)
                symptoms = active_inv.get("reported_symptoms", [])
                if not symptoms and selected_incident.get("description"):
                    symptoms = [selected_incident["description"]]

                symptom_items = "".join(f"<li style='margin-bottom: 4px;'>{_safe_html_text(s)}</li>" for s in symptoms)
                st.markdown(f"""
                <div class="diag-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #14213D; font-size: 13px;">REPORTED SYMPTOMS & OBSERVATIONS</span>
                        <span style="font-size: 11px; color: #64748B; font-weight: 600;">UNVERIFIED USER-REPORTED INFORMATION</span>
                    </div>
                    <ul style="color: #4A5568; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.5;">
                        {symptom_items}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # 2. Confirmed Technical Facts (Refinement #2: Trust boundary)
                confirmed_facts = active_inv.get("confirmed_facts", [])
                if confirmed_facts:
                    facts_items = "".join(f"<li style='margin-bottom: 4px;'>{_safe_html_text(f)}</li>" for f in confirmed_facts)
                    st.markdown(f"""
                    <div class="diag-card">
                        <div style="font-weight: 700; color: #14213D; font-size: 13px; margin-bottom: 6px;">CONFIRMED TECHNICAL FACTS</div>
                        <ul style="color: #166534; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.5;">
                            {facts_items}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Explicit explanation that no external telemetry is connected
                    st.markdown("""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; font-size: 13px; color: #64748B; margin-bottom: 12px;">
                        <div style="font-weight: 700; color: #475569; margin-bottom: 2px;">ℹ️ 0 CONFIRMED FACTS (NO INDEPENDENTLY VERIFIED TELEMETRY AVAILABLE)</div>
                        OpsPilot does not currently have an authenticated external telemetry or log streaming integration. To prevent epistemological drift, user-submitted claims and runbook reference text are never treated as confirmed runtime facts without verified telemetry.
                    </div>
                    """, unsafe_allow_html=True)

                # 3. Diagnostic Assumptions
                assumptions = active_inv.get("assumptions", [])
                if assumptions:
                    assump_items = "".join(f"<li style='margin-bottom: 4px;'>{_safe_html_text(a)}</li>" for a in assumptions)
                    st.markdown(f"""
                    <div class="diag-card">
                        <div style="font-weight: 700; color: #14213D; font-size: 13px; margin-bottom: 6px;">DIAGNOSTIC ASSUMPTIONS</div>
                        <ul style="color: #4A5568; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.5;">
                            {assump_items}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # 4. Root-Cause Hypotheses
                hypotheses = active_inv.get("root_cause_hypotheses", [])
                st.markdown('<div style="font-weight: 750; font-size: 15px; color: #14213D; margin-top: 14px; margin-bottom: 8px;">Root-Cause Hypotheses & Evidence Grounding</div>', unsafe_allow_html=True)

                if is_fallback:
                    if not has_retrieved_runbooks:
                        hypo_offline_msg = "Automated root-cause hypotheses could not be synthesized because AI diagnostic reasoning is offline. No runbooks were retrieved from the Knowledge Base. Author an operator remediation procedure in the Governance Console."
                    else:
                        hypo_offline_msg = "Automated root-cause hypotheses could not be synthesized because AI diagnostic reasoning is offline. Consult the retrieved runbook guidance and inspection steps below."
                    st.markdown(_clean_html_block(f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; font-size: 13px; color: #64748B; margin-bottom: 12px;">
                        <div style="font-weight: 700; color: #475569; margin-bottom: 2px;">ℹ️ AUTOMATED HYPOTHESES OFFLINE</div>
                        {hypo_offline_msg}
                    </div>
                    """), unsafe_allow_html=True)
                elif hypotheses:
                    for h_idx, hypo in enumerate(hypotheses):
                        h_text = _safe_html_text(hypo.get("hypothesis", "Unspecified hypothesis"))
                        h_like = str(hypo.get("likelihood", "Medium")).title()
                        like_badge_cls = "badge-med"
                        if h_like == "High":
                            like_badge_cls = "badge-high"
                        elif h_like == "Low":
                            like_badge_cls = "badge-low"

                        cites = hypo.get("supporting_evidence_citations", [])
                        cite_badges = ""
                        if cites:
                            cite_badges = " ".join(f'<span class="citation-badge">{_safe_html_text(c)}</span>' for c in cites)
                        else:
                            cite_badges = '<span style="color: #94A3B8; font-size: 11px; font-style: italic;">No specific runbook citation</span>'

                        unsupported = hypo.get("unsupported_claims", [])
                        unsupported_html = ""
                        if unsupported:
                            unsupported_text = "; ".join(_safe_html_text(u) for u in unsupported)
                            unsupported_html = f'<div style="margin-top: 6px; font-size: 12px; color: #B45309; background: #FFFBEB; padding: 4px 8px; border-radius: 4px; border: 1px solid #FDE68A;">⚠️ <strong>Unverified Inferences:</strong> {unsupported_text}</div>'

                        st.markdown(f"""
                        <div class="diag-card" style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div style="font-weight: 700; color: #14213D; font-size: 14px; line-height: 1.4; flex: 1;">
                                    {h_idx + 1}. {h_text}
                                </div>
                                <span class="{like_badge_cls}" style="margin-left: 10px; white-space: nowrap;">{h_like} Likelihood</span>
                            </div>
                            <div style="margin-top: 8px; font-size: 12px; color: #64748B;">
                                Supporting Evidence: {cite_badges}
                            </div>
                            {unsupported_html}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific root-cause hypotheses were generated.")

                # 5. Suggested Diagnostic Steps (Refinement #4)
                diag_actions = active_inv.get("diagnostic_actions", [])
                if diag_actions:
                    if is_fallback:
                        st.markdown("""
                        <div style="margin-top: 14px; margin-bottom: 6px;">
                            <span style="font-weight: 750; font-size: 15px; color: #14213D;">RECOMMENDED MANUAL DIAGNOSTIC STEPS (ADVISORY PROCEDURES — NOT AI-GENERATED HYPOTHESES)</span>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; font-size: 12px; color: #64748B; margin-bottom: 10px;">
                            ⚠️ <strong>Advisory only:</strong> These procedural inspection steps are drawn from retrieved runbook guidance for manual operator review. <strong>OpsPilot does not execute commands automatically.</strong> All commands must be reviewed and executed manually by an authorized engineer.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="margin-top: 14px; margin-bottom: 6px;">
                            <span style="font-weight: 750; font-size: 15px; color: #14213D;">Suggested Diagnostic Inspection Steps</span>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; font-size: 12px; color: #64748B; margin-bottom: 10px;">
                            ⚠️ <strong>Advisory only:</strong> These are suggested inspection steps and commands for operator review. <strong>OpsPilot does not execute commands automatically.</strong> All commands must be reviewed and executed manually by an authorized engineer.
                        </div>
                        """, unsafe_allow_html=True)
                    for d_idx, d_act in enumerate(diag_actions):
                        st.caption(f"Inspection Step {d_idx + 1}:")
                        st.code(d_act, language="bash")

                # 6. Information Gaps (Missing Information)
                missing_info = active_inv.get("missing_information", [])
                if missing_info:
                    with st.expander("🔍 Diagnostic Information Gaps & Missing Telemetry", expanded=False):
                        for m_item in missing_info:
                            st.markdown(f"• {_safe_html_text(m_item)}")

            with col_evidence:
                st.markdown('<div class="section-heading" style="font-size: 17px; margin-top: 0; margin-bottom: 4px;">Retrieved Runbook Reference Material</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size: 11px; font-weight: 700; color: #64748B; margin-bottom: 10px; letter-spacing: 0.5px;">REFERENCE MATERIAL — NOT INCIDENT FACTS</div>', unsafe_allow_html=True)

                # Citation Validation & Grounding Audit (Refinement #3)
                val_summary = active_inv.get("citation_validation_summary", {})
                total_cites = val_summary.get("total_citations", len(active_inv.get("evidence_citations", [])))
                valid_cites = val_summary.get("valid_citations", total_cites)
                flagged_cites = val_summary.get("flagged_citations", 0)

                st.markdown(_clean_html_block(f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 14px;">
                    <div style="font-weight: 700; color: #14213D; font-size: 13px; margin-bottom: 6px;">CITATION GROUNDING AUDIT</div>
                    <div style="display: flex; gap: 14px; font-size: 12px; margin-bottom: 6px;">
                        <span><span style="margin-right: 4px;">Total Cited:</span><strong>{total_cites}</strong></span>
                        <span style="color: #166534;"><span style="margin-right: 4px;">Grounded:</span><strong>{valid_cites}</strong></span>
                        <span style="color: {'#DC2626' if flagged_cites > 0 else '#64748B'};"><span style="margin-right: 4px;">Flagged:</span><strong>{flagged_cites}</strong></span>
                    </div>
                    <div style="font-size: 11px; color: #8290A5; line-height: 1.4;">
                        Validates citations against retrieved knowledge chunks. Distinguishes grounded references from unsupported inferences.
                    </div>
                </div>
                """), unsafe_allow_html=True)

                if flagged_cites > 0:
                    flagged_details = val_summary.get("flagged_details", [])
                    st.warning(f"⚠️ **Citation Alert**: {flagged_cites} citation(s) could not be verified against the retrieved runbook packet.")
                    for fd in flagged_details:
                        st.caption(f"• Flagged citation '{fd.get('citation')}' was rejected from grounding.")

                # Provenance Chunks
                prov_chunks = active_inv.get("provenance", [])
                if prov_chunks:
                    st.caption(f"Showing {len(prov_chunks)} retrieved knowledge chunk(s):")
                    for p_chunk in prov_chunks:
                        c_rank = p_chunk.get("rank", 1)
                        c_cite = _safe_html_text(p_chunk.get("citation") or "Unknown Citation")
                        c_doc = _safe_html_text(p_chunk.get("document_name") or "Runbook")
                        c_page = p_chunk.get("page_number", 1)
                        c_idx = p_chunk.get("chunk_index", 0)
                        c_bm25 = p_chunk.get("bm25_score")
                        bm25_str = f"{c_bm25:.3f}" if c_bm25 is not None else "N/A"
                        c_content = p_chunk.get("content", "")

                        st.markdown(f"""
                        <div class="evidence-item">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span class="citation-badge" style="font-size: 11px;">{c_cite}</span>
                                <span style="font-size: 11px; font-weight: 700; color: #315BEA;">Rank #{c_rank}</span>
                            </div>
                            <div style="font-size: 12px; color: #14213D; font-weight: 600;">
                                {c_doc} &nbsp;<span style="color: #64748B; font-weight: 400;">(Page {c_page}, Chunk {c_idx})</span>
                            </div>
                            <div style="font-size: 11px; color: #8290A5; margin-top: 2px; margin-bottom: 6px;">
                                BM25 Score: <code>{bm25_str}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if c_content:
                            with st.expander(f"View excerpt: {p_chunk.get('citation')}", expanded=False):
                                st.markdown(f"<div style='font-size: 12px; color: #4A5568; line-height: 1.5;'>{_safe_html_text(c_content, preserve_newlines=True)}</div>", unsafe_allow_html=True)
                else:
                    st.info("No matching enterprise runbooks were retrieved from the knowledge base for this incident.")

            st.markdown("<br>", unsafe_allow_html=True)

            # ==================================================================
            # OPERATOR GOVERNANCE & REMEDIATION CONSOLE
            # ==================================================================
            with st.container(border=True):
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span class="eyebrow" style="color: #315BEA;">HUMAN-IN-THE-LOOP GOVERNANCE</span>
                        <div style="font-weight: 800; font-size: 18px; color: #14213D; margin-top: 2px;">Remediation Plan & Operator Decision</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 11px; font-weight: 700; color: #64748B;">MODEL CONFIDENCE:</span>
                        <span style="background: #E2E8F0; color: #1E293B; padding: 2px 8px; border-radius: 10px; font-weight: 700; font-size: 12px; margin-left: 4px;">{conf_display}</span>
                        <div style="font-size: 10px; color: #8290A5; margin-top: 2px;">({conf_subtext})</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if is_fallback:
                    st.caption("⚠️ **Notice:** Automated AI diagnostic synthesis was offline. No AI confidence score was generated. Explicit operator authoring and approval are required to establish an incident remediation action.")
                else:
                    st.caption("⚠️ **Notice:** Model confidence reflects internal diagnostic heuristic self-assessment, not a calibrated mathematical probability. Human review and explicit operator approval are strictly required before proceeding.")

                # Render AI Recommendation
                raw_ai_rec = active_inv.get("ai_recommendation", "")
                if is_fallback and not has_retrieved_runbooks:
                    raw_ai_rec = _format_fallback_guidance(raw_ai_rec, citations=[], total_docs=0)
                esc_ai_rec = _safe_html_text(raw_ai_rec, preserve_newlines=True)

                rec_heading = "FALLBACK DIAGNOSTIC GUIDANCE" if is_fallback else "PROPOSED REMEDIATION PROCEDURE"
                st.markdown(_clean_html_block(f"""
                <div style="margin-top: 10px; margin-bottom: 14px; padding: 14px 16px; background: #F8FAFC; border-left: 3px solid #315BEA; border-radius: 6px;">
                    <div style="font-weight: 700; color: #14213D; font-size: 12.5px; margin-bottom: 6px; letter-spacing: 0.5px;">{rec_heading}</div>
                    <div style="color: #1E293B; font-size: 13.5px; line-height: 1.6; word-break: break-word;">{esc_ai_rec}</div>
                </div>
                """), unsafe_allow_html=True)

                # Decision Handling (Refinement #5 & Safety Correction)
                if decision == "Pending":
                    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #14213D; margin-top: 12px; margin-bottom: 6px;'>OPERATOR GOVERNANCE DECISION</div>", unsafe_allow_html=True)

                    review_notes = st.text_input(
                        "Reviewer justification / notes (optional for approval, recommended for rejection):",
                        placeholder="Add engineering notes, operational context, or review reasoning...",
                        key=f"inv_notes_input_{inv_id}",
                    )

                    if is_fallback:
                        # Safety Correction:
                        # 1. Hide or disable "Approve As-Is"
                        # 2. Keep "Edit & Customize" available (via manual procedure authoring)
                        # 3. Require operator-authored manual remediation action before approval
                        # 4. Label resulting action clearly as "Operator-authored remediation — not AI-generated"
                        st.markdown("""
                        <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
                            <div style="font-weight: 750; color: #92400E; font-size: 13px;">⚠️ OPERATOR ACTION REQUIRED — AI REMEDIATION UNAVAILABLE</div>
                            <div style="font-size: 12.5px; color: #78350F; margin-top: 4px; line-height: 1.4;">
                                Because automated AI synthesis was offline, no AI remediation proposal was generated. "Approve As-Is" is disabled. To approve remediation for this incident, an operator must author a manual procedure below.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        dec_col1, dec_col2 = st.columns([1, 1])
                        with dec_col1:
                            st.button(
                                "🚫 Approve As-Is (Disabled — No AI Proposal)",
                                key=f"btn_app_asis_{inv_id}",
                                disabled=True,
                                use_container_width=True,
                                help="Disabled because automated AI synthesis is offline and produced no remediation recommendation.",
                            )
                        with dec_col2:
                            show_reject = st.checkbox("❌ Reject Proposal", key=f"chk_rej_{inv_id}")

                        if show_reject:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.warning("Rejecting this proposal will reset the incident status back to 'Open'. An audit record will be permanently logged.")
                            rej_notes = st.text_area(
                                "Rejection Justification:",
                                placeholder="Explain why this fallback proposal is being rejected...",
                                height=70,
                                key=f"inv_rej_notes_{inv_id}",
                            )
                            if st.button("🚫 Confirm Rejection", key=f"btn_confirm_rej_{inv_id}", use_container_width=True):
                                try:
                                    success = incident_service.apply_investigation_decision(
                                        investigation_id=inv_id,
                                        decision="Rejected",
                                        reviewer_notes=rej_notes or "Operator rejected fallback proposal.",
                                        actor="Operator",
                                    )
                                    if success:
                                        st.session_state["investigation_flash"] = {
                                            "type": "info",
                                            "msg": f"ℹ️ Fallback Proposal #{inv_id} was rejected. Incident status reset to 'Open'.",
                                        }
                                        st.session_state[f"proposal_just_rejected_{inv_id}"] = True
                                        st.toast(f"ℹ️ Fallback Proposal #{inv_id} was rejected.", icon="ℹ️")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to record rejection.")
                                        st.toast("❌ Failed to record rejection.", icon="❌")
                                except Exception as rej_err:
                                    st.error(f"❌ Error during rejection: {rej_err}")
                                    st.toast(f"❌ Error during rejection: {rej_err}", icon="❌")
                        else:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("<div style='font-weight: 700; color: #14213D; font-size: 13px; margin-bottom: 4px;'>Author Manual Remediation Procedure (Required for Approval):</div>", unsafe_allow_html=True)
                            edited_action = st.text_area(
                                "Operator Remediation Action (Required):",
                                placeholder="Enter explicit manual remediation steps, commands, or runbook instructions...",
                                height=140,
                                key=f"custom_action_input_{inv_id}",
                            )
                            if st.button("💾 Save & Approve Operator Remediation", key=f"btn_save_custom_{inv_id}", use_container_width=True):
                                action_val = (st.session_state.get(f"custom_action_input_{inv_id}") or edited_action or "").strip()
                                notes_val = (st.session_state.get(f"inv_notes_input_{inv_id}") or review_notes or "").strip()
                                if not action_val:
                                    st.error("⚠️ An operator remediation procedure is required before approval can be recorded. Please author explicit manual remediation steps above.")
                                    st.toast("⚠️ Operator remediation procedure cannot be empty.", icon="⚠️")
                                else:
                                    try:
                                        safe_notes = notes_val or None
                                        success = incident_service.apply_investigation_decision(
                                            investigation_id=inv_id,
                                            decision="Approved",
                                            final_action=action_val,
                                            reviewer_notes=safe_notes,
                                            actor="Operator",
                                            is_operator_authored=True,
                                        )
                                        if success:
                                            st.session_state["investigation_flash"] = {
                                                "type": "success",
                                                "msg": f"✅ Operator-authored remediation saved and approved for Investigation #{inv_id}.",
                                            }
                                            st.session_state[f"remediation_just_approved_{inv_id}"] = True
                                            st.toast(f"✅ Operator-authored remediation approved for Investigation #{inv_id}!", icon="✅")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to record operator remediation.")
                                            st.toast("❌ Failed to record operator remediation.", icon="❌")
                                    except Exception as edit_err:
                                        st.error(f"❌ Error saving operator remediation: {edit_err}")
                                        st.toast(f"❌ Error: {edit_err}", icon="❌")

                    else:
                        # Standard AI Investigation Governance Flow
                        dec_col1, dec_col2, dec_col3 = st.columns([1, 1, 1])

                        # 1. Approve As-Is
                        with dec_col1:
                            if st.button("✅ Approve As-Is", key=f"btn_app_asis_{inv_id}", use_container_width=True):
                                try:
                                    notes_val = (st.session_state.get(f"inv_notes_input_{inv_id}") or review_notes or "").strip()
                                    success = incident_service.apply_investigation_decision(
                                        investigation_id=inv_id,
                                        decision="Approved",
                                        reviewer_notes=notes_val or None,
                                        actor="Operator",
                                    )
                                    if success:
                                        st.session_state["investigation_flash"] = {
                                            "type": "success",
                                            "msg": f"✅ AI Recommendation for Investigation #{inv_id} approved as-is.",
                                        }
                                        st.session_state[f"remediation_just_approved_{inv_id}"] = True
                                        st.toast(f"✅ AI Recommendation for Investigation #{inv_id} approved as-is!", icon="✅")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to approve recommendation.")
                                        st.toast("❌ Failed to approve recommendation.", icon="❌")
                                except Exception as app_err:
                                    st.error(f"❌ Error during approval: {app_err}")
                                    st.toast(f"❌ Error: {app_err}", icon="❌")

                        with dec_col2:
                            show_edit = st.checkbox("✏️ Edit & Customize", key=f"chk_edit_{inv_id}")

                        with dec_col3:
                            show_reject = st.checkbox("❌ Reject Proposal", key=f"chk_rej_{inv_id}")

                        if show_edit:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.info("Edit the remediation steps below. Your custom action will be saved as `final_action` while preserving the original AI recommendation in the database for auditing.")
                            edited_action = st.text_area(
                                "Custom Remediation Action:",
                                value=raw_ai_rec,
                                height=140,
                                key=f"custom_action_input_{inv_id}",
                            )
                            if st.button("💾 Save & Approve Customized Action", key=f"btn_save_custom_{inv_id}", use_container_width=True):
                                action_val = (st.session_state.get(f"custom_action_input_{inv_id}") or edited_action or "").strip()
                                notes_val = (st.session_state.get(f"inv_notes_input_{inv_id}") or review_notes or "").strip()
                                if not action_val:
                                    st.error("⚠️ Customized action cannot be empty.")
                                    st.toast("⚠️ Customized action cannot be empty.", icon="⚠️")
                                else:
                                    try:
                                        success = incident_service.apply_investigation_decision(
                                            investigation_id=inv_id,
                                            decision="Edited",
                                            final_action=action_val,
                                            reviewer_notes=notes_val or None,
                                            actor="Operator",
                                        )
                                        if success:
                                            st.session_state["investigation_flash"] = {
                                                "type": "success",
                                                "msg": f"✅ Customized remediation saved and approved for Investigation #{inv_id}. Original recommendation preserved.",
                                            }
                                            st.session_state[f"remediation_just_approved_{inv_id}"] = True
                                            st.toast(f"✅ Customized remediation for Investigation #{inv_id} approved!", icon="✅")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to record edited approval.")
                                            st.toast("❌ Failed to record edited approval.", icon="❌")
                                    except Exception as edit_err:
                                        st.error(f"❌ Error saving customized action: {edit_err}")
                                        st.toast(f"❌ Error: {edit_err}", icon="❌")

                        if show_reject:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.warning("Rejecting this proposal will reset the incident status back to 'Open' and clear any approved final action. An audit record will be permanently logged.")
                            if st.button("🚫 Confirm Rejection", key=f"btn_confirm_rej_{inv_id}", use_container_width=True):
                                try:
                                    success = incident_service.apply_investigation_decision(
                                        investigation_id=inv_id,
                                        decision="Rejected",
                                        reviewer_notes=review_notes or "Operator rejected proposed AI recommendation.",
                                        actor="Operator",
                                    )
                                    if success:
                                        st.session_state["investigation_flash"] = {
                                            "type": "info",
                                            "msg": f"ℹ️ AI Recommendation for Investigation #{inv_id} was rejected. Incident status reset to 'Open'.",
                                        }
                                        st.session_state[f"proposal_just_rejected_{inv_id}"] = True
                                        st.toast(f"ℹ️ AI Recommendation for Investigation #{inv_id} was rejected.", icon="ℹ️")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to record rejection.")
                                        st.toast("❌ Failed to record rejection.", icon="❌")
                                except Exception as rej_err:
                                    st.error(f"❌ Error during rejection: {rej_err}")
                                    st.toast(f"❌ Error: {rej_err}", icon="❌")

                elif decision == "Approved":
                    if st.session_state.pop(f"remediation_just_approved_{inv_id}", False):
                        st.success(f"✅ Remediation decision recorded successfully for Investigation #{inv_id}. Incident status updated to 'Approved'.")

                    raw_final = active_inv.get("final_action") or raw_ai_rec
                    esc_final = _format_remediation_html(raw_final)
                    notes_esc = _safe_html_text(active_inv.get("reviewer_notes") or "None recorded")
                    is_custom = bool(active_inv.get("final_action") and active_inv.get("final_action") != raw_ai_rec)
                    if is_fallback:
                        type_str = "Operator-authored remediation — not AI-generated"
                    elif is_custom:
                        type_str = "Customized by Operator"
                    else:
                        type_str = "Adopted As-Is"

                    st.markdown(_clean_html_block(f"""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; border-radius: 6px; padding: 14px; margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 750; color: #166534; font-size: 14px;">✅ APPROVED REMEDIATION ({type_str})</span>
                        </div>
                        <div style="color: #14532D; font-size: 13.5px; margin-top: 8px; line-height: 1.5; word-break: break-word;">
                            {esc_final}
                        </div>
                        <div style="font-size: 12px; color: #15803D; margin-top: 8px;">
                            <span style="margin-right: 4px;">Reviewer Notes:</span><em>{notes_esc}</em>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

                elif decision == "Rejected":
                    if st.session_state.pop(f"proposal_just_rejected_{inv_id}", False):
                        st.info(f"ℹ️ Proposal #{inv_id} was rejected. Incident status reset to 'Open'.")

                    rej_notes = _safe_html_text(active_inv.get("reviewer_notes") or "None specified")
                    st.markdown(_clean_html_block(f"""
                    <div style="background: #FEF2F2; border: 1px solid #FECACA; border-left: 4px solid #DC2626; border-radius: 6px; padding: 14px; margin-top: 10px;">
                        <div style="font-weight: 750; color: #991B1B; font-size: 14px;">❌ RECOMMENDATION REJECTED</div>
                        <div style="font-size: 13px; color: #7F1D1D; margin-top: 6px;">
                            <span style="margin-right: 4px;">Justification:</span><em>{rej_notes}</em>
                        </div>
                        <div style="font-size: 12px; color: #B91C1C; margin-top: 6px;">
                            <span style="margin-right: 4px;">Status:</span>Incident has been returned to <strong style="color: #991B1B;">Open</strong> status with no approved final action.
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

            # Historical Audit Trail & Previous Iterations Tabs
            st.markdown("<br>", unsafe_allow_html=True)
            tab_audit, tab_hist = st.tabs(["📜 Incident Governance Audit Trail", "🔍 Previous Investigation Proposals"])

            with tab_audit:
                try:
                    audit_logs = incident_service.get_audit_history_for_incident(selected_id)
                except Exception:
                    audit_logs = []

                if audit_logs:
                    latest_log = audit_logs[-1]
                    latest_act = latest_log.get("action", "")
                    latest_actor = latest_log.get("actor", "Operator")
                    latest_ts = helpers.format_timestamp(latest_log.get("created_at"))
                    if latest_act == "GOVERNANCE_AUDIT_CORRECTION" or "CORRECTION" in latest_act:
                        st.caption(f"⚖️ **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")
                    elif "APPROVED" in latest_act:
                        st.caption(f"🔒 **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")
                    elif "REJECTED" in latest_act:
                        st.caption(f"⚠️ **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")
                    elif "EDITED" in latest_act:
                        st.caption(f"✏️ **Latest Governance Event**: `{latest_act}` by `{latest_actor}` ({latest_ts})")

                    audit_records = [
                        {
                            "Timestamp": helpers.format_timestamp(entry.get("created_at")),
                            "Action": entry.get("action", ""),
                            "Actor": entry.get("actor", "System"),
                            "Details": entry.get("notes") or "",
                        }
                        for entry in audit_logs
                    ]
                    st.dataframe(audit_records, use_container_width=True, hide_index=True)
                else:
                    st.info("No audit logs recorded for this incident.")

            with tab_hist:
                if len(existing_inv_rows) > 1:
                    for prev_raw in existing_inv_rows[1:]:
                        prev_inv = incident_service.deserialize_investigation(prev_raw)
                        p_id = prev_inv.get("id")
                        p_time = _safe_html_text(helpers.format_timestamp(prev_inv.get("created_at")))
                        p_dec = _safe_html_text(prev_inv.get("human_decision", "Pending"))
                        p_rec_raw = prev_inv.get("ai_recommendation", "")
                        p_cites = prev_inv.get("evidence_citations") or []
                        p_is_fallback = (
                            prev_inv.get("is_fallback") is True
                            or prev_inv.get("confidence") is None
                            or "Automated AI diagnostic synthesis is currently unavailable" in p_rec_raw
                        )
                        if p_is_fallback and not p_cites:
                            p_rec_raw = _format_fallback_guidance(p_rec_raw, citations=p_cites, total_docs=0)
                        p_rec = _safe_html_text(p_rec_raw)
                        st.markdown(_clean_html_block(f"""
                        <div class="metric-card" style="margin-bottom: 10px; padding: 14px;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>Investigation #{p_id}</strong>
                                <span class="metric-note">{p_time} | Decision: <code>{p_dec}</code></span>
                            </div>
                            <div style="font-size: 13px; color: #4A5568; margin-top: 6px;">
                                {p_rec[:200]}...
                            </div>
                        </div>
                        """), unsafe_allow_html=True)
                else:
                    st.caption("No prior investigation iterations recorded for this incident.")

elif page == "Knowledge Base":
    st.markdown('<div class="eyebrow">DOCUMENTS / RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Manage standard operating procedures and runbooks.</div>',
        unsafe_allow_html=True
    )
    st.info("The Knowledge Base and RAG pipeline will be connected in Phase 4.")

elif page == "Simulation Lab":
    st.markdown('<div class="eyebrow">LAB / SCENARIOS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Simulation Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Run realistic incident drills and benchmark workflows.</div>',
        unsafe_allow_html=True
    )
    st.info("The Simulation Lab will be connected in Phase 6.")

elif page == "Evaluation":
    st.markdown('<div class="eyebrow">BENCHMARK / METRICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Verify classification accuracy and retrieval precision.</div>',
        unsafe_allow_html=True
    )
    st.info("The Evaluation Dashboard will be connected in Phase 6.")

st.divider()
st.caption("OpsPilot AI • Evidence-grounded incident intelligence • Human-supervised decisions")
