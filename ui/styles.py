"""Custom CSS styles for the Clinical Trial Finder Streamlit app."""

import streamlit as st

def apply_custom_css():
    """Apply custom black and white theme CSS to the Streamlit app."""
    
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Chat message containers */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background-color: #000000;
        color: #ffffff;
        border-color: #333333;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f8f9fa;
        color: #000000;
        border-color: #e0e0e0;
    }
    
    /* Trial card styling */
    .trial-card {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .trial-card:hover {
        border-color: #000000;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .trial-title {
        color: #000000;
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 10px;
        line-height: 1.4;
    }
    
    .trial-meta {
        color: #666666;
        font-size: 0.9em;
        margin: 5px 0;
    }
    
    .trial-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-recruiting {
        background-color: #000000;
        color: #ffffff;
    }
    
    .status-active {
        background-color: #666666;
        color: #ffffff;
    }
    
    .status-completed {
        background-color: #e0e0e0;
        color: #333333;
    }
    
    /* Match score styling */
    .match-score {
        background: linear-gradient(45deg, #000000, #333333);
        color: #ffffff;
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        font-weight: 600;
    }
    
    .match-criteria {
        background-color: #f8f9fa;
        border-left: 4px solid #000000;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .match-criteria h4 {
        color: #000000;
        margin: 0 0 10px 0;
        font-size: 1em;
        font-weight: 600;
    }
    
    .match-item {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 5px 0;
        display: inline-block;
        font-size: 0.9em;
    }
    
    .match-item.highlight {
        background-color: #000000;
        color: #ffffff;
        border-color: #000000;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #000000;
        color: #ffffff;
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #ffffff;
        color: #000000;
        border-color: #000000;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px 15px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #000000;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading spinner */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .loading-text {
        color: #666666;
        font-style: italic;
        margin-left: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        color: #000000;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #000000;
        font-size: 2.5em;
        font-weight: 300;
        margin: 20px 0;
        letter-spacing: 1px;
    }
    
    .sub-header {
        text-align: center;
        color: #666666;
        font-size: 1.1em;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .trial-card {
            padding: 15px;
            margin: 8px 0;
        }
        
        .main-header {
            font-size: 2em;
        }
        
        .sub-header {
            font-size: 1em;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_loading_indicator(text="Processing..."):
    """Render a custom loading indicator."""
    return f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <span class="loading-text">{text}</span>
    </div>
    """

def render_match_score(score, max_score=1.0):
    """Render a match score indicator."""
    percentage = int((score / max_score) * 100)
    return f"""
    <div class="match-score">
        <strong>Match Score: {percentage}%</strong>
        <br>
        <small>Based on your profile and trial criteria</small>
    </div>
    """

def render_trial_status_badge(status):
    """Render a trial status badge with appropriate styling."""
    status_lower = status.lower()
    
    if "recruiting" in status_lower:
        css_class = "status-recruiting"
    elif "active" in status_lower:
        css_class = "status-active"
    else:
        css_class = "status-completed"
    
    return f'<span class="trial-status {css_class}">{status}</span>'