"""
Clinical Trial Finder - Streamlit Chat Interface

A sleek black and white chatbot interface for finding and understanding clinical trials.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import json
import time

# Import UI components
from ui.styles import apply_custom_css
from ui.api_client import get_api_client, with_loading
from ui.components import (
    render_trial_card, 
    render_match_explanation, 
    render_patient_profile_summary,
    render_message_history
)

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Finder",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply custom styling
apply_custom_css()

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = None
    
    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = None
    
    if "api_client" not in st.session_state:
        st.session_state.api_client = get_api_client()

def handle_chat_message(user_message: str):
    """Process a chat message and get AI response."""
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now()
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_message)
    
    # Show assistant response
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤔 Thinking...")
        
        # Check if this looks like a patient profile description
        is_patient_profile = any(keyword in user_message.lower() for keyword in [
            "i am", "i'm", "i have", "diagnosed with", "taking", "medication",
            "year old", "years old", "condition", "symptoms"
        ])
        
        if is_patient_profile and len(user_message) > 50:
            # Use patient extraction and matching
            thinking_placeholder.markdown("👤 Analyzing your profile...")
            
            result = with_loading(
                st.session_state.api_client.extract_and_match_patient,
                user_message,
                loading_text="Extracting patient information and finding matches..."
            )
            
            if result:
                # Store patient profile
                st.session_state.patient_profile = result.get("patient_info", {})
                matched_trials = result.get("matched_trials", [])
                patient_summary = result.get("patient_summary", "")
                
                # Create response with patient profile and trial matches
                response_content = f"""I've analyzed your profile and found some relevant clinical trials for you.

{patient_summary}

I found {len(matched_trials)} clinical trials that may be relevant to your situation. Let me show you the most promising options below."""
                
                thinking_placeholder.empty()
                st.markdown(response_content)
                
                # Display patient profile summary
                if st.session_state.patient_profile:
                    render_patient_profile_summary(st.session_state.patient_profile)
                
                # Display trial matches
                if matched_trials:
                    st.markdown("### 🎯 Recommended Clinical Trials")
                    
                    for i, trial in enumerate(matched_trials[:5]):  # Show top 5
                        with st.container():
                            st.markdown(f"#### Match #{i+1}")
                            render_trial_card(trial, show_match_info=True)
                            
                            # Show match explanation
                            with st.expander("🎯 Why this trial matches your profile"):
                                render_match_explanation(st.session_state.patient_profile, trial)
                            
                            st.markdown("---")
                
                # Store results for later reference
                st.session_state.last_search_results = matched_trials
                
            else:
                response_content = "❌ Sorry, I couldn't process your profile. Please try again."
                thinking_placeholder.empty()
                st.markdown(response_content)
        
        else:
            # Regular chat conversation
            thinking_placeholder.markdown("💬 Generating response...")
            
            result = with_loading(
                st.session_state.api_client.chat,
                user_message,
                st.session_state.conversation_id,
                include_search=True,
                loading_text="Processing your message..."
            )
            
            if result:
                # Update conversation ID
                st.session_state.conversation_id = result.get("conversation_id")
                
                # Display response
                response_content = result.get("content", "")
                search_results = result.get("search_results", [])
                
                thinking_placeholder.empty()
                st.markdown(response_content)
                
                # Display search results if available
                if search_results:
                    st.markdown("### 🔍 Related Clinical Trials")
                    
                    for trial in search_results[:3]:  # Show top 3
                        render_trial_card(trial, show_match_info=False)
                        st.markdown("---")
                
                st.session_state.last_search_results = search_results
                
            else:
                response_content = "❌ Sorry, I couldn't process your message. Please try again."
                thinking_placeholder.empty()
                st.markdown(response_content)
    
    # Add assistant response to messages history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_content if 'response_content' in locals() else "I apologize, but I encountered an error.",
        "timestamp": datetime.now()
    })

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Simple header
    st.markdown("""
    <div class="main-header">Clinical Trial Finder</div>
    <div class="sub-header">Ask me about clinical trials</div>
    """, unsafe_allow_html=True)
    
    # Add some vertical spacing to center the interface
    st.markdown("<div style='margin-top: 250px;'></div>", unsafe_allow_html=True)
    
    # Simple chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display existing message history (excluding the current message being processed)
        if st.session_state.messages:
            render_message_history(st.session_state.messages)
    
    # Chat input - process new messages AFTER displaying history
    if prompt := st.chat_input("Ask me about clinical trials..."):
        handle_chat_message(prompt)

if __name__ == "__main__":
    main()