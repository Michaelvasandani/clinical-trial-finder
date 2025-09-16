"""Reusable UI components for the Clinical Trial Finder Streamlit app."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from datetime import datetime
from .styles import render_match_score, render_trial_status_badge

def render_trial_card(trial_data: Dict, show_match_info: bool = False) -> None:
    """Render an interactive trial card with expand/collapse details."""
    
    # Extract trial information - handle both flat and nested formats
    nct_id = trial_data.get("nct_id") or trial_data.get("NCTId", "")
    title = trial_data.get("title") or trial_data.get("metadata", {}).get("BriefTitle", "No title available")
    condition = trial_data.get("condition") or trial_data.get("metadata", {}).get("Condition", "Not specified")
    status = trial_data.get("status") or trial_data.get("metadata", {}).get("OverallStatus", "Unknown")
    phase = trial_data.get("phase") or trial_data.get("metadata", {}).get("Phase", "Not specified")
    location = trial_data.get("location") or trial_data.get("metadata", {}).get("LocationState", "Not specified")
    score = trial_data.get("score", 0)
    reranked_score = trial_data.get("reranked_score", score)
    
    # Create unique identifier for buttons (fallback when NCT ID is missing)
    unique_id = nct_id if nct_id else f"trial_{hash(str(trial_data)) % 100000}"
    
    # Create the main card container
    with st.container():
        # Card header with trial info
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="trial-card">
                <div class="trial-title">{title}</div>
                <div class="trial-meta"><strong>NCT ID:</strong> {nct_id}</div>
                <div class="trial-meta"><strong>Condition:</strong> {condition}</div>
                <div class="trial-meta"><strong>Phase:</strong> {phase}</div>
                <div class="trial-meta"><strong>Location:</strong> {location}</div>
                <div style="margin-top: 10px;">
                    {render_trial_status_badge(status)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if show_match_info:
                # Display match score
                st.markdown(render_match_score(reranked_score), unsafe_allow_html=True)
        
        # Expandable section for detailed information
        with st.expander("📋 View Details", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Overview", "Eligibility", "Contact"])
            
            with tab1:
                # Display available information only
                st.write("**Trial Information:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if nct_id:
                        st.write(f"**NCT ID:** {nct_id}")
                    
                    if condition and condition != "Not specified":
                        st.write(f"**Condition:** {condition}")
                    
                    if phase and phase != "Not specified":
                        st.write(f"**Phase:** {phase}")
                
                with col2:
                    if status and status != "Unknown":
                        st.write(f"**Status:** {status}")
                    
                    if location and location != "Not specified":
                        st.write(f"**Location:** {location}")
                    
                    if score > 0:
                        st.write(f"**Match Score:** {int(score * 100)}%")
                
                # Show full title if available
                if title and title != "No title available":
                    st.write("**Full Title:**")
                    st.write(title)
                
                # Note about additional information
                st.info("💡 For complete trial details, visit the ClinicalTrials.gov link below.")
            
            with tab2:
                eligibility = trial_data.get("metadata", {}).get("EligibilityCriteria", "")
                if eligibility:
                    st.write("**Eligibility Criteria:**")
                    # Clean up and format eligibility criteria
                    eligibility_formatted = eligibility.replace("\\n", "\n").strip()
                    st.text_area("", eligibility_formatted, height=200, disabled=True)
                else:
                    st.write("Eligibility criteria not available for this trial.")
                
                # Age criteria
                min_age = trial_data.get("metadata", {}).get("MinimumAge", "")
                max_age = trial_data.get("metadata", {}).get("MaximumAge", "")
                gender = trial_data.get("metadata", {}).get("Gender", "")
                
                if min_age or max_age or gender:
                    st.write("**Quick Eligibility Check:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if min_age:
                            st.metric("Minimum Age", min_age)
                    
                    with col2:
                        if max_age:
                            st.metric("Maximum Age", max_age)
                    
                    with col3:
                        if gender:
                            st.metric("Gender", gender)
            
            with tab3:
                # Contact information
                contact_name = trial_data.get("metadata", {}).get("ContactName", "")
                contact_phone = trial_data.get("metadata", {}).get("ContactPhone", "")
                contact_email = trial_data.get("metadata", {}).get("ContactEmail", "")
                facility = trial_data.get("metadata", {}).get("LocationFacility", "")
                city = trial_data.get("metadata", {}).get("LocationCity", "")
                
                if contact_name or contact_phone or contact_email:
                    st.write("**Study Contact:**")
                    if contact_name:
                        st.write(f"**Name:** {contact_name}")
                    if contact_phone:
                        st.write(f"**Phone:** {contact_phone}")
                    if contact_email:
                        st.write(f"**Email:** {contact_email}")
                else:
                    st.write("Contact information not available.")
                
                if facility or city:
                    st.write("**Study Location:**")
                    if facility:
                        st.write(f"**Facility:** {facility}")
                    if city:
                        st.write(f"**City:** {city}")
        
        # ClinicalTrials.gov link
        if nct_id:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px;">
                <a href="https://clinicaltrials.gov/ct2/show/{nct_id}" target="_blank" 
                   style="background-color: #f0f2f6; padding: 8px 16px; border-radius: 4px; 
                          text-decoration: none; color: #262730; font-weight: 500;">
                    📋 View Full Details on ClinicalTrials.gov
                </a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 Complete trial details available on ClinicalTrials.gov")

def render_match_explanation(patient_info: Dict, trial_data: Dict) -> None:
    """Render an explanation of why a patient matches a trial."""
    
    st.markdown("### 🎯 Why You Match This Trial")
    
    # Extract matching criteria
    patient_age = patient_info.get("age")
    patient_gender = patient_info.get("gender", "").lower()
    patient_conditions = patient_info.get("conditions", [])
    patient_location = patient_info.get("location", "")
    
    trial_condition = trial_data.get("metadata", {}).get("Condition", "")
    trial_min_age = trial_data.get("metadata", {}).get("MinimumAge", "")
    trial_max_age = trial_data.get("metadata", {}).get("MaximumAge", "")
    trial_gender = trial_data.get("metadata", {}).get("Gender", "ALL")
    trial_location = trial_data.get("metadata", {}).get("LocationState", "")
    
    # Match criteria analysis
    matches = []
    concerns = []
    
    # Age matching
    if patient_age:
        age_match = check_age_eligibility(patient_age, trial_min_age, trial_max_age)
        if age_match["eligible"]:
            matches.append(f"✅ **Age Match**: You are {patient_age} years old, which fits the trial's age range ({age_match['range']})")
        else:
            concerns.append(f"⚠️ **Age Concern**: You are {patient_age} years old, but the trial requires {age_match['range']}")
    
    # Gender matching
    if patient_gender and trial_gender != "ALL":
        if patient_gender.lower() in trial_gender.lower() or trial_gender.lower() in patient_gender.lower():
            matches.append(f"✅ **Gender Match**: The trial accepts {trial_gender.lower()} participants")
        else:
            concerns.append(f"⚠️ **Gender Concern**: The trial is for {trial_gender.lower()} participants only")
    
    # Condition matching
    condition_matches = []
    for condition in patient_conditions:
        if condition.lower() in trial_condition.lower():
            condition_matches.append(condition)
    
    if condition_matches:
        matches.append(f"✅ **Condition Match**: Your condition(s) {', '.join(condition_matches)} match the trial focus")
    
    # Location proximity
    if patient_location and trial_location:
        if patient_location.lower() in trial_location.lower() or trial_location.lower() in patient_location.lower():
            matches.append(f"✅ **Location Match**: Trial is in {trial_location}, close to your location")
    
    # Display matches
    if matches:
        st.markdown("#### Strong Matches")
        for match in matches:
            st.markdown(match)
    
    # Display concerns
    if concerns:
        st.markdown("#### Points to Discuss with Your Doctor")
        for concern in concerns:
            st.markdown(concern)
    
    # Overall recommendation
    match_score = len(matches) / max(len(matches) + len(concerns), 1)
    
    if match_score >= 0.8:
        recommendation = "🟢 **Strong Match** - This trial appears to be a good fit for your profile."
    elif match_score >= 0.5:
        recommendation = "🟡 **Moderate Match** - This trial may be suitable, but discuss concerns with your doctor."
    else:
        recommendation = "🔴 **Weak Match** - Consider other trials or discuss eligibility with the research team."
    
    st.markdown(f"#### Overall Assessment")
    st.markdown(recommendation)

def check_age_eligibility(patient_age: int, min_age_str: str, max_age_str: str) -> Dict:
    """Check if patient age meets trial eligibility."""
    result = {"eligible": True, "range": "No age restrictions"}
    
    def parse_age(age_str: str) -> Optional[int]:
        if not age_str:
            return None
        # Extract number from strings like "18 Years"
        import re
        match = re.search(r'(\d+)', age_str)
        return int(match.group(1)) if match else None
    
    min_age = parse_age(min_age_str)
    max_age = parse_age(max_age_str)
    
    if min_age is not None and max_age is not None:
        result["range"] = f"{min_age}-{max_age} years"
        result["eligible"] = min_age <= patient_age <= max_age
    elif min_age is not None:
        result["range"] = f"{min_age}+ years"
        result["eligible"] = patient_age >= min_age
    elif max_age is not None:
        result["range"] = f"Up to {max_age} years"
        result["eligible"] = patient_age <= max_age
    
    return result

def render_search_progress(stage: str) -> None:
    """Render search progress indicator."""
    stages = [
        "Analyzing your profile...",
        "Searching clinical trials...",
        "Matching trials to your criteria...",
        "Ranking results by relevance...",
        "Preparing personalized recommendations..."
    ]
    
    current_index = next((i for i, s in enumerate(stages) if stage in s), 0)
    progress = (current_index + 1) / len(stages)
    
    st.progress(progress)
    st.markdown(f"**{stage}**")

def render_patient_profile_summary(patient_info: Dict) -> None:
    """Render a summary of the extracted patient profile."""
    
    st.markdown("### 👤 Your Profile Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if patient_info.get("age"):
            st.metric("Age", f"{patient_info['age']} years")
        
        if patient_info.get("gender"):
            st.metric("Gender", patient_info["gender"].title())
    
    with col2:
        if patient_info.get("location"):
            st.metric("Location", patient_info["location"])
        
        conditions = patient_info.get("conditions", [])
        if conditions:
            st.metric("Conditions", f"{len(conditions)} identified")
    
    with col3:
        medications = patient_info.get("medications", [])
        if medications:
            st.metric("Medications", f"{len(medications)} listed")
    
    # Detailed sections
    if patient_info.get("conditions"):
        st.write("**Medical Conditions:**")
        for condition in patient_info["conditions"]:
            st.markdown(f"• {condition}")
    
    if patient_info.get("medications"):
        st.write("**Current Medications:**")
        for medication in patient_info["medications"]:
            st.markdown(f"• {medication}")


def render_message_history(messages: List[Dict]) -> None:
    """Render the chat message history."""
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp")
        
        with st.chat_message(role):
            st.markdown(content)
            
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except:
                        timestamp = None
                
                if timestamp:
                    st.caption(f"_{timestamp.strftime('%H:%M:%S')}_")

def render_conversation_starter() -> None:
    """Render conversation starter buttons."""
    
    st.markdown("### 💬 How can I help you today?")
    
    col1, col2, col3 = st.columns(3)
    
    starter_options = [
        ("🔍 Find Clinical Trials", "I'm looking for clinical trials that might be suitable for me."),
        ("❓ Learn About Trials", "Can you explain what clinical trials are and how they work?"),
        ("📋 Check Eligibility", "I found a trial and want to understand if I might be eligible.")
    ]
    
    for i, (button_text, message) in enumerate(starter_options):
        col = [col1, col2, col3][i]
        with col:
            if st.button(button_text, key=f"starter_{i}", use_container_width=True):
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                st.session_state.messages.append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now()
                })
                st.rerun()

def render_sidebar_info() -> None:
    """Render sidebar information and controls."""
    
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant helps you find and understand clinical trials. 
        
        **What I can do:**
        • Find relevant clinical trials
        • Explain medical terms
        • Help assess eligibility
        • Provide trial information
        
        **What I cannot do:**
        • Give medical advice
        • Recommend specific treatments
        • Replace your doctor
        """)
        
        st.markdown("### 🔧 Settings")
        
        # API connection status
        from .api_client import get_api_client
        api_client = get_api_client()
        
        if api_client.health_check():
            st.success("✅ Connected to backend")
        else:
            st.error("❌ Backend unavailable")
        
        # Conversation controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "conversation_id" in st.session_state:
                del st.session_state.conversation_id
            st.rerun()
        
        st.markdown("### ⚠️ Important")
        st.warning("""
        This tool is for informational purposes only. 
        Always consult with your healthcare provider before 
        making any medical decisions.
        """)