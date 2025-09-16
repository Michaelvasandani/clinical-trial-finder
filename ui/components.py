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
        with st.expander("View Details", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Overview", "Eligibility", "Contact"])

            with tab1:
                # Display trial overview
                col1, col2 = st.columns(2)

                with col1:
                    if nct_id:
                        st.markdown(f"**NCT ID:** `{nct_id}`")

                    if status and status != "Unknown":
                        st.markdown(f"**Status:** {status}")

                    if phase and phase != "Not specified":
                        st.markdown(f"**Phase:** {phase}")

                    study_type = trial_data.get("metadata", {}).get("StudyType", "")
                    if study_type:
                        st.markdown(f"**Study Type:** {study_type}")

                with col2:
                    if condition and condition != "Not specified":
                        st.markdown(f"**Condition:** {condition}")

                    if location and location != "Not specified":
                        st.markdown(f"**Location:** {location}")

                    if score > 0:
                        st.markdown(f"**Match Score:** {int(score * 100)}%")

                    enrollment = trial_data.get("metadata", {}).get("EnrollmentCount", "")
                    if enrollment:
                        st.markdown(f"**Enrollment:** {enrollment}")

                # Brief summary if available
                brief_summary = trial_data.get("metadata", {}).get("BriefSummary", "")
                if brief_summary and len(brief_summary) > 50:
                    st.markdown("**Study Description:**")
                    summary_clean = brief_summary.replace("\\n", "\n").replace("\\t", " ").strip()
                    # Limit summary length for cleaner display
                    if len(summary_clean) > 500:
                        summary_clean = summary_clean[:500] + "..."
                    st.text(summary_clean)

                # Intervention info
                intervention = trial_data.get("metadata", {}).get("InterventionName", "") or trial_data.get("metadata", {}).get("Intervention", "")
                if intervention:
                    st.markdown(f"**Treatment:** {intervention}")

            with tab2:
                # Basic eligibility criteria
                min_age = trial_data.get("metadata", {}).get("MinimumAge", "")
                max_age = trial_data.get("metadata", {}).get("MaximumAge", "")
                gender = trial_data.get("metadata", {}).get("Gender", "")

                if min_age or max_age or gender:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Min Age:** {min_age if min_age else 'None'}")
                    with col2:
                        st.markdown(f"**Max Age:** {max_age if max_age else 'None'}")
                    with col3:
                        st.markdown(f"**Gender:** {gender if gender else 'All'}")

                # Detailed eligibility criteria
                eligibility = trial_data.get("metadata", {}).get("EligibilityCriteria", "")
                if eligibility and len(eligibility) > 20:
                    st.markdown("**Eligibility Criteria:**")
                    eligibility_clean = eligibility.replace("\\n", "\n").replace("\\t", " ").strip()
                    # Limit length for cleaner display
                    if len(eligibility_clean) > 800:
                        eligibility_clean = eligibility_clean[:800] + "..."
                    st.text_area("", eligibility_clean, height=150, disabled=True, key="eligibility")
                else:
                    st.info("Detailed eligibility criteria not available. Contact the study team for requirements.")

            with tab3:
                st.markdown("### 📞 Get More Information")

                # Central contact information
                central_name = trial_data.get("metadata", {}).get("CentralContactName", "")
                central_role = trial_data.get("metadata", {}).get("CentralContactRole", "")
                central_phone = trial_data.get("metadata", {}).get("CentralContactPhone", "")
                central_email = trial_data.get("metadata", {}).get("CentralContactEmail", "")

                # Location contact information
                location_name = trial_data.get("metadata", {}).get("LocationContactName", "")
                location_role = trial_data.get("metadata", {}).get("LocationContactRole", "")
                location_phone = trial_data.get("metadata", {}).get("LocationContactPhone", "")
                location_email = trial_data.get("metadata", {}).get("LocationContactEmail", "")

                has_central_contact = central_name or central_phone or central_email
                has_location_contact = location_name or location_phone or location_email

                if has_central_contact:
                    st.markdown("#### Study Contact")
                    if central_name:
                        role_text = f" ({central_role})" if central_role else ""
                        st.markdown(f"**Name:** {central_name}{role_text}")
                    if central_phone:
                        st.markdown(f"**Phone:** {central_phone}")
                    if central_email:
                        st.markdown(f"**Email:** {central_email}")

                if has_location_contact:
                    header = "#### Site Contact" if has_central_contact else "#### Study Contact"
                    st.markdown(header)
                    # Show first location contact (split by semicolon for multiple locations)
                    first_location_name = location_name.split(';')[0].strip() if location_name else ""
                    first_location_role = location_role.split(';')[0].strip() if location_role else ""
                    first_location_phone = location_phone.split(';')[0].strip() if location_phone else ""
                    first_location_email = location_email.split(';')[0].strip() if location_email else ""

                    if first_location_name:
                        role_text = f" ({first_location_role})" if first_location_role else ""
                        st.markdown(f"**Name:** {first_location_name}{role_text}")
                    if first_location_phone:
                        st.markdown(f"**Phone:** {first_location_phone}")
                    if first_location_email:
                        st.markdown(f"**Email:** {first_location_email}")

                if not has_central_contact and not has_location_contact:
                    st.info("Use the ClinicalTrials.gov link below to find current contact details.")

                # Study location
                facility = trial_data.get("metadata", {}).get("LocationFacility", "")
                city = trial_data.get("metadata", {}).get("LocationCity", "")
                state = trial_data.get("metadata", {}).get("LocationState", "")
                zip_code = trial_data.get("metadata", {}).get("LocationZip", "")

                if facility or city:
                    st.markdown("#### 🏥 Study Location")
                    if facility:
                        st.markdown(f"**🏥 Facility:** {facility}")
                    if city:
                        location_str = city
                        if state:
                            location_str += f", {state}"
                        if zip_code:
                            location_str += f" {zip_code}"
                        st.markdown(f"**📍 Location:** {location_str}")

                # Sponsor information
                sponsor = trial_data.get("metadata", {}).get("Sponsor", "")
                if sponsor:
                    st.markdown("#### 🏢 Study Sponsor")
                    st.markdown(f"**Organization:** {sponsor}")

                # Next steps
                st.markdown("#### 🚀 Next Steps")
                st.markdown("""
                1. **Review the details** above to see if you might be eligible
                2. **Discuss with your doctor** about whether this trial could be right for you
                3. **Contact the study team** using the information above
                4. **Visit ClinicalTrials.gov** for the most up-to-date information
                """)

                if nct_id:
                    st.markdown("#### Additional Resources")
                    st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/ct2/show/{nct_id})")
                    st.markdown(f"[Search for similar trials](https://clinicaltrials.gov/ct2/results?cond={condition.replace(' ', '+')})")


    # ClinicalTrials.gov link at bottom
    if nct_id:
        st.markdown(f"""<div style="text-align: center; margin-top: 20px; padding-top: 15px; border-top: 1px solid #e0e0e0;">
            <a href="https://clinicaltrials.gov/ct2/show/{nct_id}" target="_blank"
               style="color: #1f77b4; text-decoration: none; font-size: 14px;">
                View Complete Details on ClinicalTrials.gov →
            </a>
        </div>""", unsafe_allow_html=True)

def render_match_explanation(patient_info: Dict, trial_data: Dict) -> None:
    """Render an explanation of why a patient matches a trial."""

    st.markdown("### 🎯 Why You Match This Trial")

    # Get the actual similarity score from the search results
    similarity_score = trial_data.get("score", 0)
    reranked_score = trial_data.get("reranked_score", similarity_score)
    primary_score = max(similarity_score, reranked_score)

    # Display the AI match strength based on actual similarity
    st.markdown("#### 🤖 AI Semantic Match Analysis")
    score_percentage = int(primary_score * 100)

    if primary_score >= 0.7:
        match_strength = "🟢 **Strong Semantic Match**"
        explanation = f"Our AI found a **{score_percentage}% similarity** between your profile and this trial's content. This indicates strong alignment in medical concepts, treatment approaches, and study objectives."
    elif primary_score >= 0.5:
        match_strength = "🟡 **Moderate Semantic Match**"
        explanation = f"Our AI found a **{score_percentage}% similarity** between your profile and this trial. This suggests good alignment in some key areas, though you may want to review the details carefully."
    else:
        match_strength = "🔴 **Lower Semantic Match**"
        explanation = f"Our AI found a **{score_percentage}% similarity** between your profile and this trial. While not the strongest match, it may still be relevant depending on your specific situation."

    st.markdown(match_strength)
    st.markdown(explanation)

    # Content-based matching analysis
    st.markdown("#### 📝 Content-Based Matching")
    _render_content_analysis(patient_info, trial_data)

    # Extract matching criteria for additional context
    patient_age = patient_info.get("age")
    patient_gender = patient_info.get("gender", "").lower()
    patient_conditions = patient_info.get("conditions", [])
    patient_location = patient_info.get("location", "")

    trial_condition = trial_data.get("metadata", {}).get("Condition", "")
    trial_min_age = trial_data.get("metadata", {}).get("MinimumAge", "")
    trial_max_age = trial_data.get("metadata", {}).get("MaximumAge", "")
    trial_gender = trial_data.get("metadata", {}).get("Gender", "ALL")
    trial_location = trial_data.get("metadata", {}).get("LocationState", "")

    # Additional matching criteria analysis
    demographic_matches = []
    demographic_concerns = []

    # Age matching
    if patient_age:
        age_match = check_age_eligibility(patient_age, trial_min_age, trial_max_age)
        if age_match["eligible"]:
            demographic_matches.append(f"✅ **Age Eligible**: You are {patient_age} years old, within the trial's age range ({age_match['range']})")
        else:
            demographic_concerns.append(f"⚠️ **Age Question**: You are {patient_age} years old, trial requires {age_match['range']} - discuss with research team")

    # Gender matching
    if patient_gender and trial_gender != "ALL":
        if patient_gender.lower() in trial_gender.lower() or trial_gender.lower() in patient_gender.lower():
            demographic_matches.append(f"✅ **Gender Eligible**: Trial accepts {trial_gender.lower()} participants")
        else:
            demographic_concerns.append(f"⚠️ **Gender Restriction**: Trial is for {trial_gender.lower()} participants only")
    elif trial_gender == "ALL":
        demographic_matches.append("✅ **Gender Eligible**: Trial open to all genders")

    # Condition matching
    condition_matches = []
    for condition in patient_conditions:
        if condition.lower() in trial_condition.lower():
            condition_matches.append(condition)

    if condition_matches:
        demographic_matches.append(f"✅ **Condition Alignment**: Your {', '.join(condition_matches)} aligns with trial focus")

    # Location proximity
    if patient_location and trial_location:
        if patient_location.lower() in trial_location.lower() or trial_location.lower() in patient_location.lower():
            demographic_matches.append(f"✅ **Location Convenient**: Trial location in {trial_location} may be accessible")

    # Display additional matching factors
    if demographic_matches or demographic_concerns:
        st.markdown("#### 📋 Additional Eligibility Factors")

        if demographic_matches:
            for match in demographic_matches:
                st.markdown(match)

        if demographic_concerns:
            st.markdown("**Points to Clarify:**")
            for concern in demographic_concerns:
                st.markdown(concern)

    # Final recommendation based on similarity score
    st.markdown("#### 🎯 Recommendation")

    if primary_score >= 0.7:
        st.success("This trial shows strong semantic alignment with your profile. Consider discussing with your healthcare provider.")
    elif primary_score >= 0.5:
        st.warning("This trial shows moderate alignment. Review the details and discuss with your healthcare team.")
    else:
        st.info("While this trial has lower similarity, it may still be relevant. Consult with medical professionals for personalized advice.")

    st.markdown("---")
    st.caption(f"💡 **How it works**: Our AI analyzes the semantic meaning of trial descriptions and your profile using advanced language models to find conceptual matches beyond keyword searching.")

def _render_content_analysis(patient_info: Dict, trial_data: Dict) -> None:
    """Analyze trial content to show specific matches with patient profile."""

    # Extract patient information
    patient_conditions = patient_info.get("conditions", [])
    patient_medications = patient_info.get("medications", [])
    patient_age = patient_info.get("age")
    patient_gender = patient_info.get("gender", "")

    # Extract trial content fields
    metadata = trial_data.get("metadata", {})
    brief_summary = metadata.get("BriefSummary", "")
    detailed_description = metadata.get("DetailedDescription", "")
    intervention_name = metadata.get("InterventionName", "")
    intervention_description = metadata.get("InterventionDescription", "")
    condition = metadata.get("Condition", "")
    primary_outcome = metadata.get("PrimaryOutcomeMeasure", "")

    # Combine all trial text content for analysis
    trial_content = f"{brief_summary} {detailed_description} {intervention_description}".strip()

    content_matches = []

    # Check for condition matches in trial content
    if patient_conditions and trial_content:
        for patient_condition in patient_conditions:
            condition_lower = patient_condition.lower()
            # Check for direct mentions or related terms
            if condition_lower in trial_content.lower() or condition_lower in condition.lower():
                # Find relevant excerpt
                excerpt = _find_relevant_excerpt(trial_content, patient_condition)
                if excerpt:
                    content_matches.append({
                        "type": "🎯 Condition Match",
                        "patient_item": patient_condition,
                        "trial_content": excerpt,
                        "explanation": f"The trial directly addresses your condition: **{patient_condition}**"
                    })

    # Check for treatment/intervention relevance
    if intervention_name and (patient_conditions or patient_medications):
        intervention_text = f"{intervention_name} {intervention_description}".strip()
        # Look for connections between patient profile and treatment approach
        for condition in patient_conditions:
            if _semantic_overlap(condition, intervention_text):
                content_matches.append({
                    "type": "💊 Treatment Approach",
                    "patient_item": condition,
                    "trial_content": intervention_text[:200] + "..." if len(intervention_text) > 200 else intervention_text,
                    "explanation": f"The trial's treatment approach may be relevant for **{condition}**"
                })

        # Check medication connections
        for medication in patient_medications:
            if medication.lower() in intervention_text.lower():
                content_matches.append({
                    "type": "💊 Medication Connection",
                    "patient_item": medication,
                    "trial_content": intervention_text[:200] + "..." if len(intervention_text) > 200 else intervention_text,
                    "explanation": f"The trial involves **{medication}** which you're currently taking"
                })

    # Check for study objective alignment
    if primary_outcome and patient_conditions:
        for condition in patient_conditions:
            if _semantic_overlap(condition, primary_outcome):
                content_matches.append({
                    "type": "🎯 Study Objective",
                    "patient_item": condition,
                    "trial_content": primary_outcome,
                    "explanation": f"The study's primary goal aligns with outcomes relevant to **{condition}**"
                })

    # Always provide detailed compatibility analysis using available metadata
    compatibility_analysis = _analyze_trial_compatibility(patient_info, trial_data)

    # Display content matches if found
    if content_matches:
        st.markdown("**Here's what our AI found in the trial content:**")

        for i, match in enumerate(content_matches):
            with st.container():
                st.markdown(f"**{match['type']}**")
                st.markdown(match["explanation"])

                if match["trial_content"]:
                    # Clean and format the trial content
                    clean_content = match["trial_content"].replace("\\n", " ").replace("\\t", " ").strip()
                    clean_content = " ".join(clean_content.split())  # Remove extra whitespace

                    # Highlight the matching patient item if possible
                    if match["patient_item"].lower() in clean_content.lower():
                        # Use markdown formatting to highlight
                        highlighted_content = clean_content.replace(
                            match["patient_item"],
                            f"**{match['patient_item']}**"
                        )
                        st.markdown(f"*'{highlighted_content}'*")
                    else:
                        st.markdown(f"*'{clean_content}'*")

                if i < len(content_matches) - 1:  # Add separator except for last item
                    st.markdown("---")

        st.markdown("---")
        st.markdown("**Additional Compatibility Analysis:**")
    else:
        st.markdown("**Here's why this trial was suggested for your profile:**")

    # Display compatibility analysis
    for analysis_item in compatibility_analysis:
        icon = "✅" if analysis_item["compatible"] else "❌" if analysis_item["incompatible"] else "⚠️"
        st.markdown(f"{icon} **{analysis_item['category']}:** {analysis_item['explanation']}")

def _find_relevant_excerpt(text: str, keyword: str, context_words: int = 20) -> str:
    """Find a relevant excerpt from text containing the keyword."""
    if not text or not keyword:
        return ""

    text_lower = text.lower()
    keyword_lower = keyword.lower()

    # Find the keyword position
    pos = text_lower.find(keyword_lower)
    if pos == -1:
        return ""

    # Extract context around the keyword
    words = text.split()
    word_positions = []
    current_pos = 0

    for i, word in enumerate(words):
        if current_pos <= pos < current_pos + len(word):
            # Found the word containing our keyword
            start_idx = max(0, i - context_words)
            end_idx = min(len(words), i + context_words + 1)
            excerpt = " ".join(words[start_idx:end_idx])
            return excerpt
        current_pos += len(word) + 1  # +1 for space

    # Fallback: return first part of text
    return " ".join(text.split()[:context_words * 2])

def _semantic_overlap(term1: str, text: str) -> bool:
    """Check for semantic overlap between a term and text."""
    if not term1 or not text:
        return False

    term1_lower = term1.lower()
    text_lower = text.lower()

    # Direct match
    if term1_lower in text_lower:
        return True

    # Check for word-level matches
    term1_words = set(term1_lower.split())
    text_words = set(text_lower.split())

    # If more than half the words in term1 appear in text
    common_words = term1_words.intersection(text_words)
    if len(common_words) >= len(term1_words) * 0.5 and len(common_words) > 0:
        return True

    return False

def _analyze_trial_compatibility(patient_info: Dict, trial_data: Dict) -> List[Dict]:
    """Analyze compatibility between patient profile and trial using all available metadata."""

    compatibility_analysis = []
    metadata = trial_data.get("metadata", {})

    # Extract patient information
    patient_conditions = patient_info.get("conditions", [])
    patient_age = patient_info.get("age")
    patient_gender = patient_info.get("gender", "").lower()

    # Extract trial information
    trial_condition = metadata.get("Condition", "")
    min_age_str = metadata.get("MinimumAge", "")
    max_age_str = metadata.get("MaximumAge", "")
    trial_gender = metadata.get("Gender", "").lower()
    trial_status = metadata.get("OverallStatus", "")
    trial_phase = metadata.get("Phase", "")
    study_type = metadata.get("StudyType", "")

    # 1. Condition Analysis
    if patient_conditions and trial_condition:
        condition_match_found = False
        for patient_condition in patient_conditions:
            patient_condition_lower = patient_condition.lower()
            trial_condition_lower = trial_condition.lower()

            # Check for direct matches
            if patient_condition_lower in trial_condition_lower or any(word in trial_condition_lower for word in patient_condition_lower.split()):
                compatibility_analysis.append({
                    "category": "Condition Match",
                    "compatible": True,
                    "incompatible": False,
                    "explanation": f"You have {patient_condition} → Trial studies {trial_condition}"
                })
                condition_match_found = True
                break

            # Check for related conditions (diabetes-cardiovascular, cancer types, etc.)
            related_conditions = {
                "diabetes": ["cardiovascular", "heart", "kidney", "diabetic", "metabolic"],
                "cancer": ["oncology", "tumor", "malignant", "carcinoma", "lymphoma", "leukemia"],
                "heart": ["cardiovascular", "cardiac", "hypertension", "blood pressure"],
                "kidney": ["renal", "dialysis", "nephrology"],
                "arthritis": ["joint", "rheumatoid", "osteoarthritis", "musculoskeletal"]
            }

            for condition_key, related_terms in related_conditions.items():
                if condition_key in patient_condition_lower:
                    if any(term in trial_condition_lower for term in related_terms):
                        compatibility_analysis.append({
                            "category": "Related Condition",
                            "compatible": False,
                            "incompatible": False,
                            "explanation": f"You have {patient_condition} → Trial studies {trial_condition} (potentially related condition)"
                        })
                        condition_match_found = True
                        break

        if not condition_match_found:
            compatibility_analysis.append({
                "category": "Condition Mismatch",
                "compatible": False,
                "incompatible": True,
                "explanation": f"You have {', '.join(patient_conditions)} → Trial focuses on {trial_condition}"
            })

    # 2. Age Eligibility Analysis
    if patient_age:
        age_result = check_age_eligibility(patient_age, min_age_str, max_age_str)
        if age_result["eligible"]:
            compatibility_analysis.append({
                "category": "Age Eligible",
                "compatible": True,
                "incompatible": False,
                "explanation": f"You're {patient_age} years old → Trial accepts {age_result['range']}"
            })
        else:
            # Calculate how far off they are
            def parse_age(age_str: str) -> Optional[int]:
                if not age_str:
                    return None
                import re
                match = re.search(r'(\d+)', age_str)
                return int(match.group(1)) if match else None

            min_age = parse_age(min_age_str)
            max_age = parse_age(max_age_str)

            if min_age and patient_age < min_age:
                years_under = min_age - patient_age
                compatibility_analysis.append({
                    "category": "Age Ineligible",
                    "compatible": False,
                    "incompatible": True,
                    "explanation": f"You're {patient_age} years old → Trial requires {age_result['range']} ({years_under} years too young)"
                })
            elif max_age and patient_age > max_age:
                years_over = patient_age - max_age
                compatibility_analysis.append({
                    "category": "Age Ineligible",
                    "compatible": False,
                    "incompatible": True,
                    "explanation": f"You're {patient_age} years old → Trial requires {age_result['range']} ({years_over} years over limit)"
                })

    # 3. Gender Compatibility
    if patient_gender and trial_gender and trial_gender != "all":
        if trial_gender == patient_gender or trial_gender in ["both", "all"]:
            compatibility_analysis.append({
                "category": "Gender Match",
                "compatible": True,
                "incompatible": False,
                "explanation": f"No gender restrictions apply to you"
            })
        else:
            compatibility_analysis.append({
                "category": "Gender Restricted",
                "compatible": False,
                "incompatible": True,
                "explanation": f"You are {patient_gender} → Trial only accepts {trial_gender} participants"
            })

    # 4. Study Status Analysis
    recruiting_statuses = ["recruiting", "not yet recruiting", "enrolling by invitation"]
    if trial_status.lower() in recruiting_statuses:
        compatibility_analysis.append({
            "category": "Study Active",
            "compatible": True,
            "incompatible": False,
            "explanation": f"Trial is currently {trial_status.lower()} participants"
        })
    elif trial_status.lower() in ["completed", "terminated", "withdrawn", "suspended"]:
        compatibility_analysis.append({
            "category": "Study Inactive",
            "compatible": False,
            "incompatible": True,
            "explanation": f"Trial is {trial_status.lower()} - no longer accepting participants"
        })

    # 5. Study Phase Information (educational)
    if trial_phase and study_type.lower() == "interventional":
        phase_explanations = {
            "phase 1": "Early-stage testing of new treatments for safety",
            "phase 2": "Testing effectiveness and optimal dosing",
            "phase 3": "Large-scale comparison with standard treatments",
            "phase 4": "Post-market monitoring of approved treatments"
        }

        phase_lower = trial_phase.lower()
        for phase_key, explanation in phase_explanations.items():
            if phase_key in phase_lower:
                compatibility_analysis.append({
                    "category": "Study Phase",
                    "compatible": False,
                    "incompatible": False,
                    "explanation": f"This is a {trial_phase} study: {explanation}"
                })
                break

    # 6. Study Type Information
    if study_type:
        if study_type.lower() == "interventional":
            compatibility_analysis.append({
                "category": "Study Type",
                "compatible": False,
                "incompatible": False,
                "explanation": "Interventional study: Testing new treatments or procedures"
            })
        elif study_type.lower() == "observational":
            compatibility_analysis.append({
                "category": "Study Type",
                "compatible": False,
                "incompatible": False,
                "explanation": "Observational study: Monitoring health outcomes without new treatments"
            })

    # If no analysis items found, add a default explanation
    if not compatibility_analysis:
        compatibility_analysis.append({
            "category": "Basic Match",
            "compatible": False,
            "incompatible": False,
            "explanation": "This trial was suggested based on semantic similarity to your profile"
        })

    return compatibility_analysis

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