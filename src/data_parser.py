"""Parser for ClinicalTrials.gov API responses."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalTrialsParser:
    """Parser for extracting and cleaning clinical trial data."""
    
    @staticmethod
    def safe_get(data: Dict, path: str, default: Any = None) -> Any:
        """
        Safely extract nested values from dictionary.
        
        Args:
            data: Dictionary to extract from
            path: Dot-separated path to the value
            default: Default value if path not found
            
        Returns:
            Extracted value or default
        """
        try:
            keys = path.split(".")
            value = data
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and value:
                    # Handle list access
                    if key.isdigit():
                        idx = int(key)
                        value = value[idx] if idx < len(value) else None
                    else:
                        # Extract key from all items in list
                        value = [item.get(key) if isinstance(item, dict) else None 
                                for item in value]
                else:
                    return default
                    
                if value is None:
                    return default
            
            return value
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    
    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """
        Clean text by removing extra whitespace and HTML tags.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def parse_study(self, study: Dict) -> Dict:
        """
        Parse a single study record.
        
        Args:
            study: Raw study data from API
            
        Returns:
            Parsed and cleaned study data
        """
        parsed = {}
        protocol_section = study.get("protocolSection", {})
        
        # Identification
        id_module = protocol_section.get("identificationModule", {})
        parsed["NCTId"] = id_module.get("nctId", "")
        parsed["BriefTitle"] = self.clean_text(id_module.get("briefTitle"))
        parsed["OfficialTitle"] = self.clean_text(id_module.get("officialTitle"))
        
        # Status
        status_module = protocol_section.get("statusModule", {})
        parsed["OverallStatus"] = status_module.get("overallStatus", "")
        parsed["StartDate"] = self._parse_date(status_module.get("startDateStruct"))
        parsed["PrimaryCompletionDate"] = self._parse_date(
            status_module.get("primaryCompletionDateStruct")
        )
        parsed["CompletionDate"] = self._parse_date(
            status_module.get("completionDateStruct")
        )
        parsed["StudyFirstPostDate"] = self._parse_date(
            status_module.get("studyFirstPostDateStruct")
        )
        parsed["LastUpdatePostDate"] = self._parse_date(
            status_module.get("lastUpdatePostDateStruct")
        )
        
        # Design
        design_module = protocol_section.get("designModule", {})
        parsed["StudyType"] = design_module.get("studyType", "")
        parsed["Phase"] = self._parse_phases(design_module.get("phases", []))
        parsed["EnrollmentCount"] = design_module.get("enrollmentInfo", {}).get("count")
        parsed["EnrollmentType"] = design_module.get("enrollmentInfo", {}).get("type")
        
        # Conditions
        conditions_module = protocol_section.get("conditionsModule", {})
        parsed["Condition"] = self._join_list(conditions_module.get("conditions", []))
        
        # Interventions
        arms_module = protocol_section.get("armsInterventionsModule", {})
        interventions = arms_module.get("interventions", [])
        parsed["InterventionName"] = self._parse_interventions(interventions, "name")
        parsed["InterventionType"] = self._parse_interventions(interventions, "type")
        parsed["InterventionDescription"] = self._parse_interventions(
            interventions, "description"
        )
        
        # Description
        desc_module = protocol_section.get("descriptionModule", {})
        parsed["BriefSummary"] = self.clean_text(desc_module.get("briefSummary"))
        parsed["DetailedDescription"] = self.clean_text(
            desc_module.get("detailedDescription")
        )
        
        # Eligibility
        eligibility = protocol_section.get("eligibilityModule", {})
        parsed["EligibilityCriteria"] = self.clean_text(
            eligibility.get("eligibilityCriteria")
        )
        parsed["MinimumAge"] = eligibility.get("minimumAge", "")
        parsed["MaximumAge"] = eligibility.get("maximumAge", "")
        parsed["Gender"] = eligibility.get("sex", "")
        parsed["HealthyVolunteers"] = eligibility.get("healthyVolunteers", "")
        
        # Outcomes
        outcomes_module = protocol_section.get("outcomesModule", {})
        parsed.update(self._parse_outcomes(outcomes_module))
        
        # Contacts and Locations
        locations_module = protocol_section.get("contactsLocationsModule", {})
        parsed.update(self._parse_central_contacts(locations_module))
        parsed.update(self._parse_locations(locations_module))
        
        # Sponsors
        sponsor_module = protocol_section.get("sponsorCollaboratorsModule", {})
        parsed.update(self._parse_sponsors(sponsor_module))
        
        return parsed
    
    def _parse_date(self, date_struct: Optional[Dict]) -> str:
        """Parse date structure to string."""
        if not date_struct:
            return ""
        
        date_str = date_struct.get("date", "")
        date_type = date_struct.get("type", "")
        
        if date_type == "ESTIMATED":
            date_str = f"~{date_str}"
        
        return date_str
    
    def _parse_phases(self, phases: List[str]) -> str:
        """Parse phase list to string."""
        if not phases:
            return ""
        
        # Clean phase names
        cleaned_phases = []
        for phase in phases:
            if phase and phase != "NA":
                cleaned_phases.append(phase.replace("PHASE", "Phase "))
        
        return "; ".join(cleaned_phases)
    
    def _parse_interventions(
        self, 
        interventions: List[Dict], 
        field: str
    ) -> str:
        """Parse intervention field from list."""
        if not interventions:
            return ""
        
        values = []
        for intervention in interventions:
            value = intervention.get(field, "")
            if value:
                values.append(self.clean_text(value))
        
        return "; ".join(values)
    
    def _parse_outcomes(self, outcomes_module: Dict) -> Dict:
        """Parse outcome measures."""
        parsed = {}
        
        # Primary outcomes
        primary_outcomes = outcomes_module.get("primaryOutcomes", [])
        if primary_outcomes:
            measures = []
            descriptions = []
            timeframes = []
            
            for outcome in primary_outcomes:
                measures.append(self.clean_text(outcome.get("measure", "")))
                descriptions.append(self.clean_text(outcome.get("description", "")))
                timeframes.append(outcome.get("timeFrame", ""))
            
            parsed["PrimaryOutcomeMeasure"] = "; ".join(measures)
            parsed["PrimaryOutcomeDescription"] = "; ".join(descriptions)
            parsed["PrimaryOutcomeTimeFrame"] = "; ".join(timeframes)
        else:
            parsed["PrimaryOutcomeMeasure"] = ""
            parsed["PrimaryOutcomeDescription"] = ""
            parsed["PrimaryOutcomeTimeFrame"] = ""
        
        # Secondary outcomes
        secondary_outcomes = outcomes_module.get("secondaryOutcomes", [])
        if secondary_outcomes:
            measures = []
            descriptions = []
            timeframes = []
            
            for outcome in secondary_outcomes:
                measures.append(self.clean_text(outcome.get("measure", "")))
                descriptions.append(self.clean_text(outcome.get("description", "")))
                timeframes.append(outcome.get("timeFrame", ""))
            
            parsed["SecondaryOutcomeMeasure"] = "; ".join(measures)
            parsed["SecondaryOutcomeDescription"] = "; ".join(descriptions)
            parsed["SecondaryOutcomeTimeFrame"] = "; ".join(timeframes)
        else:
            parsed["SecondaryOutcomeMeasure"] = ""
            parsed["SecondaryOutcomeDescription"] = ""
            parsed["SecondaryOutcomeTimeFrame"] = ""
        
        return parsed
    
    def _parse_locations(self, locations_module: Dict) -> Dict:
        """Parse location information."""
        parsed = {
            "LocationCity": "",
            "LocationState": "",
            "LocationCountry": "",
            "LocationFacility": "",
            "LocationStatus": "",
            "LocationContactName": "",
            "LocationContactRole": "",
            "LocationContactPhone": "",
            "LocationContactEmail": ""
        }

        locations = locations_module.get("locations", [])
        if not locations:
            return parsed

        cities = []
        states = []
        countries = []
        facilities = []
        statuses = []
        contact_names = []
        contact_roles = []
        contact_phones = []
        contact_emails = []

        for location in locations:
            cities.append(location.get("city", ""))
            states.append(location.get("state", ""))
            countries.append(location.get("country", ""))
            facilities.append(location.get("facility", ""))
            statuses.append(location.get("status", ""))

            # Extract contact information from location
            location_contacts = location.get("contacts", [])
            if location_contacts:
                # Use the first contact for this location
                contact = location_contacts[0]
                contact_names.append(contact.get("name", ""))
                contact_roles.append(contact.get("role", ""))
                contact_phones.append(self._format_phone(
                    contact.get("phone", ""),
                    contact.get("phoneExt", "")
                ))
                contact_emails.append(contact.get("email", ""))
            else:
                # No contact information for this location
                contact_names.append("")
                contact_roles.append("")
                contact_phones.append("")
                contact_emails.append("")

        parsed["LocationCity"] = "; ".join(filter(None, cities))
        parsed["LocationState"] = "; ".join(filter(None, states))
        parsed["LocationCountry"] = "; ".join(filter(None, countries))
        parsed["LocationFacility"] = "; ".join(filter(None, facilities))
        parsed["LocationStatus"] = "; ".join(filter(None, statuses))
        parsed["LocationContactName"] = "; ".join(filter(None, contact_names))
        parsed["LocationContactRole"] = "; ".join(filter(None, contact_roles))
        parsed["LocationContactPhone"] = "; ".join(filter(None, contact_phones))
        parsed["LocationContactEmail"] = "; ".join(filter(None, contact_emails))

        return parsed
    
    def _parse_sponsors(self, sponsor_module: Dict) -> Dict:
        """Parse sponsor information."""
        parsed = {
            "ResponsiblePartyType": "",
            "LeadSponsorName": "",
            "LeadSponsorClass": "",
            "CollaboratorName": "",
            "CollaboratorClass": ""
        }
        
        # Responsible party
        responsible_party = sponsor_module.get("responsibleParty", {})
        parsed["ResponsiblePartyType"] = responsible_party.get("type", "")
        
        # Lead sponsor
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        parsed["LeadSponsorName"] = lead_sponsor.get("name", "")
        parsed["LeadSponsorClass"] = lead_sponsor.get("class", "")
        
        # Collaborators
        collaborators = sponsor_module.get("collaborators", [])
        if collaborators:
            names = []
            classes = []
            
            for collab in collaborators:
                names.append(collab.get("name", ""))
                classes.append(collab.get("class", ""))
            
            parsed["CollaboratorName"] = "; ".join(filter(None, names))
            parsed["CollaboratorClass"] = "; ".join(filter(None, classes))
        
        return parsed

    def _parse_central_contacts(self, contacts_locations_module: Dict) -> Dict:
        """Parse central contact information."""
        parsed = {
            "CentralContactName": "",
            "CentralContactRole": "",
            "CentralContactPhone": "",
            "CentralContactEmail": ""
        }

        central_contacts = contacts_locations_module.get("centralContacts", [])
        if not central_contacts:
            return parsed

        # Extract information from the primary central contact
        primary_contact = central_contacts[0]

        parsed["CentralContactName"] = primary_contact.get("name", "")
        parsed["CentralContactRole"] = primary_contact.get("role", "")
        parsed["CentralContactPhone"] = self._format_phone(
            primary_contact.get("phone", ""),
            primary_contact.get("phoneExt", "")
        )
        parsed["CentralContactEmail"] = primary_contact.get("email", "")

        return parsed

    def _format_phone(self, phone: str, ext: str) -> str:
        """Format phone number with extension."""
        if not phone:
            return ""

        formatted_phone = phone
        if ext:
            formatted_phone += f" ext. {ext}"

        return formatted_phone

    def _join_list(self, items: List[str]) -> str:
        """Join list items with semicolon."""
        if not items:
            return ""
        return "; ".join(filter(None, items))
    
    def parse_studies(self, studies: List[Dict]) -> pd.DataFrame:
        """
        Parse multiple study records into DataFrame.
        
        Args:
            studies: List of raw study data
            
        Returns:
            DataFrame with parsed study data
        """
        parsed_studies = []
        
        for study in studies:
            try:
                parsed = self.parse_study(study)
                parsed_studies.append(parsed)
            except Exception as e:
                logger.error(f"Error parsing study: {e}")
                continue
        
        df = pd.DataFrame(parsed_studies)
        
        # Ensure all required columns exist
        from config import ALL_FIELDS
        for field in ALL_FIELDS:
            if field not in df.columns:
                df[field] = ""

        # Reorder columns
        df = df[ALL_FIELDS]
        
        logger.info(f"Parsed {len(parsed_studies)} studies successfully")
        
        return df