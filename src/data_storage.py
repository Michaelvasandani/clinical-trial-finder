"""Data storage module for saving clinical trials data."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import (
    OUTPUT_FORMATS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


class DataStorage:
    """Handle data storage operations for clinical trials data."""
    
    def __init__(self):
        """Initialize data storage."""
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_raw_response(
        self, 
        data: List[Dict], 
        condition: str
    ) -> Path:
        """
        Save raw API response to JSON file.
        
        Args:
            data: Raw API response data
            condition: Condition name for filename
            
        Returns:
            Path to saved file
        """
        filename = f"{condition.replace(' ', '_')}_raw_{self.timestamp}.json"
        filepath = self.raw_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved raw data to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
            raise
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename_prefix: str = "clinical_trials",
        formats: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Save processed data in multiple formats.
        
        Args:
            df: DataFrame with processed data
            filename_prefix: Prefix for output files
            formats: List of output formats (csv, json, parquet)
            
        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = OUTPUT_FORMATS
        
        saved_files = {}
        
        for format_type in formats:
            try:
                if format_type == "csv":
                    filepath = self._save_csv(df, filename_prefix)
                elif format_type == "json":
                    filepath = self._save_json(df, filename_prefix)
                elif format_type == "parquet":
                    filepath = self._save_parquet(df, filename_prefix)
                else:
                    logger.warning(f"Unknown format: {format_type}")
                    continue
                
                saved_files[format_type] = filepath
                logger.info(f"Saved {format_type} to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving {format_type}: {e}")
        
        return saved_files
    
    def _save_csv(self, df: pd.DataFrame, prefix: str) -> Path:
        """Save DataFrame to CSV."""
        filename = f"{prefix}_{self.timestamp}.csv"
        filepath = self.processed_dir / filename
        
        df.to_csv(filepath, index=False, encoding="utf-8")
        return filepath
    
    def _save_json(self, df: pd.DataFrame, prefix: str) -> Path:
        """Save DataFrame to JSON."""
        filename = f"{prefix}_{self.timestamp}.json"
        filepath = self.processed_dir / filename
        
        df.to_json(
            filepath,
            orient="records",
            indent=2,
            force_ascii=False,
            date_format="iso"
        )
        return filepath
    
    def _save_parquet(self, df: pd.DataFrame, prefix: str) -> Path:
        """Save DataFrame to Parquet."""
        filename = f"{prefix}_{self.timestamp}.parquet"
        filepath = self.processed_dir / filename
        
        df.to_parquet(filepath, index=False, engine="pyarrow")
        return filepath
    
    def save_summary_stats(
        self,
        df: pd.DataFrame,
        conditions_data: Dict[str, int]
    ) -> Path:
        """
        Save summary statistics about the ingested data.
        
        Args:
            df: Complete DataFrame with all trials
            conditions_data: Dictionary mapping conditions to trial counts
            
        Returns:
            Path to summary file
        """
        summary = {
            "timestamp": self.timestamp,
            "total_trials": len(df),
            "conditions_breakdown": conditions_data,
            "unique_nct_ids": df["NCTId"].nunique(),
            "status_distribution": df["OverallStatus"].value_counts().to_dict(),
            "phase_distribution": df["Phase"].value_counts().to_dict(),
            "study_type_distribution": df["StudyType"].value_counts().to_dict(),
            "countries": self._extract_country_stats(df),
            "missing_data_report": self._generate_missing_data_report(df),
            "data_quality_score": self._calculate_quality_score(df)
        }
        
        filename = f"summary_stats_{self.timestamp}.json"
        filepath = self.processed_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary statistics to {filepath}")
        return filepath
    
    def _extract_country_stats(self, df: pd.DataFrame) -> Dict:
        """Extract country statistics from location data."""
        countries = []
        
        for country_list in df["LocationCountry"].dropna():
            if country_list:
                countries.extend([c.strip() for c in country_list.split(";")])
        
        country_counts = pd.Series(countries).value_counts()
        
        return {
            "total_countries": len(country_counts),
            "top_10_countries": country_counts.head(10).to_dict()
        }
    
    def _generate_missing_data_report(self, df: pd.DataFrame) -> Dict:
        """Generate report on missing data."""
        missing_report = {}
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            empty_count = (df[column] == "").sum()
            total_missing = missing_count + empty_count
            
            if total_missing > 0:
                missing_report[column] = {
                    "missing_count": int(total_missing),
                    "missing_percentage": round(total_missing / len(df) * 100, 2)
                }
        
        return missing_report
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score based on completeness.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Quality score between 0 and 100
        """
        critical_fields = [
            "NCTId", "BriefTitle", "OverallStatus", "Condition",
            "BriefSummary", "EligibilityCriteria"
        ]
        
        scores = []
        for field in critical_fields:
            if field in df.columns:
                non_empty = ((df[field].notna()) & (df[field] != "")).sum()
                score = non_empty / len(df) * 100
                scores.append(score)
        
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    
    def load_processed_data(
        self,
        format_type: str = "parquet",
        latest: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load previously saved processed data.
        
        Args:
            format_type: Format to load (csv, json, parquet)
            latest: Whether to load the most recent file
            
        Returns:
            DataFrame or None if no files found
        """
        pattern = f"clinical_trials_*.{format_type}"
        files = list(self.processed_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No {format_type} files found")
            return None
        
        if latest:
            filepath = max(files, key=lambda p: p.stat().st_mtime)
        else:
            filepath = files[0]
        
        try:
            if format_type == "csv":
                df = pd.read_csv(filepath)
            elif format_type == "json":
                df = pd.read_json(filepath)
            elif format_type == "parquet":
                df = pd.read_parquet(filepath)
            else:
                logger.error(f"Unknown format: {format_type}")
                return None
            
            logger.info(f"Loaded data from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None