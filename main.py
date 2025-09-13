"""Main script for ingesting clinical trials data from ClinicalTrials.gov."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    CONDITIONS,
    LOG_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_TRIALS_PER_CONDITION,
)
from data_ingestion import (
    ClinicalTrialsAPIClient,
    ClinicalTrialsParser,
    DataStorage,
)


def setup_logging():
    """Set up logging configuration."""
    log_filename = LOG_DIR / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def ingest_clinical_trials(
    conditions: list = None,
    max_per_condition: int = None,
    test_mode: bool = False
):
    """
    Main function to ingest clinical trials data.
    
    Args:
        conditions: List of conditions to search (uses config default if None)
        max_per_condition: Max trials per condition (uses config default if None)
        test_mode: If True, only fetch a small sample for testing
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting Clinical Trials Data Ingestion")
    logger.info("=" * 60)
    
    # Use defaults from config if not provided
    if conditions is None:
        conditions = CONDITIONS
    if max_per_condition is None:
        max_per_condition = MAX_TRIALS_PER_CONDITION
    
    # Test mode settings
    if test_mode:
        logger.info("Running in TEST MODE - fetching limited data")
        conditions = conditions[:2]  # Only first 2 conditions
        max_per_condition = 10  # Only 10 trials per condition
    
    # Initialize components
    api_client = ClinicalTrialsAPIClient()
    parser = ClinicalTrialsParser()
    storage = DataStorage()
    
    # Test API connection
    logger.info("Testing API connection...")
    if not api_client.test_connection():
        logger.error("Failed to connect to ClinicalTrials.gov API")
        return False
    
    logger.info("API connection successful")
    
    # Collect data for each condition
    all_trials_data = []
    conditions_summary = {}
    
    for i, condition in enumerate(conditions, 1):
        logger.info(f"\n[{i}/{len(conditions)}] Processing condition: {condition}")
        logger.info("-" * 40)
        
        try:
            # Fetch trials from API
            raw_studies = api_client.search_studies(
                condition=condition,
                max_results=max_per_condition
            )
            
            if not raw_studies:
                logger.warning(f"No studies found for {condition}")
                conditions_summary[condition] = 0
                continue
            
            # Save raw response
            storage.save_raw_response(raw_studies, condition)
            
            # Parse the studies
            logger.info(f"Parsing {len(raw_studies)} studies for {condition}...")
            parsed_df = parser.parse_studies(raw_studies)
            
            # Add condition tag to help with filtering
            parsed_df["SearchCondition"] = condition
            
            # Append to main collection
            all_trials_data.append(parsed_df)
            conditions_summary[condition] = len(parsed_df)
            
            logger.info(f"Successfully processed {len(parsed_df)} trials for {condition}")
            
        except Exception as e:
            logger.error(f"Error processing {condition}: {e}")
            conditions_summary[condition] = 0
            continue
    
    # Combine all data
    if not all_trials_data:
        logger.error("No data collected from any condition")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("Combining and saving all data...")
    
    # Combine DataFrames
    combined_df = pd.concat(all_trials_data, ignore_index=True)
    
    # Remove duplicates based on NCT ID
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["NCTId"], keep="first")
    duplicates_removed = initial_count - len(combined_df)
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate trials")
    
    # Save processed data in multiple formats
    logger.info(f"Saving {len(combined_df)} unique trials...")
    saved_files = storage.save_processed_data(combined_df, "clinical_trials")
    
    # Save summary statistics
    storage.save_summary_stats(combined_df, conditions_summary)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total unique trials collected: {len(combined_df)}")
    logger.info("\nBreakdown by condition:")
    
    for condition, count in conditions_summary.items():
        logger.info(f"  - {condition}: {count} trials")
    
    logger.info(f"\nData saved to:")
    for format_type, filepath in saved_files.items():
        logger.info(f"  - {format_type}: {filepath}")
    
    # Data quality summary
    logger.info("\nData Quality Metrics:")
    for col in ["NCTId", "BriefTitle", "Condition", "EligibilityCriteria"]:
        non_empty = ((combined_df[col].notna()) & (combined_df[col] != "")).sum()
        pct = (non_empty / len(combined_df)) * 100
        logger.info(f"  - {col}: {pct:.1f}% complete")
    
    logger.info("\n" + "=" * 60)
    logger.info("Data ingestion completed successfully!")
    logger.info("=" * 60)
    
    return True


def main():
    """Command-line interface for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest clinical trials data from ClinicalTrials.gov"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited data"
    )
    
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Specific conditions to search (e.g., cancer diabetes)"
    )
    
    parser.add_argument(
        "--max-trials",
        type=int,
        help="Maximum trials per condition"
    )
    
    args = parser.parse_args()
    
    # Run ingestion
    success = ingest_clinical_trials(
        conditions=args.conditions,
        max_per_condition=args.max_trials,
        test_mode=args.test
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()