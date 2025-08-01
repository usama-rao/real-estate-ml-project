"""
Data cleaning module for real estate dataset.
Handles missing values, outliers, and data quality issues.

Author: Usama Rao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstateDataCleaner:
    """
    Comprehensive data cleaning for real estate datasets.
    """
    
    def __init__(self, data_path=None):
        """Initialize the data cleaner."""
        self.data_path = data_path
        self.raw_data = None
        self.cleaned_data = None
        self.cleaning_report = {}
    
    def load_data(self, data/raw/kc_house_data.csv):
        """Load raw data from file."""
        try:
            self.raw_data = pd.read_csv(data/raw/kc_house_data.csv)
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report."""
        if self.raw_data is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        report = {}
        
        # Basic info
        report['total_rows'] = len(self.raw_data)
        report['total_columns'] = len(self.raw_data.columns)
        
        # Missing values analysis
        missing_values = self.raw_data.isnull().sum()
        report['missing_values'] = missing_values[missing_values > 0].to_dict()
        report['missing_percentage'] = (missing_values / len(self.raw_data) * 100).round(2).to_dict()
        
        # Data types
        report['data_types'] = self.raw_data.dtypes.to_dict()
        
        # Duplicates
        report['duplicate_rows'] = self.raw_data.duplicated().sum()
        
        # Basic statistics for numeric columns
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        report['numeric_columns'] = list(numeric_cols)
        
        self.cleaning_report = report
        return report
    
    def print_data_quality_report(self):
        """Print formatted data quality report."""
        if not self.cleaning_report:
            self.generate_data_quality_report()
        
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        
        print(f"Dataset Shape: {self.cleaning_report['total_rows']} rows x {self.cleaning_report['total_columns']} columns")
        print(f"Duplicate Rows: {self.cleaning_report['duplicate_rows']}")
        
        if self.cleaning_report['missing_values']:
            print("\nMISSING VALUES:")
            for col, count in self.cleaning_report['missing_values'].items():
                percentage = self.cleaning_report['missing_percentage'][col]
                print(f"  {col}: {count} ({percentage}%)")
        else:
            print("\nNo missing values found!")
        
        print(f"\nNUMERIC COLUMNS: {len(self.cleaning_report['numeric_columns'])}")
        for col in self.cleaning_report['numeric_columns']:
            print(f"  - {col}")

    def handle_missing_values(self, strategy='default'):
        """Handle missing values based on column type and business logic."""
        if self.raw_data is None:
            logger.error("No data loaded.")
            return None
        
        # Create copy for cleaning
        self.cleaned_data = self.raw_data.copy()
        initial_rows = len(self.cleaned_data)
        
        logger.info(f"Starting missing value handling with strategy: {strategy}")
        
        if strategy == 'default':
            # Custom logic for real estate data
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.cleaned_data.select_dtypes(include=['object']).columns
            
            # Fill numeric with median
            for col in numeric_cols:
                if self.cleaned_data[col].isnull().sum() > 0:
                    median_val = self.cleaned_data[col].median()
                    self.cleaned_data[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled {col} missing values with median: {median_val}")
            
            # Fill categorical with mode
            for col in categorical_cols:
                if self.cleaned_data[col].isnull().sum() > 0:
                    mode_val = self.cleaned_data[col].mode()[0] if len(self.cleaned_data[col].mode()) > 0 else 'Unknown'
                    self.cleaned_data[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled {col} missing values with mode: {mode_val}")
        
        elif strategy == 'drop':
            # Drop rows with any missing values
            self.cleaned_data.dropna(inplace=True)
            logger.info(f"Dropped rows with missing values. Rows remaining: {len(self.cleaned_data)}")
        
        final_rows = len(self.cleaned_data)
        logger.info(f"Missing value handling complete. Rows: {initial_rows} -> {final_rows}")
        
        return self.cleaned_data

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        if self.cleaned_data is None:
            logger.error("No cleaned data available. Run handle_missing_values first.")
            return None
        
        initial_rows = len(self.cleaned_data)
        self.cleaned_data.drop_duplicates(inplace=True)
        final_rows = len(self.cleaned_data)
        
        removed = initial_rows - final_rows
        logger.info(f"Removed {removed} duplicate rows. Rows remaining: {final_rows}")
        
        return self.cleaned_data

    def detect_outliers(self, method='iqr', columns=None):
        """Detect outliers in numeric columns."""
        if self.cleaned_data is None:
            logger.error("No cleaned data available.")
            return None
        
        if columns is None:
            columns = self.cleaned_data.select_dtypes(include=[np.number]).columns
        
        outliers_info = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.cleaned_data[col].quantile(0.25)
                Q3 = self.cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.cleaned_data[(self.cleaned_data[col] < lower_bound) | 
                                           (self.cleaned_data[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((self.cleaned_data[col] - self.cleaned_data[col].mean()) / self.cleaned_data[col].std())
                outliers = self.cleaned_data[z_scores > 3]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.cleaned_data) * 100,
                'indices': outliers.index.tolist()
            }
            
            logger.info(f"{col}: {len(outliers)} outliers ({outliers_info[col]['percentage']:.2f}%)")
        
        return outliers_info

    def save_cleaned_data(self, data/raw/kc_house_data.csv):
        """Save cleaned data to file."""
        if self.cleaned_data is None:
            logger.error("No cleaned data to save.")
            return False
        
        try:
            # Create directory if it doesn't exist
            Path(data/raw/kc_house_data.csv).parent.mkdir(parents=True, exist_ok=True)
            
            self.cleaned_data.to_csv(data/raw/kc_house_data.csv, index=False)
            logger.info(f"Cleaned data saved to: {data/raw/kc_house_data.csv}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False

    def clean_data_pipeline(self, input_path, output_path, missing_strategy='default', 
                           remove_outliers=False, outlier_method='iqr'):
        """Complete data cleaning pipeline."""
        logger.info("Starting data cleaning pipeline...")
        
        # Step 1: Load data
        if not self.load_data(input_path):
            return False
        
        # Step 2: Generate initial report
        self.generate_data_quality_report()
        print("INITIAL DATA QUALITY:")
        self.print_data_quality_report()
        
        # Step 3: Handle missing values
        self.handle_missing_values(strategy=missing_strategy)
        
        # Step 4: Remove duplicates
        self.remove_duplicates()
        
        # Step 5: Handle outliers (optional)
        if remove_outliers:
            outliers_info = self.detect_outliers(method=outlier_method)
            # Remove outliers (implement based on business rules)
            logger.info("Outlier removal not implemented - requires business logic")
        else:
            outliers_info = self.detect_outliers(method=outlier_method)
        
        # Step 6: Final validation
        final_missing = self.cleaned_data.isnull().sum().sum()
        logger.info(f"Final missing values: {final_missing}")
        
        # Step 7: Save cleaned data
        success = self.save_cleaned_data(output_path)
        
        if success:
            logger.info("Data cleaning pipeline completed successfully!")
            print(f"\nFINAL CLEANED DATA:")
            print(f"Shape: {self.cleaned_data.shape}")
            print(f"Missing values: {final_missing}")
            print(f"Saved to: {output_path}")
        
        return success


if __name__ == "__main__":
    # Example usage of complete pipeline
    cleaner = RealEstateDataCleaner()
    
    # PLACEHOLDERS - Update these paths with your actual data files
    input_file = "input_file = "data/raw/kc_house_data.csv"  # UPDATE THIS
    output_file = "data/processed/cleaned_housing_data.csv"
    
    # Run complete pipeline
    success = cleaner.clean_data_pipeline(
        input_path=input_file,
        output_path=output_file,
        missing_strategy='default',
        remove_outliers=False
    )
    
    if success:
        print("Data cleaning completed successfully!")
    else:
        print("Data cleaning failed!")
    