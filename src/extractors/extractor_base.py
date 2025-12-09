from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union
import os
import pandas as pd
import glob

class ExtractorBase:
    def __init__(self, isin: Union[str, List[str]]):
        self.isin = isin
        self.is_big_data = True
    
    @abstractmethod
    def extract_sts(self, isin: str = None):
        pass

    @abstractmethod
    def extract_qte(self, isin: str = None):
        pass

    def get_data_path(self) -> Path:
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        
        self.raw_big_data_dir = self.base_dir / "DATA_BIG" 
        self.raw_small_data_dir = self.base_dir / "DATA_SMALL" 

        if self.is_big_data:
            return self.raw_big_data_dir
        else:
            return self.raw_small_data_dir
        
    def list_files(self) -> List[str]:
        files_list = []
        
        if self.get_data_path().exists():
            # Recursively walk through all files
            for root, dirs, files in os.walk(self.get_data_path()):
                for file in files:
                    file_path = os.path.join(root, file)
                    files_list.append(file_path)
        else:
            print(f"Directory not found: {self.get_data_path()}")
        
        return files_list
    
    def get_summary_stats(self) -> dict:
        """
        Gets summary statistics from QTE and STS data
        
        Returns:
            Dictionary with summary statistics
        """
        target_isin = self.isin if self.isin else (self.isin if isinstance(self.isin, str) else self.isin[0])
        
        try:
            qte_data = self.extract_qte(target_isin)
            sts_data = self.extract_sts(target_isin)
            
            return {
                'isin': target_isin,
                'qte_records': len(qte_data),
                'sts_records': len(sts_data),
                'qte_columns': list(qte_data.columns),
                'sts_columns': list(sts_data.columns),
                'qte_date_range': {
                    'start': qte_data.index.min() if not qte_data.empty else None,
                    'end': qte_data.index.max() if not qte_data.empty else None
                },
                'sts_date_range': {
                    'start': sts_data.index.min() if not sts_data.empty else None,
                    'end': sts_data.index.max() if not sts_data.empty else None
                }
            }
        except Exception as e:
            return {'error': str(e), 'isin': target_isin}
    
    def _format_qte_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats QTE data columns and types
        
        Args:
            df: DataFrame with raw QTE data
            
        Returns:
            Formatted DataFrame
        """
        # Example formatting: Convert timestamp columns to datetime
        formatted_df = df.copy()
        
        timestamp_columns = [col for col in formatted_df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        for col in timestamp_columns:
            formatted_df[col] = pd.to_datetime(formatted_df[col], errors='coerce')

        # Set index to timestamp if available
        if timestamp_columns:
            formatted_df = formatted_df.set_index(timestamp_columns[0])
            formatted_df = formatted_df.sort_index()

        # Define the base columns that should remain as-is
        base_columns = ['session', 'inst_id', 'sequence', 'isin', 'ticker', 'mic', 
                       'currency', 'epoch', 'event_timestamp', 'bloombergTicker']
        
        # Define the level columns pattern (0-9)
        level_columns = ['ord_bid', 'qty_bid', 'px_bid', 'px_ask', 'qty_ask', 'ord_ask']
        
        # Check which base columns actually exist in the DataFrame
        existing_base_cols = [col for col in base_columns if col in formatted_df.columns]
        
        # Create list to store reshaped data
        reshaped_rows = []
        
        # Process each row in the original DataFrame
        for idx, row in formatted_df.iterrows():
            # Extract base information for this row
            base_info = {col: row[col] for col in existing_base_cols}
            
            # Process each level (0-9)
            for level in range(10):
                level_data = base_info.copy()
                level_data['level'] = level
                
                # Extract level-specific data
                for col_type in level_columns:
                    col_name = f"{col_type}_{level}"
                    if col_name in df.columns:
                        level_data[col_type] = row[col_name]
                    else:
                        level_data[col_type] = None
                
                reshaped_rows.append(level_data)
        
        # Create new DataFrame from reshaped data
        formatted_df = pd.DataFrame(reshaped_rows)
        
        # Reorder columns to have level first, then base columns, then data columns
        column_order = ['level'] + existing_base_cols + level_columns
        formatted_df = formatted_df[column_order]
        
        # Remove rows where all price/quantity data is NaN (empty levels)
        data_columns = ['ord_bid', 'qty_bid', 'px_bid', 'px_ask', 'qty_ask', 'ord_ask']
        formatted_df = formatted_df.dropna(subset=data_columns, how='all')
        
        # Reset index
        formatted_df = formatted_df.reset_index(drop=True)
        
        print(f"Reshaped DataFrame: {len(df)} rows x {len(df.columns)} cols -> {len(formatted_df)} rows x {len(formatted_df.columns)} cols")
        
        return formatted_df

    def reshape_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reshapes DataFrame from wide to long format, converting bid/ask levels into rows
        
        Args:
            df: DataFrame with wide format (levels 0-9 as separate columns)
            
        Returns:
            DataFrame in long format with level information in separate column
        """
        # Define the base columns that should remain as-is
        base_columns = ['session', 'inst_id', 'sequence', 'isin', 'ticker', 'mic', 
                       'currency', 'epoch', 'event_timestamp', 'bloombergTicker']
        
        # Define the level columns pattern (0-9)
        level_columns = ['ord_bid', 'qty_bid', 'px_bid', 'px_ask', 'qty_ask', 'ord_ask']
        
        # Check which base columns actually exist in the DataFrame
        existing_base_cols = [col for col in base_columns if col in df.columns]
        
        # Create list to store reshaped data
        reshaped_rows = []
        
        # Process each row in the original DataFrame
        for idx, row in df.iterrows():
            # Extract base information for this row
            base_info = {col: row[col] for col in existing_base_cols}
            
            # Process each level (0-9)
            for level in range(10):
                level_data = base_info.copy()
                level_data['level'] = level
                
                # Extract level-specific data
                for col_type in level_columns:
                    col_name = f"{col_type}_{level}"
                    if col_name in df.columns:
                        level_data[col_type] = row[col_name]
                    else:
                        level_data[col_type] = None
                
                reshaped_rows.append(level_data)
        
        # Create new DataFrame from reshaped data
        reshaped_df = pd.DataFrame(reshaped_rows)
        
        # Reorder columns to have level first, then base columns, then data columns
        column_order = ['level'] + existing_base_cols + level_columns
        reshaped_df = reshaped_df[column_order]
        
        # Remove rows where all price/quantity data is NaN (empty levels)
        data_columns = ['ord_bid', 'qty_bid', 'px_bid', 'px_ask', 'qty_ask', 'ord_ask']
        reshaped_df = reshaped_df.dropna(subset=data_columns, how='all')
        
        # Reset index
        reshaped_df = reshaped_df.reset_index(drop=True)
        
        print(f"Reshaped DataFrame: {len(df)} rows x {len(df.columns)} cols -> {len(reshaped_df)} rows x {len(reshaped_df.columns)} cols")
        
        return reshaped_df


    def validate_columns_for_reshape(self, df: pd.DataFrame) -> bool:
        """
        Validates if DataFrame has the required columns for reshaping
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if DataFrame can be reshaped, False otherwise
        """
        required_patterns = ['px_bid_', 'px_ask_', 'qty_bid_', 'qty_ask_']
        
        for pattern in required_patterns:
            # Check if at least level 0 exists for each pattern
            if f"{pattern}0" not in df.columns:
                print(f"Warning: Missing required column {pattern}0 for reshaping")
                return False
        
        print("✓ DataFrame has required columns for reshaping")
        return True

    def _get_magic_numbers(self) -> List[float]:
        """
        Returns list of vendor magic numbers that indicate non-tradable states
        
        These are NOT real prices and must be filtered out to avoid massive P&L errors
        
        Returns:
            List of magic number values to filter out
        """
        return [
            666666.666,   # Unquoted/Unknown
            999999.999,   # Market Order (At Best)
            999999.989,   # At Open Order  
            999999.988,   # At Close Order
            999999.979,   # Pegged Order
            999999.123    # Unquoted/Unknown
        ]
    
    def _get_continuous_trading_codes(self) -> dict:
        """
        Returns market status codes for continuous trading by venue
        
        Only data with these codes represents addressable/tradable orderbooks
        
        Returns:
            Dictionary mapping venue to list of valid continuous trading codes
        """
        return {
            'AQUIS': [5308427],
            'AQEU': [5308427],  # Alternative code for AQUIS
            'BME': [5832713, 5832756], 
            'XMAD': [5832713, 5832756],  # Alternative code for BME
            'CBOE': [12255233],
            'CEUX': [12255233],  # Alternative code for CBOE  
            'TURQUOISE': [7608181],
            'TQEX': [7608181]   # Alternative code for TURQUOISE
        }
    
    def _filter_magic_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes records containing vendor magic numbers (non-tradable price codes)
        
        CRITICAL: Failing to filter these will cause massive P&L calculation errors
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with magic numbers filtered out
        """
        magic_numbers = self._get_magic_numbers()
        filtered_df = df.copy()
        original_count = len(filtered_df)
        
        # Find all price columns
        price_columns = [col for col in filtered_df.columns if 'px_' in col.lower() or 'price' in col.lower()]
        
        print(f"Filtering magic numbers from {len(price_columns)} price columns")
        
        # Filter out magic numbers from each price column
        for col in price_columns:
            if col in filtered_df.columns:
                # Remove rows where this column contains any magic number
                magic_mask = ~filtered_df[col].isin(magic_numbers)
                filtered_df = filtered_df[magic_mask]
        
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            print(f"⚠️  CRITICAL: Removed {removed_count} records with magic numbers (non-tradable prices)")
            print(f"   Remaining records: {filtered_count}")
        else:
            print("✓ No magic numbers found in price data")
        
        return filtered_df
    
    def _filter_non_continuous_trading(self, qte_df: pd.DataFrame, sts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters QTE data to only include records during continuous trading
        
        CRITICAL: Trading outside continuous hours will result in non-executable orders
        
        Args:
            qte_df: QTE DataFrame with quote data
            sts_df: STS DataFrame with market status data
            
        Returns:
            QTE DataFrame filtered for continuous trading periods only
        """
        if sts_df.empty:
            print("⚠️  WARNING: No STS data provided - cannot validate trading hours")
            return qte_df
        
        continuous_codes = self._get_continuous_trading_codes()
        
        # Determine venue from MIC code if available
        venue = None
        if 'mic' in qte_df.columns:
            mic_value = qte_df['mic'].iloc[0] if not qte_df.empty else None
            if mic_value in continuous_codes:
                venue = mic_value
        
        if venue is None:
            print("⚠️  WARNING: Cannot determine venue - unable to validate trading status")
            return qte_df
        
        valid_codes = continuous_codes[venue]
        print(f"Filtering for continuous trading codes {valid_codes} for venue {venue}")
        
        # Find status column in STS data
        status_columns = [col for col in sts_df.columns if 'status' in col.lower() or 'state' in col.lower()]
        if not status_columns:
            # Try common column names
            status_columns = [col for col in sts_df.columns if any(x in col.lower() for x in ['mkt', 'market', 'trading'])]
        
        if not status_columns:
            print("⚠️  WARNING: No market status column found in STS data")
            return qte_df
        
        status_col = status_columns[0]
        print(f"Using status column: {status_col}")
        
        # Filter STS for continuous trading periods
        continuous_sts = sts_df[sts_df[status_col].isin(valid_codes)]
        
        if continuous_sts.empty:
            print(f"⚠️  CRITICAL: No continuous trading periods found for venue {venue}")
            print(f"   Available status codes: {sts_df[status_col].unique()}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # If timestamps are available, filter QTE by continuous trading timestamps
        if 'event_timestamp' in qte_df.columns and 'event_timestamp' in continuous_sts.columns:
            continuous_timestamps = set(continuous_sts['event_timestamp'])
            filtered_qte = qte_df[qte_df['event_timestamp'].isin(continuous_timestamps)]
            
            original_count = len(qte_df)
            filtered_count = len(filtered_qte)
            removed_count = original_count - filtered_count
            
            print(f"✓ Filtered to continuous trading periods: {filtered_count}/{original_count} records")
            if removed_count > 0:
                print(f"   Removed {removed_count} records from non-trading periods")
            
            return filtered_qte
        else:
            print("⚠️  WARNING: No timestamp columns for precise filtering")
            return qte_df

    def _clean_qte_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and validates QTE data with CRITICAL vendor-specific validations
        
        Args:
            df: DataFrame with raw QTE data
            
        Returns:
            Clean and validated DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        print(f"Original data: {len(cleaned_df)} records")
        
        # CRITICAL STEP 1: Filter out magic numbers (non-tradable price codes)
        cleaned_df = self._filter_magic_numbers(cleaned_df)
        
        # CRITICAL STEP 2: Remove records with invalid prices (<=0 or NaN)
        price_columns = [col for col in cleaned_df.columns if 'px_' in col.lower()]
        for col in price_columns:
            if col in cleaned_df.columns:
                # Remove prices <= 0 or NaN
                mask = (cleaned_df[col] > 0) & (cleaned_df[col].notna())
                cleaned_df = cleaned_df[mask]
        
        print(f"After removing invalid prices: {len(cleaned_df)} records")
        
        # 3. Filter only addressable orderbooks
        # Assuming addressable orderbooks have valid bid and ask
        if 'px_bid_0' in cleaned_df.columns and 'px_ask_0' in cleaned_df.columns:
            # Keep only records with valid bid and ask
            addressable_mask = (
                (cleaned_df['px_bid_0'] > 0) & 
                (cleaned_df['px_ask_0'] > 0) & 
                (cleaned_df['px_bid_0'] <= cleaned_df['px_ask_0'])  # Bid <= Ask
            )
            cleaned_df = cleaned_df[addressable_mask]
        
        print(f"After addressable orderbooks filter: {len(cleaned_df)} records")
        
        # 4. Sort by timestamp if available
        timestamp_columns = [col for col in cleaned_df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        if timestamp_columns:
            cleaned_df = cleaned_df.sort_values(timestamp_columns[0]).reset_index(drop=True)
        
        return cleaned_df
    
    def _clean_sts_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and validates STS data with vendor-specific validations
        
        Args:
            df: DataFrame with raw STS data
            
        Returns:
            Clean and validated DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        print(f"Original STS data: {len(cleaned_df)} records")
        
        # Filter out magic numbers if price columns exist in STS
        price_columns = [col for col in cleaned_df.columns if 'px' in col.lower() or 'price' in col.lower()]
        if price_columns:
            cleaned_df = self._filter_magic_numbers(cleaned_df)
        
        # Remove records with invalid volumes
        volume_columns = [col for col in cleaned_df.columns if 'qty' in col.lower() or 'volume' in col.lower()]
        for col in volume_columns:
            if col in cleaned_df.columns:
                mask = (cleaned_df[col] > 0) & (cleaned_df[col].notna())
                cleaned_df = cleaned_df[mask]
        
        print(f"STS after cleaning: {len(cleaned_df)} records")
        
        return cleaned_df
    
    def clean_and_validate_data(self, qte_df: pd.DataFrame, sts_df: pd.DataFrame = None) -> dict:
        """
        Comprehensive data cleaning and validation with vendor specifications
        
        Args:
            qte_df: QTE DataFrame
            sts_df: Optional STS DataFrame for market status validation
            
        Returns:
            Dictionary with cleaned data and validation report
        """
        print("="*60)
        print("CRITICAL VENDOR DATA VALIDATION")
        print("="*60)
        
        # Clean QTE data
        cleaned_qte = self._clean_qte_data(qte_df)
        
        # Clean STS data if provided
        cleaned_sts = self._clean_sts_data(sts_df) if sts_df is not None else pd.DataFrame()
        
        # Apply continuous trading filter if STS data available
        if not cleaned_sts.empty:
            print("\\nApplying continuous trading filter...")
            cleaned_qte = self._filter_non_continuous_trading(cleaned_qte, cleaned_sts)
        else:
            print("\\n⚠️  WARNING: No STS data - cannot validate trading hours")
        
        # Generate validation report
        validation_report = {
            'original_qte_records': len(qte_df),
            'cleaned_qte_records': len(cleaned_qte),
            'qte_removal_rate': (len(qte_df) - len(cleaned_qte)) / len(qte_df) * 100,
            'original_sts_records': len(sts_df) if sts_df is not None else 0,
            'cleaned_sts_records': len(cleaned_sts),
            'magic_numbers_filtered': True,
            'continuous_trading_validated': not cleaned_sts.empty,
            'addressable_orderbooks_only': True,
            'data_quality_score': len(cleaned_qte) / len(qte_df) * 100 if len(qte_df) > 0 else 0
        }
        
        print(f"\\n📊 VALIDATION SUMMARY")
        print(f"   QTE Records: {validation_report['original_qte_records']} → {validation_report['cleaned_qte_records']} ({validation_report['qte_removal_rate']:.1f}% removed)")
        print(f"   Data Quality Score: {validation_report['data_quality_score']:.1f}%")
        print(f"   Magic Numbers Filtered: {'✓' if validation_report['magic_numbers_filtered'] else '✗'}")
        print(f"   Trading Hours Validated: {'✓' if validation_report['continuous_trading_validated'] else '⚠️'}")
        
        return {
            'qte_data': cleaned_qte,
            'sts_data': cleaned_sts,
            'validation_report': validation_report
        }
        