"""
Consolidated Tape Implementation

This module implements a comprehensive consolidated tape system for cross-venue arbitrage detection.
It follows object-oriented design principles and best practices.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class ConsolidatedTapeBase(ABC):
    """
    Abstract base class for consolidated tape implementations
    """
    
    @abstractmethod
    def add_venue_data(self, venue: str, qte_data: pd.DataFrame) -> None:
        """Add venue data to the consolidated tape"""
        pass
    
    @abstractmethod
    def build_tape(self) -> pd.DataFrame:
        """Build the consolidated tape DataFrame"""
        pass
    
    @abstractmethod
    def get_arbitrage_opportunities(self) -> pd.DataFrame:
        """Detect arbitrage opportunities across venues"""
        pass

class VenueData:
    """
    Encapsulates venue-specific market data and operations
    """
    
    def __init__(self, venue_name: str, mic_code: str):
        self.venue_name = venue_name
        self.mic_code = mic_code
        self.qte_data: Optional[pd.DataFrame] = None
        self.best_quotes: Optional[pd.DataFrame] = None
    
    def set_data(self, qte_data: pd.DataFrame) -> None:
        """
        Set the QTE data for this venue
        
        Args:
            qte_data: Clean QTE DataFrame with timestamp index
        """
        if qte_data.empty:
            raise ValueError(f"Empty data provided for venue {self.venue_name}")
        
        # Ensure timestamp is the index
        if 'event_timestamp' in qte_data.columns:
            qte_data = qte_data.set_index('event_timestamp')
        elif not isinstance(qte_data.index, pd.DatetimeIndex):
            raise ValueError(f"Venue {self.venue_name}: Data must have datetime index or event_timestamp column")
        
        # Sort by timestamp
        qte_data = qte_data.sort_index()
        
        self.qte_data = qte_data
        self._extract_best_quotes()
    
    def _extract_best_quotes(self) -> None:
        """
        Extract best bid and ask prices from QTE data
        """
        if self.qte_data is None:
            raise ValueError(f"No data set for venue {self.venue_name}")
        
        # Extract best bid and ask (level 0)
        best_quotes = pd.DataFrame(index=self.qte_data.index)
        
        # Best bid and ask prices
        if 'px_bid_0' in self.qte_data.columns:
            best_quotes[f'{self.venue_name}_best_bid'] = self.qte_data['px_bid_0']
        
        if 'px_ask_0' in self.qte_data.columns:
            best_quotes[f'{self.venue_name}_best_ask'] = self.qte_data['px_ask_0']
        
        # Best bid and ask quantities
        if 'qty_bid_0' in self.qte_data.columns:
            best_quotes[f'{self.venue_name}_bid_qty'] = self.qte_data['qty_bid_0']
        
        if 'qty_ask_0' in self.qte_data.columns:
            best_quotes[f'{self.venue_name}_ask_qty'] = self.qte_data['qty_ask_0']
        
        # Calculate spread
        if f'{self.venue_name}_best_bid' in best_quotes.columns and f'{self.venue_name}_best_ask' in best_quotes.columns:
            best_quotes[f'{self.venue_name}_spread'] = (
                best_quotes[f'{self.venue_name}_best_ask'] - best_quotes[f'{self.venue_name}_best_bid']
            )
            
            # Calculate spread in basis points
            best_quotes[f'{self.venue_name}_spread_bps'] = (
                best_quotes[f'{self.venue_name}_spread'] / 
                best_quotes[f'{self.venue_name}_best_bid'] * 10000
            )
        
        # Remove invalid quotes (NaN or zero prices)
        price_columns = [f'{self.venue_name}_best_bid', f'{self.venue_name}_best_ask']
        for col in price_columns:
            if col in best_quotes.columns:
                best_quotes = best_quotes[
                    (best_quotes[col] > 0) & 
                    (best_quotes[col].notna())
                ]
        
        self.best_quotes = best_quotes
        print(f"✓ Extracted {len(self.best_quotes)} best quotes for {self.venue_name}")
    
    def get_best_quotes(self) -> pd.DataFrame:
        """
        Get the best quotes DataFrame
        
        Returns:
            DataFrame with best bid/ask data
        """
        if self.best_quotes is None:
            raise ValueError(f"No best quotes available for venue {self.venue_name}")
        
        return self.best_quotes.copy()
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics for this venue's data
        
        Returns:
            Dictionary with summary information
        """
        if self.best_quotes is None:
            return {'venue': self.venue_name, 'status': 'No data'}
        
        bid_col = f'{self.venue_name}_best_bid'
        ask_col = f'{self.venue_name}_best_ask'
        
        summary = {
            'venue': self.venue_name,
            'mic_code': self.mic_code,
            'records': len(self.best_quotes),
            'time_range': {
                'start': self.best_quotes.index.min(),
                'end': self.best_quotes.index.max()
            }
        }
        
        if bid_col in self.best_quotes.columns:
            summary['best_bid_stats'] = {
                'min': self.best_quotes[bid_col].min(),
                'max': self.best_quotes[bid_col].max(),
                'mean': self.best_quotes[bid_col].mean()
            }
        
        if ask_col in self.best_quotes.columns:
            summary['best_ask_stats'] = {
                'min': self.best_quotes[ask_col].min(),
                'max': self.best_quotes[ask_col].max(),
                'mean': self.best_quotes[ask_col].mean()
            }
        
        return summary

class ConsolidatedTape(ConsolidatedTapeBase):
    """
    Main consolidated tape implementation for cross-venue price comparison
    """
    
    def __init__(self, isin: str):
        self.isin = isin
        self.venues: Dict[str, VenueData] = {}
        self.consolidated_data: Optional[pd.DataFrame] = None
        self.arbitrage_opportunities: Optional[pd.DataFrame] = None
    
    def add_venue_data(self, venue: str, qte_data: pd.DataFrame, mic_code: str = None) -> None:
        """
        Add venue data to the consolidated tape
        
        Args:
            venue: Venue name (e.g., 'BME', 'AQUIS', 'CBOE')
            qte_data: Clean QTE DataFrame for this venue
            mic_code: Optional MIC code for the venue
        """
        if venue in self.venues:
            print(f"⚠️ Overwriting existing data for venue {venue}")
        
        mic_code = mic_code or self._get_default_mic_code(venue)
        venue_data = VenueData(venue, mic_code)
        
        try:
            venue_data.set_data(qte_data)
            self.venues[venue] = venue_data
            print(f"✓ Added venue data for {venue} ({mic_code})")
            
            # Reset consolidated data since we have new venue data
            self.consolidated_data = None
            self.arbitrage_opportunities = None
            
        except Exception as e:
            print(f"❌ Failed to add venue data for {venue}: {e}")
            raise
    
    def _get_default_mic_code(self, venue: str) -> str:
        """
        Get default MIC code for venue
        
        Args:
            venue: Venue name
            
        Returns:
            Default MIC code
        """
        default_mics = {
            'BME': 'XMAD',
            'AQUIS': 'AQEU', 
            'CBOE': 'CEUX',
            'TURQUOISE': 'TQEX'
        }
        return default_mics.get(venue.upper(), f'{venue}_MIC')
    
    def build_tape(self, resample_freq: str = None) -> pd.DataFrame:
        """
        Build the consolidated tape DataFrame
        
        Args:
            resample_freq: Optional resampling frequency (e.g., '1S', '100ms')
            
        Returns:
            Consolidated DataFrame with timestamp index and venue columns
        """
        if not self.venues:
            raise ValueError("No venue data added to consolidated tape")
        
        print(f"🔧 Building consolidated tape for {len(self.venues)} venues...")
        
        # Collect all venue best quotes
        venue_quotes = []
        for venue_name, venue_data in self.venues.items():
            best_quotes = venue_data.get_best_quotes()
            venue_quotes.append(best_quotes)
        
        # Combine all venue data using outer join (union of all timestamps)
        consolidated = pd.concat(venue_quotes, axis=1, join='outer', sort=True)
        
        # Forward fill missing values (use last known price)
        consolidated = consolidated.fillna(method='ffill')
        
        # Apply resampling if requested
        if resample_freq:
            print(f"📊 Resampling to {resample_freq} frequency...")
            consolidated = consolidated.resample(resample_freq).last()
            consolidated = consolidated.dropna()
        
        # Add consolidated metrics
        consolidated = self._add_consolidated_metrics(consolidated)
        
        self.consolidated_data = consolidated
        print(f"✅ Consolidated tape built: {len(consolidated)} timestamps, {len(consolidated.columns)} columns")
        
        return consolidated.copy()
    
    def _add_consolidated_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-venue metrics to consolidated data
        
        Args:
            df: Consolidated DataFrame
            
        Returns:
            DataFrame with additional metrics
        """
        # Find best overall bid and ask across all venues
        bid_columns = [col for col in df.columns if '_best_bid' in col]
        ask_columns = [col for col in df.columns if '_best_ask' in col]
        
        if bid_columns:
            # Best bid is the highest bid across all venues
            df['best_bid_overall'] = df[bid_columns].max(axis=1)
            df['best_bid_venue'] = df[bid_columns].idxmax(axis=1).str.replace('_best_bid', '')
        
        if ask_columns:
            # Best ask is the lowest ask across all venues
            df['best_ask_overall'] = df[ask_columns].min(axis=1)
            df['best_ask_venue'] = df[ask_columns].idxmin(axis=1).str.replace('_best_ask', '')
        
        if bid_columns and ask_columns:
            # Overall spread
            df['overall_spread'] = df['best_ask_overall'] - df['best_bid_overall']
            df['overall_spread_bps'] = (df['overall_spread'] / df['best_bid_overall'] * 10000)
            
            # Arbitrage flag: when best bid > best ask (inverted spread)
            df['arbitrage_opportunity'] = df['best_bid_overall'] > df['best_ask_overall']
            
            # Potential arbitrage profit per share
            df['arbitrage_profit'] = np.where(
                df['arbitrage_opportunity'],
                df['best_bid_overall'] - df['best_ask_overall'],
                0
            )
        
        return df
    
    def get_arbitrage_opportunities(self, min_profit_bps: float = 1.0) -> pd.DataFrame:
        """
        Detect arbitrage opportunities across venues
        
        Args:
            min_profit_bps: Minimum profit in basis points to consider as arbitrage
            
        Returns:
            DataFrame with arbitrage opportunities
        """
        if self.consolidated_data is None:
            raise ValueError("Must build consolidated tape first")
        
        arbitrage_data = self.consolidated_data[
            self.consolidated_data['arbitrage_opportunity'] == True
        ].copy()
        
        if arbitrage_data.empty:
            print("📊 No arbitrage opportunities found")
            return pd.DataFrame()
        
        # Calculate profit in basis points
        arbitrage_data['profit_bps'] = (
            arbitrage_data['arbitrage_profit'] / 
            arbitrage_data['best_ask_overall'] * 10000
        )
        
        # Filter by minimum profit threshold
        significant_arbitrage = arbitrage_data[
            arbitrage_data['profit_bps'] >= min_profit_bps
        ]
        
        if significant_arbitrage.empty:
            print(f"📊 No significant arbitrage opportunities (>{min_profit_bps} bps)")
            return pd.DataFrame()
        
        # Select relevant columns for arbitrage analysis
        arbitrage_cols = [
            'best_bid_overall', 'best_bid_venue',
            'best_ask_overall', 'best_ask_venue', 
            'arbitrage_profit', 'profit_bps'
        ]
        
        # Add venue-specific prices for context
        bid_cols = [col for col in arbitrage_data.columns if '_best_bid' in col and 'overall' not in col]
        ask_cols = [col for col in arbitrage_data.columns if '_best_ask' in col and 'overall' not in col]
        
        result_cols = arbitrage_cols + bid_cols + ask_cols
        available_cols = [col for col in result_cols if col in arbitrage_data.columns]
        
        self.arbitrage_opportunities = significant_arbitrage[available_cols]
        
        print(f"🚨 Found {len(significant_arbitrage)} arbitrage opportunities (>{min_profit_bps} bps)")
        
        return self.arbitrage_opportunities.copy()
    
    def get_venue_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all venues
        
        Returns:
            DataFrame with venue statistics
        """
        summaries = []
        for venue_data in self.venues.values():
            summary = venue_data.get_data_summary()
            summaries.append(summary)
        
        # Convert to DataFrame for better display
        summary_df = pd.DataFrame(summaries)
        return summary_df
    
    def export_tape(self, filepath: str, include_arbitrage: bool = True) -> None:
        """
        Export consolidated tape to file
        
        Args:
            filepath: Output file path
            include_arbitrage: Whether to include arbitrage opportunities sheet
        """
        if self.consolidated_data is None:
            raise ValueError("No consolidated data to export")
        
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.csv':
            self.consolidated_data.to_csv(filepath)
            print(f"✅ Consolidated tape exported to {filepath}")
        
        elif filepath.suffix.lower() in ['.xlsx', '.xls']:
            with pd.ExcelWriter(filepath) as writer:
                self.consolidated_data.to_excel(writer, sheet_name='Consolidated_Tape')
                
                if include_arbitrage and self.arbitrage_opportunities is not None:
                    self.arbitrage_opportunities.to_excel(writer, sheet_name='Arbitrage_Opportunities')
                
                # Add venue summary
                venue_summary = self.get_venue_summary()
                venue_summary.to_excel(writer, sheet_name='Venue_Summary', index=False)
            
            print(f"✅ Consolidated tape exported to {filepath} (multiple sheets)")
        
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

class ConsolidatedTapeFactory:
    """
    Factory class for creating consolidated tape instances
    """
    
    @staticmethod
    def create_tape(isin: str) -> ConsolidatedTape:
        """
        Create a new consolidated tape instance
        
        Args:
            isin: ISIN code for the instrument
            
        Returns:
            ConsolidatedTape instance
        """
        return ConsolidatedTape(isin)
    
    @staticmethod
    def create_from_extractors(isin: str, extractors: Dict[str, object]) -> ConsolidatedTape:
        """
        Create consolidated tape from extractor instances
        
        Args:
            isin: ISIN code
            extractors: Dictionary mapping venue names to extractor instances
            
        Returns:
            ConsolidatedTape with data loaded from extractors
        """
        tape = ConsolidatedTape(isin)
        
        for venue_name, extractor in extractors.items():
            try:
                # Extract QTE data using the extractor
                qte_data = extractor.extract_qte(isin)
                
                # Add to consolidated tape
                tape.add_venue_data(venue_name, qte_data)
                
            except Exception as e:
                print(f"⚠️ Failed to load data for venue {venue_name}: {e}")
        
        return tape