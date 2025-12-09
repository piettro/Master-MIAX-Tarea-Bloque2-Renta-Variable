"""
Consolidated Tape Integration Module

This module provides integration between the consolidated tape system and existing extractors.
It implements the Step 2 functionality with real data integration.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.models.consolidated_tape import ConsolidatedTape, ConsolidatedTapeFactory
from src.extractors.extractor_aquis import AquisExtractor

class ConsolidatedTapeManager:
    """
    Manager class for handling consolidated tape operations with real market data
    """
    
    def __init__(self, isin: str):
        self.isin = isin
        self.tape: Optional[ConsolidatedTape] = None
        self.extractors: Dict[str, object] = {}
        
    def add_venue_extractor(self, venue: str, extractor_class, **kwargs):
        """
        Add venue extractor for data loading
        
        Args:
            venue: Venue name
            extractor_class: Extractor class to use
            **kwargs: Additional arguments for extractor initialization
        """
        try:
            extractor = extractor_class(isin=self.isin, **kwargs)
            self.extractors[venue] = extractor
            print(f"✅ Added {venue} extractor")
        except Exception as e:
            print(f"❌ Failed to add {venue} extractor: {e}")
    
    def create_consolidated_tape(self, use_validated_data: bool = True) -> ConsolidatedTape:
        """
        Create consolidated tape with data from all configured extractors
        
        Args:
            use_validated_data: Whether to use validated data (recommended)
            
        Returns:
            ConsolidatedTape instance with loaded data
        """
        if not self.extractors:
            raise ValueError("No extractors configured. Add venue extractors first.")
        
        print(f"🏗️ Creating consolidated tape for {self.isin}...")
        self.tape = ConsolidatedTapeFactory.create_tape(self.isin)
        
        successful_venues = []
        failed_venues = []
        
        for venue, extractor in self.extractors.items():
            try:
                print(f"\\n📊 Loading data for {venue}...")
                
                if use_validated_data and hasattr(extractor, 'extract_validated_data'):
                    # Use validated data extraction if available
                    result = extractor.extract_validated_data(self.isin)
                    qte_data = result['qte_data']
                    
                    # Display validation results
                    report = result['validation_report']
                    print(f"   Validation: {report['data_quality_score']:.1f}% data quality")
                    print(f"   Records: {report['original_qte_records']} → {report['cleaned_qte_records']}")
                else:
                    # Fallback to regular QTE extraction
                    qte_data = extractor.extract_qte(self.isin)
                
                if qte_data.empty:
                    print(f"⚠️ No data available for {venue}")
                    failed_venues.append(venue)
                    continue
                
                # Add to consolidated tape
                self.tape.add_venue_data(venue, qte_data)
                successful_venues.append(venue)
                
            except Exception as e:
                print(f"❌ Failed to load {venue} data: {e}")
                failed_venues.append(venue)
        
        print(f"\\n📊 LOADING SUMMARY:")
        print(f"   Successful venues: {successful_venues}")
        if failed_venues:
            print(f"   Failed venues: {failed_venues}")
        
        if not successful_venues:
            raise ValueError("No venue data could be loaded")
        
        return self.tape
    
    def build_and_analyze(self, 
                         resample_freq: str = None,
                         min_arbitrage_bps: float = 1.0) -> Dict:
        """
        Build consolidated tape and perform arbitrage analysis
        
        Args:
            resample_freq: Resampling frequency (e.g., '1S', '100ms')
            min_arbitrage_bps: Minimum arbitrage profit in basis points
            
        Returns:
            Dictionary with analysis results
        """
        if self.tape is None:
            raise ValueError("Must create consolidated tape first")
        
        print(f"\\n🔧 Building consolidated tape...")
        consolidated_df = self.tape.build_tape(resample_freq=resample_freq)
        
        print(f"\\n🚨 Detecting arbitrage opportunities...")
        arbitrage_ops = self.tape.get_arbitrage_opportunities(min_profit_bps=min_arbitrage_bps)
        
        # Generate comprehensive analysis
        analysis = {
            'consolidated_data': consolidated_df,
            'arbitrage_opportunities': arbitrage_ops,
            'venue_summary': self.tape.get_venue_summary(),
            'metrics': self._calculate_metrics(consolidated_df, arbitrage_ops)
        }
        
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _calculate_metrics(self, consolidated_df: pd.DataFrame, arbitrage_ops: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive metrics from consolidated data
        
        Args:
            consolidated_df: Consolidated tape DataFrame
            arbitrage_ops: Arbitrage opportunities DataFrame
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {
            'total_timestamps': len(consolidated_df),
            'arbitrage_opportunities': len(arbitrage_ops),
            'arbitrage_rate_pct': len(arbitrage_ops) / len(consolidated_df) * 100 if len(consolidated_df) > 0 else 0,
        }
        
        if not arbitrage_ops.empty:
            metrics.update({
                'total_arbitrage_profit': arbitrage_ops['arbitrage_profit'].sum(),
                'avg_arbitrage_profit': arbitrage_ops['arbitrage_profit'].mean(),
                'max_arbitrage_profit': arbitrage_ops['arbitrage_profit'].max(),
                'avg_arbitrage_bps': arbitrage_ops['profit_bps'].mean() if 'profit_bps' in arbitrage_ops.columns else 0,
                'max_arbitrage_bps': arbitrage_ops['profit_bps'].max() if 'profit_bps' in arbitrage_ops.columns else 0
            })
        
        # Calculate venue statistics
        if 'best_bid_venue' in consolidated_df.columns:
            metrics['best_bid_venue_distribution'] = consolidated_df['best_bid_venue'].value_counts().to_dict()
        
        if 'best_ask_venue' in consolidated_df.columns:
            metrics['best_ask_venue_distribution'] = consolidated_df['best_ask_venue'].value_counts().to_dict()
        
        # Calculate spread statistics
        spread_cols = [col for col in consolidated_df.columns if '_spread_bps' in col and 'overall' not in col]
        if spread_cols:
            metrics['venue_spreads'] = {}
            for col in spread_cols:
                venue = col.replace('_spread_bps', '')
                metrics['venue_spreads'][venue] = {
                    'mean': consolidated_df[col].mean(),
                    'std': consolidated_df[col].std(),
                    'min': consolidated_df[col].min(),
                    'max': consolidated_df[col].max()
                }
        
        return metrics
    
    def _print_analysis_summary(self, analysis: Dict):
        """
        Print comprehensive analysis summary
        
        Args:
            analysis: Analysis results dictionary
        """
        metrics = analysis['metrics']
        
        print(f"\\n📊 CONSOLIDATED TAPE ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total timestamps: {metrics['total_timestamps']:,}")
        print(f"Arbitrage opportunities: {metrics['arbitrage_opportunities']:,}")
        print(f"Arbitrage rate: {metrics['arbitrage_rate_pct']:.2f}%")
        
        if metrics['arbitrage_opportunities'] > 0:
            print(f"\\n💰 ARBITRAGE PROFIT ANALYSIS:")
            print(f"Total potential profit: €{metrics.get('total_arbitrage_profit', 0):.4f}")
            print(f"Average profit per opportunity: €{metrics.get('avg_arbitrage_profit', 0):.4f}")
            print(f"Maximum single opportunity: €{metrics.get('max_arbitrage_profit', 0):.4f}")
            print(f"Average profit: {metrics.get('avg_arbitrage_bps', 0):.2f} basis points")
            print(f"Maximum profit: {metrics.get('max_arbitrage_bps', 0):.2f} basis points")
        
        if 'best_bid_venue_distribution' in metrics:
            print(f"\\n🏆 BEST BID VENUE LEADERSHIP:")
            for venue, count in metrics['best_bid_venue_distribution'].items():
                pct = count / metrics['total_timestamps'] * 100
                print(f"   {venue}: {pct:.1f}% ({count:,} times)")
        
        if 'best_ask_venue_distribution' in metrics:
            print(f"\\n🎯 BEST ASK VENUE LEADERSHIP:")
            for venue, count in metrics['best_ask_venue_distribution'].items():
                pct = count / metrics['total_timestamps'] * 100
                print(f"   {venue}: {pct:.1f}% ({count:,} times)")
        
        if 'venue_spreads' in metrics:
            print(f"\\n📈 VENUE SPREAD ANALYSIS (basis points):")
            for venue, stats in metrics['venue_spreads'].items():
                print(f"   {venue}: Mean={stats['mean']:.2f}, Range={stats['min']:.2f}-{stats['max']:.2f}")
    
    def export_analysis(self, filepath: str, analysis: Dict = None):
        """
        Export consolidated tape analysis to file
        
        Args:
            filepath: Output file path
            analysis: Analysis results (if None, will build first)
        """
        if analysis is None:
            if self.tape is None:
                raise ValueError("No tape or analysis available to export")
            analysis = self.build_and_analyze()
        
        if self.tape:
            self.tape.export_tape(filepath, include_arbitrage=True)
        
        # Also export metrics as JSON if requested
        if filepath.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                # Convert numpy types for JSON serialization
                json_metrics = self._convert_for_json(analysis['metrics'])
                json.dump(json_metrics, f, indent=2, default=str)
            print(f"✅ Metrics exported to {filepath}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj

# Convenience function for quick analysis
def quick_arbitrage_analysis(isin: str, venues: List[str] = None) -> Dict:
    """
    Quick arbitrage analysis for an ISIN across specified venues
    
    Args:
        isin: ISIN code to analyze
        venues: List of venue names (defaults to ['AQUIS'] if None)
        
    Returns:
        Analysis results dictionary
    """
    if venues is None:
        venues = ['AQUIS']  # Default to available extractor
    
    manager = ConsolidatedTapeManager(isin)
    
    # Add extractors for specified venues
    for venue in venues:
        if venue.upper() == 'AQUIS':
            manager.add_venue_extractor('AQUIS', AquisExtractor)
        else:
            print(f"⚠️ Extractor for {venue} not implemented yet")
    
    # Create tape and analyze
    manager.create_consolidated_tape(use_validated_data=True)
    analysis = manager.build_and_analyze(resample_freq='1S', min_arbitrage_bps=0.5)
    
    return analysis