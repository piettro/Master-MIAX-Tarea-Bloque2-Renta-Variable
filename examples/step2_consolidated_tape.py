"""
Step 2 Implementation Example: Consolidated Tape for Arbitrage Detection

This script demonstrates a complete Step 2 implementation using simulated data
and standalone implementation for cross-venue arbitrage detection.
No external extractors required - complete standalone implementation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

class StandaloneConsolidatedTapeManager:
    """
    Standalone consolidated tape manager with no external dependencies
    Creates market data simulation and performs arbitrage analysis
    """
    
    def __init__(self, isin: str, venues: List[str] = None):
        self.isin = isin
        self.venues = venues or ['AQUIS', 'BME', 'CBOE', 'TURQUOISE']
        self.venue_data = {}
        self.consolidated_tape = None
        
        # Market simulation parameters
        self.base_price = 100.0
        self.volatility = 0.02
        self.venue_spreads = {
            'AQUIS': 0.01,
            'BME': 0.015,
            'CBOE': 0.012,
            'TURQUOISE': 0.008
        }
        
    def generate_venue_data(self, num_points: int = 10000, freq_seconds: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Generate realistic market data for multiple venues
        
        Args:
            num_points: Number of data points to generate
            freq_seconds: Frequency in seconds between updates
            
        Returns:
            Dictionary of venue DataFrames with market data
        """
        print(f"🎲 Generating market data for {len(self.venues)} venues...")
        
        # Create time index
        start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        time_index = [start_time + timedelta(seconds=i*freq_seconds) for i in range(num_points)]
        
        # Generate correlated price movements
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0, self.volatility/np.sqrt(252*24*60*60/freq_seconds), num_points)
        price_path = self.base_price * np.exp(np.cumsum(returns))
        
        venue_data = {}
        
        for venue in self.venues:
            # Add venue-specific noise and latency effects
            venue_noise = np.random.normal(0, 0.001, num_points)
            venue_prices = price_path + venue_noise
            
            # Calculate bid/ask with venue-specific spreads
            spread = self.venue_spreads.get(venue, 0.01)
            half_spread = spread / 2
            
            bids = venue_prices - half_spread
            asks = venue_prices + half_spread
            
            # Add realistic quantities
            bid_qtys = np.random.randint(1000, 10000, num_points)
            ask_qtys = np.random.randint(1000, 10000, num_points)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': time_index,
                'isin': self.isin,
                'bid': bids,
                'ask': asks,
                'bid_qty': bid_qtys,
                'ask_qty': ask_qtys,
                'venue': venue
            })
            
            df.set_index('timestamp', inplace=True)
            venue_data[venue] = df
            
            print(f"   ✅ {venue}: {len(df)} records generated")
        
        self.venue_data = venue_data
        return venue_data
    
    def create_consolidated_tape(self, resample_freq: str = '1S') -> pd.DataFrame:
        """
        Create consolidated tape from venue data
        
        Args:
            resample_freq: Resampling frequency (e.g., '1S' for 1 second)
            
        Returns:
            Consolidated DataFrame with all venue data aligned
        """
        print(f"🔧 Creating consolidated tape with {resample_freq} resolution...")
        
        if not self.venue_data:
            self.generate_venue_data()
        
        consolidated_data = {}
        
        # Process each venue
        for venue, df in self.venue_data.items():
            # Resample to consistent frequency
            resampled = df.resample(resample_freq).last().fillna(method='ffill')
            
            # Rename columns with venue prefix
            for col in ['bid', 'ask', 'bid_qty', 'ask_qty']:
                if col in resampled.columns:
                    consolidated_data[f'{venue}_{col}'] = resampled[col]
        
        # Create consolidated DataFrame
        self.consolidated_tape = pd.DataFrame(consolidated_data)
        
        # Add ISIN column
        self.consolidated_tape['isin'] = self.isin
        
        # Forward fill to handle any missing data
        self.consolidated_tape = self.consolidated_tape.fillna(method='ffill').fillna(method='bfill')
        
        print(f"   ✅ Consolidated tape created: {self.consolidated_tape.shape}")
        
        return self.consolidated_tape
    
    def detect_arbitrage_opportunities(self, min_profit_bps: float = 0.5) -> pd.DataFrame:
        """
        Detect arbitrage opportunities in consolidated tape
        
        Args:
            min_profit_bps: Minimum profit in basis points
            
        Returns:
            DataFrame with arbitrage opportunities
        """
        print(f"🚨 Detecting arbitrage opportunities (min: {min_profit_bps} bps)...")
        
        if self.consolidated_tape is None:
            self.create_consolidated_tape()
        
        opportunities = []
        
        for timestamp, row in self.consolidated_tape.iterrows():
            # Find best bid and ask across venues
            bids = {}
            asks = {}
            
            for venue in self.venues:
                bid_col = f'{venue}_bid'
                ask_col = f'{venue}_ask'
                
                if bid_col in row.index and ask_col in row.index:
                    bid_price = row[bid_col]
                    ask_price = row[ask_col]
                    
                    if pd.notna(bid_price) and pd.notna(ask_price) and bid_price > 0 and ask_price > 0:
                        bids[venue] = bid_price
                        asks[venue] = ask_price
            
            if len(bids) >= 2 and len(asks) >= 2:
                # Find best bid and ask
                best_bid_venue = max(bids.items(), key=lambda x: x[1])
                best_ask_venue = min(asks.items(), key=lambda x: x[1])
                
                best_bid_price = best_bid_venue[1]
                best_ask_price = best_ask_venue[1]
                best_bid_venue_name = best_bid_venue[0]
                best_ask_venue_name = best_ask_venue[0]
                
                # Check for arbitrage opportunity
                if (best_bid_price > best_ask_price and 
                    best_bid_venue_name != best_ask_venue_name):
                    
                    profit_absolute = best_bid_price - best_ask_price
                    profit_bps = (profit_absolute / best_ask_price) * 10000
                    
                    if profit_bps >= min_profit_bps:
                        # Calculate maximum tradeable quantity
                        bid_qty = row.get(f'{best_bid_venue_name}_bid_qty', 0)
                        ask_qty = row.get(f'{best_ask_venue_name}_ask_qty', 0)
                        max_qty = min(bid_qty, ask_qty)
                        
                        opportunity = {
                            'timestamp': timestamp,
                            'best_bid_venue': best_bid_venue_name,
                            'best_ask_venue': best_ask_venue_name,
                            'best_bid_price': best_bid_price,
                            'best_ask_price': best_ask_price,
                            'arbitrage_profit': profit_absolute,
                            'profit_bps': profit_bps,
                            'max_quantity': max_qty,
                            'total_profit_potential': profit_absolute * max_qty
                        }
                        
                        opportunities.append(opportunity)
        
        arbitrage_df = pd.DataFrame(opportunities)
        if not arbitrage_df.empty:
            arbitrage_df.set_index('timestamp', inplace=True)
        
        print(f"   🎯 Found {len(opportunities)} arbitrage opportunities")
        
        return arbitrage_df
    
    def calculate_metrics(self, arbitrage_ops: pd.DataFrame) -> Dict:
        """
        Calculate analysis metrics
        
        Args:
            arbitrage_ops: DataFrame with arbitrage opportunities
            
        Returns:
            Dictionary with metrics
        """
        if arbitrage_ops.empty:
            return {
                'total_opportunities': 0,
                'total_profit_potential': 0.0,
                'average_profit_bps': 0.0,
                'max_profit_bps': 0.0,
                'venue_pair_frequency': {}
            }
        
        # Calculate venue pair frequencies
        venue_pairs = arbitrage_ops.apply(
            lambda row: f"{row['best_bid_venue']}→{row['best_ask_venue']}", 
            axis=1
        ).value_counts().to_dict()
        
        metrics = {
            'total_opportunities': len(arbitrage_ops),
            'total_profit_potential': arbitrage_ops['total_profit_potential'].sum(),
            'average_profit_bps': arbitrage_ops['profit_bps'].mean(),
            'max_profit_bps': arbitrage_ops['profit_bps'].max(),
            'min_profit_bps': arbitrage_ops['profit_bps'].min(),
            'venue_pair_frequency': venue_pairs,
            'time_span_hours': (arbitrage_ops.index.max() - arbitrage_ops.index.min()).total_seconds() / 3600
        }
        
        return metrics
    
    def build_and_analyze(self, resample_freq: str = '1S', min_arbitrage_bps: float = 0.5) -> Dict:
        """
        Complete build and analysis pipeline
        
        Args:
            resample_freq: Resampling frequency
            min_arbitrage_bps: Minimum arbitrage profit in bps
            
        Returns:
            Complete analysis results
        """
        # Generate data and create tape
        self.generate_venue_data()
        consolidated_data = self.create_consolidated_tape(resample_freq)
        
        # Detect arbitrage opportunities
        arbitrage_ops = self.detect_arbitrage_opportunities(min_arbitrage_bps)
        
        # Calculate metrics
        metrics = self.calculate_metrics(arbitrage_ops)
        
        return {
            'consolidated_data': consolidated_data,
            'arbitrage_opportunities': arbitrage_ops,
            'metrics': metrics,
            'isin': self.isin,
            'venues': self.venues,
            'parameters': {
                'resample_freq': resample_freq,
                'min_arbitrage_bps': min_arbitrage_bps
            }
        }
    
    def export_analysis(self, filepath: str, analysis: Dict):
        """
        Export analysis results to file
        
        Args:
            filepath: Output file path
            analysis: Analysis results dictionary
        """
        if filepath.endswith('.xlsx'):
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(filepath) as writer:
                analysis['consolidated_data'].to_excel(writer, sheet_name='Consolidated_Tape')
                
                if not analysis['arbitrage_opportunities'].empty:
                    analysis['arbitrage_opportunities'].to_excel(writer, sheet_name='Arbitrage_Opportunities')
                
                # Metrics sheet
                metrics_df = pd.Series(analysis['metrics']).to_frame('Value')
                metrics_df.to_excel(writer, sheet_name='Metrics')
            
            print(f"✅ Excel analysis exported to {filepath}")
            
        elif filepath.endswith('.json'):
            # Export metrics to JSON
            export_data = analysis['metrics'].copy()
            export_data['isin'] = analysis['isin']
            export_data['venues'] = analysis['venues']
            export_data['parameters'] = analysis['parameters']
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ JSON metrics exported to {filepath}")

def quick_arbitrage_analysis(isin: str, venues: List[str] = None, num_points: int = 5000) -> Dict:
    """
    Quick standalone arbitrage analysis
    
    Args:
        isin: Target ISIN
        venues: List of venues (default: all 4)
        num_points: Number of data points to generate
        
    Returns:
        Analysis results dictionary
    """
    manager = StandaloneConsolidatedTapeManager(isin, venues)
    return manager.build_and_analyze()

def step2_complete_example():
    """
    Complete Step 2 example: Create consolidated tape for arbitrage detection
    """
    print("🚀 STEP 2: STANDALONE CONSOLIDATED TAPE IMPLEMENTATION")
    print("="*70)
    print("Creating consolidated tape for cross-venue arbitrage detection")
    print("📝 Note: Using simulated market data (no external dependencies)")
    
    # Configuration
    isin = "ES0113900J37"  # Santander
    
    print(f"\\n📊 Target ISIN: {isin}")
    print(f"🎯 Objective: Detect arbitrage opportunities across venues")
    
    try:
        # Method 1: Quick analysis (simple approach)
        print("\\n" + "="*50)
        print("METHOD 1: QUICK ARBITRAGE ANALYSIS")
        print("="*50)
        
        print("🔍 Running quick arbitrage analysis...")
        analysis = quick_arbitrage_analysis(isin, num_points=3000)
        
        print("✅ Quick analysis completed!")
        print(f"   📊 Found {analysis['metrics']['total_opportunities']} opportunities")
        print(f"   💰 Total profit potential: €{analysis['metrics']['total_profit_potential']:.2f}")
        
        # Method 2: Detailed analysis (full control)
        print("\\n" + "="*50) 
        print("METHOD 2: DETAILED CONSOLIDATED TAPE ANALYSIS")
        print("="*50)
        
        # Create manager
        manager = StandaloneConsolidatedTapeManager(isin)
        
        # Build and analyze with custom parameters
        print("\\n🔧 Building consolidated tape and detecting arbitrage...")
        detailed_analysis = manager.build_and_analyze(
            resample_freq='1S',      # 1-second resolution
            min_arbitrage_bps=0.5    # Minimum 0.5 basis points profit
        )
        
        # Display detailed results
        display_detailed_results(detailed_analysis)
        
        # Export results
        export_results(manager, detailed_analysis)
        
        return detailed_analysis
        
    except Exception as e:
        print(f"❌ STEP 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_detailed_results(analysis: dict):
    """
    Display detailed analysis results
    
    Args:
        analysis: Analysis results dictionary
    """
    print(f"\\n" + "="*50)
    print("📋 DETAILED ANALYSIS RESULTS")
    print("="*50)
    
    consolidated_df = analysis['consolidated_data']
    arbitrage_ops = analysis['arbitrage_opportunities'] 
    metrics = analysis['metrics']
    
    # Data structure overview
    print(f"\\n🏗️ CONSOLIDATED TAPE STRUCTURE:")
    print(f"   Shape: {consolidated_df.shape}")
    print(f"   Time range: {consolidated_df.index.min()} to {consolidated_df.index.max()}")
    print(f"   Columns: {len(consolidated_df.columns)}")
    
    # Show key columns
    bid_cols = [col for col in consolidated_df.columns if '_bid' in col and 'overall' not in col]
    ask_cols = [col for col in consolidated_df.columns if '_ask' in col and 'overall' not in col]
    
    print(f"\\n📊 VENUE PRICE COLUMNS:")
    print(f"   Bid columns: {bid_cols}")
    print(f"   Ask columns: {ask_cols}")
    
    # Sample of consolidated data (show first few rows with key columns)
    print(f"\\n📋 SAMPLE CONSOLIDATED DATA:")
    sample_cols = ['isin'] + bid_cols[:2] + ask_cols[:2]  # Show first 2 venues
    available_cols = [col for col in sample_cols if col in consolidated_df.columns]
    
    if available_cols:
        print(consolidated_df[available_cols].head())
        
        # Show basic statistics
        print(f"\\n📈 BASIC STATISTICS:")
        for col in bid_cols + ask_cols:
            if col in consolidated_df.columns:
                col_data = consolidated_df[col].dropna()
                if len(col_data) > 0:
                    print(f"   {col}: mean={col_data.mean():.4f}, std={col_data.std():.4f}")
    else:
        print("   (Displaying first few columns)")
        print(consolidated_df.head())
    
    # Arbitrage opportunities detail
    if not arbitrage_ops.empty:
        print(f"\\n🚨 ARBITRAGE OPPORTUNITIES DETAIL:")
        print(f"   Total opportunities: {len(arbitrage_ops)}")
        print(f"   Time span: {arbitrage_ops.index.min()} to {arbitrage_ops.index.max()}")
        
        if 'profit_bps' in arbitrage_ops.columns:
            print(f"   Profit range: {arbitrage_ops['profit_bps'].min():.2f} - {arbitrage_ops['profit_bps'].max():.2f} bps")
        
        # Show top opportunities
        display_cols = ['best_bid_venue', 'best_ask_venue', 'arbitrage_profit']
        if 'profit_bps' in arbitrage_ops.columns:
            display_cols.append('profit_bps')
        
        available_display_cols = [col for col in display_cols if col in arbitrage_ops.columns]
        
        if available_display_cols:
            print(f"\\n   Top 5 opportunities:")
            top_opportunities = arbitrage_ops.nlargest(5, 'arbitrage_profit')
            print(top_opportunities[available_display_cols])

def export_results(manager: StandaloneConsolidatedTapeManager, analysis: dict):
    """
    Export analysis results to files
    
    Args:
        manager: StandaloneConsolidatedTapeManager instance
        analysis: Analysis results
    """
    print(f"\\n" + "="*50)
    print("💾 EXPORTING RESULTS")
    print("="*50)
    
    try:
        # Export to Excel (comprehensive)
        excel_path = "consolidated_tape_analysis.xlsx"
        manager.export_analysis(excel_path, analysis)
        
        # Export metrics to JSON
        json_path = "arbitrage_metrics.json"
        manager.export_analysis(json_path, analysis)
        
        # Export CSV for quick access
        csv_path = "consolidated_tape_data.csv"
        analysis['consolidated_data'].to_csv(csv_path)
        print(f"✅ CSV data exported to {csv_path}")
        
        if not analysis['arbitrage_opportunities'].empty:
            arb_csv_path = "arbitrage_opportunities.csv"
            analysis['arbitrage_opportunities'].to_csv(arb_csv_path)
            print(f"✅ Arbitrage opportunities exported to {arb_csv_path}")
        
        print(f"\\n📁 EXPORTED FILES:")
        print(f"   📊 {excel_path} - Complete analysis (multiple sheets)")
        print(f"   📈 {csv_path} - Consolidated tape data")
        if not analysis['arbitrage_opportunities'].empty:
            print(f"   🚨 {arb_csv_path} - Arbitrage opportunities")
        print(f"   📋 {json_path} - Analysis metrics")
        
    except Exception as e:
        print(f"⚠️ Export warning: {e}")

def demonstrate_arbitrage_strategy():
    """
    Demonstrate how to use consolidated tape for arbitrage strategy
    """
    print(f"\\n" + "="*60)
    print("💡 ARBITRAGE STRATEGY DEMONSTRATION")
    print("="*60)
    
    print(f"\\n🎯 STEP 2 ACCOMPLISHMENTS:")
    print("✅ Created consolidated tape with timestamp-aligned simulated data")
    print("✅ Implemented cross-venue price comparison")
    print("✅ Detected arbitrage opportunities automatically")
    print("✅ Calculated potential profits in basis points")
    print("✅ Provided venue attribution for best quotes")
    print("✅ No external dependencies - fully standalone implementation")
    
    print(f"\\n📊 DATA STRUCTURE BENEFITS:")
    print("• Timestamp index enables exact time comparison")
    print("• Venue columns allow cross-venue analysis")
    print("• Forward-fill handles latency differences")
    print("• Resampling enables different time resolutions")
    print("• Realistic market microstructure simulation")
    
    print(f"\\n🚨 ARBITRAGE DETECTION FEATURES:")
    print("• Automatic identification of inverted spreads")
    print("• Profit calculation per opportunity")
    print("• Minimum profit threshold filtering")
    print("• Venue identification for execution")
    print("• Quantity-based profit potential calculation")
    
    print(f"\\n📈 SIMULATION FEATURES:")
    print("• Correlated price movements across venues")
    print("• Venue-specific spreads and noise")
    print("• Realistic bid/ask quantities")
    print("• Configurable volatility and frequency")
    print("• Reproducible results with random seed")
    
    print(f"\\n⚡ NEXT STEPS (Step 3 and beyond):")
    print("• Implement execution logic for arbitrage trades")
    print("• Add transaction cost analysis")
    print("• Develop position sizing algorithms")
    print("• Create risk management rules")
    print("• Build performance monitoring")

def main():
    """
    Main execution function
    """
    print("📈 STANDALONE CONSOLIDATED TAPE STEP 2 - COMPLETE IMPLEMENTATION")
    print("="*70)
    print("🎯 No external dependencies - complete standalone implementation")
    
    # Run complete Step 2 example
    analysis = step2_complete_example()
    
    if analysis:
        # Demonstrate strategy concepts
        demonstrate_arbitrage_strategy()
        
        print(f"\\n{'='*70}")
        print("🎉 STEP 2 COMPLETED SUCCESSFULLY!")
        print("✅ Consolidated tape created with simulated data")
        print("✅ Cross-venue arbitrage detection implemented") 
        print("✅ Analysis results exported")
        print("✅ No external dependencies required")
        print("🎯 Ready for Step 3: Arbitrage execution strategy")
        print("="*70)
    else:
        print(f"\\n{'='*70}")
        print("❌ STEP 2 FAILED")
        print("Please check the implementation and error messages")
        print("="*70)

if __name__ == "__main__":
    main()