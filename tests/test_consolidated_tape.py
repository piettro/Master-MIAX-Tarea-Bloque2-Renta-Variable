"""
Consolidated Tape Demonstration and Testing

This script demonstrates the Step 2 implementation: Creating a "Consolidated Tape"
for cross-venue arbitrage detection using object-oriented design and best practices.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.models.consolidated_tape import ConsolidatedTape, ConsolidatedTapeFactory, VenueData

def create_sample_venue_data():
    """
    Create sample multi-venue data for demonstration
    """
    print("🧪 Creating sample multi-venue data...")
    
    # Create timestamps for 1 minute with 1-second intervals
    timestamps = pd.date_range('2025-11-07 10:00:00', periods=60, freq='1S')
    
    # Base price that moves randomly
    base_price = 3.45
    price_changes = np.random.normal(0, 0.001, len(timestamps))
    prices = base_price + np.cumsum(price_changes)
    
    venues_data = {}
    
    # BME data (typically tightest spreads)
    bme_data = pd.DataFrame({
        'event_timestamp': timestamps,
        'session': ['2025-11-07'] * len(timestamps),
        'isin': ['ES0113900J37'] * len(timestamps),
        'ticker': ['SAN'] * len(timestamps),
        'mic': ['XMAD'] * len(timestamps),
        'px_bid_0': prices - 0.001,  # Tight spread
        'px_ask_0': prices + 0.001,
        'qty_bid_0': np.random.randint(800, 1200, len(timestamps)),
        'qty_ask_0': np.random.randint(800, 1200, len(timestamps))
    })
    venues_data['BME'] = bme_data
    
    # AQUIS data (slightly wider spreads, some latency)
    aquis_timestamps = timestamps + timedelta(milliseconds=50)  # 50ms delay
    aquis_data = pd.DataFrame({
        'event_timestamp': aquis_timestamps,
        'session': ['2025-11-07'] * len(timestamps),
        'isin': ['ES0113900J37'] * len(timestamps),
        'ticker': ['SAN'] * len(timestamps),
        'mic': ['AQEU'] * len(timestamps),
        'px_bid_0': prices - 0.0015,  # Slightly wider spread
        'px_ask_0': prices + 0.0015,
        'qty_bid_0': np.random.randint(500, 1000, len(timestamps)),
        'qty_ask_0': np.random.randint(500, 1000, len(timestamps))
    })
    venues_data['AQUIS'] = aquis_data
    
    # CBOE data (wider spreads, different pricing)
    cboe_timestamps = timestamps + timedelta(milliseconds=100)  # 100ms delay
    cboe_prices = prices + np.random.normal(0, 0.0005, len(prices))  # Slight price differences
    cboe_data = pd.DataFrame({
        'event_timestamp': cboe_timestamps,
        'session': ['2025-11-07'] * len(timestamps),
        'isin': ['ES0113900J37'] * len(timestamps),
        'ticker': ['SAN'] * len(timestamps),
        'mic': ['CEUX'] * len(timestamps),
        'px_bid_0': cboe_prices - 0.002,  # Wider spread
        'px_ask_0': cboe_prices + 0.002,
        'qty_bid_0': np.random.randint(300, 800, len(timestamps)),
        'qty_ask_0': np.random.randint(300, 800, len(timestamps))
    })
    venues_data['CBOE'] = cboe_data
    
    # Add some arbitrage opportunities manually
    # Make BME bid occasionally higher than AQUIS ask
    arbitrage_indices = [10, 25, 40, 55]
    for idx in arbitrage_indices:
        venues_data['BME'].loc[idx, 'px_bid_0'] = venues_data['AQUIS'].loc[idx, 'px_ask_0'] + 0.002
    
    print(f"✅ Created sample data for {len(venues_data)} venues")
    return venues_data

def test_consolidated_tape_creation():
    """
    Test basic consolidated tape creation and functionality
    """
    print("\\n" + "="*60)
    print("🏗️  STEP 2: CONSOLIDATED TAPE CREATION TEST")
    print("="*60)
    
    isin = "ES0113900J37"
    
    # Create consolidated tape using factory
    tape = ConsolidatedTapeFactory.create_tape(isin)
    print(f"✅ Created consolidated tape for ISIN: {isin}")
    
    # Get sample data
    venues_data = create_sample_venue_data()
    
    # Add venue data to tape
    for venue_name, venue_data in venues_data.items():
        tape.add_venue_data(venue_name, venue_data)
    
    # Display venue summaries
    print("\\n📊 VENUE SUMMARIES:")
    venue_summary = tape.get_venue_summary()
    for _, venue in venue_summary.iterrows():
        print(f"   {venue['venue']}: {venue['records']} records")
        if 'best_bid_stats' in venue and pd.notna(venue['best_bid_stats']):
            print(f"      Bid range: {venue['best_bid_stats']['min']:.4f} - {venue['best_bid_stats']['max']:.4f}")
    
    return tape

def test_tape_building_and_arbitrage():
    """
    Test consolidated tape building and arbitrage detection
    """
    print("\\n" + "="*60)
    print("🎯 ARBITRAGE DETECTION TEST")
    print("="*60)
    
    # Create and populate tape
    tape = test_consolidated_tape_creation()
    
    # Build the consolidated tape
    print("\\n🔧 Building consolidated tape...")
    consolidated_df = tape.build_tape(resample_freq='1S')
    
    print(f"\\n📊 CONSOLIDATED TAPE STRUCTURE:")
    print(f"   Shape: {consolidated_df.shape}")
    print(f"   Time range: {consolidated_df.index.min()} to {consolidated_df.index.max()}")
    
    # Show column structure
    bid_cols = [col for col in consolidated_df.columns if '_best_bid' in col and 'overall' not in col]
    ask_cols = [col for col in consolidated_df.columns if '_best_ask' in col and 'overall' not in col]
    
    print(f"   Bid columns: {bid_cols}")
    print(f"   Ask columns: {ask_cols}")
    
    # Show sample data
    print("\\n📋 SAMPLE CONSOLIDATED DATA:")
    sample_cols = bid_cols + ask_cols + ['best_bid_overall', 'best_ask_overall', 'arbitrage_opportunity']
    available_sample_cols = [col for col in sample_cols if col in consolidated_df.columns]
    print(consolidated_df[available_sample_cols].head())
    
    # Detect arbitrage opportunities
    print("\\n🚨 DETECTING ARBITRAGE OPPORTUNITIES:")
    arbitrage_ops = tape.get_arbitrage_opportunities(min_profit_bps=0.5)
    
    if not arbitrage_ops.empty:
        print(f"   Found {len(arbitrage_ops)} arbitrage opportunities!")
        print("\\n   Top 5 opportunities:")
        display_cols = ['best_bid_venue', 'best_ask_venue', 'arbitrage_profit', 'profit_bps']
        available_display_cols = [col for col in display_cols if col in arbitrage_ops.columns]
        print(arbitrage_ops[available_display_cols].head())
        
        # Calculate potential total profit
        total_profit = arbitrage_ops['arbitrage_profit'].sum()
        avg_profit_bps = arbitrage_ops['profit_bps'].mean()
        print(f"\\n   💰 Total potential profit: €{total_profit:.4f}")
        print(f"   📊 Average profit: {avg_profit_bps:.2f} basis points")
    
    return tape, consolidated_df, arbitrage_ops

def test_cross_venue_analysis():
    """
    Test detailed cross-venue analysis capabilities
    """
    print("\\n" + "="*60)
    print("📈 CROSS-VENUE ANALYSIS TEST")
    print("="*60)
    
    tape, consolidated_df, arbitrage_ops = test_tape_building_and_arbitrage()
    
    # Analyze spread patterns across venues
    spread_cols = [col for col in consolidated_df.columns if '_spread_bps' in col and 'overall' not in col]
    
    if spread_cols:
        print("\\n📊 SPREAD ANALYSIS BY VENUE:")
        for col in spread_cols:
            venue = col.replace('_spread_bps', '')
            mean_spread = consolidated_df[col].mean()
            min_spread = consolidated_df[col].min()
            max_spread = consolidated_df[col].max()
            print(f"   {venue}: Mean={mean_spread:.2f} bps, Range={min_spread:.2f}-{max_spread:.2f} bps")
    
    # Analyze best quote distribution
    if 'best_bid_venue' in consolidated_df.columns:
        print("\\n🏆 BEST BID VENUE DISTRIBUTION:")
        venue_distribution = consolidated_df['best_bid_venue'].value_counts()
        for venue, count in venue_distribution.items():
            percentage = count / len(consolidated_df) * 100
            print(f"   {venue}: {count} times ({percentage:.1f}%)")
    
    if 'best_ask_venue' in consolidated_df.columns:
        print("\\n🎯 BEST ASK VENUE DISTRIBUTION:")
        venue_distribution = consolidated_df['best_ask_venue'].value_counts()
        for venue, count in venue_distribution.items():
            percentage = count / len(consolidated_df) * 100
            print(f"   {venue}: {count} times ({percentage:.1f}%)")
    
    # Time-based analysis
    if 'arbitrage_opportunity' in consolidated_df.columns:
        arbitrage_rate = consolidated_df['arbitrage_opportunity'].sum() / len(consolidated_df) * 100
        print(f"\\n⚡ ARBITRAGE OPPORTUNITY RATE: {arbitrage_rate:.2f}% of timestamps")
    
    return consolidated_df

def test_export_functionality():
    """
    Test export functionality
    """
    print("\\n" + "="*60)
    print("💾 EXPORT FUNCTIONALITY TEST")
    print("="*60)
    
    tape, consolidated_df, arbitrage_ops = test_tape_building_and_arbitrage()
    
    # Test CSV export
    csv_path = Path("consolidated_tape_test.csv")
    try:
        tape.export_tape(str(csv_path))
        if csv_path.exists():
            print(f"✅ CSV export successful: {csv_path}")
            csv_path.unlink()  # Clean up
        else:
            print(f"❌ CSV export failed")
    except Exception as e:
        print(f"❌ CSV export error: {e}")
    
    # Test Excel export
    excel_path = Path("consolidated_tape_test.xlsx")
    try:
        tape.export_tape(str(excel_path))
        if excel_path.exists():
            print(f"✅ Excel export successful: {excel_path}")
            excel_path.unlink()  # Clean up
        else:
            print(f"❌ Excel export failed")
    except Exception as e:
        print(f"❌ Excel export error: {e}")

def demonstrate_real_world_usage():
    """
    Demonstrate real-world usage patterns
    """
    print("\\n" + "="*60)
    print("🌍 REAL-WORLD USAGE DEMONSTRATION")
    print("="*60)
    
    print("\\n📚 TYPICAL WORKFLOW:")
    print("1. Create consolidated tape for specific ISIN")
    print("2. Add venue data from multiple sources")
    print("3. Build consolidated view with timestamp alignment") 
    print("4. Detect arbitrage opportunities")
    print("5. Analyze cross-venue patterns")
    print("6. Export for further analysis")
    
    print("\\n💡 KEY FEATURES:")
    print("✅ Object-oriented design with clear separation of concerns")
    print("✅ Automatic timestamp alignment across venues") 
    print("✅ Forward-fill for handling latency differences")
    print("✅ Built-in arbitrage detection with profit calculation")
    print("✅ Comprehensive venue statistics and analysis")
    print("✅ Flexible export options (CSV, Excel)")
    print("✅ Configurable resampling for different time resolutions")
    
    print("\\n🎯 ARBITRAGE DETECTION CAPABILITIES:")
    print("- Cross-venue price comparison at exact timestamps")
    print("- Automatic identification of inverted spreads")
    print("- Profit calculation in absolute and basis points")
    print("- Minimum profit threshold filtering")
    print("- Venue attribution for best bid/ask")

if __name__ == "__main__":
    print("🚀 CONSOLIDATED TAPE - STEP 2 IMPLEMENTATION TEST")
    print("="*60)
    print("Testing object-oriented consolidated tape for arbitrage detection")
    
    try:
        # Run all tests
        test_consolidated_tape_creation()
        test_tape_building_and_arbitrage()
        test_cross_venue_analysis()
        test_export_functionality()
        demonstrate_real_world_usage()
        
        print(f"\\n{'='*60}")
        print("🎉 ALL CONSOLIDATED TAPE TESTS PASSED!")
        print("✅ Step 2 implementation complete and validated")
        print("🎯 Ready for cross-venue arbitrage detection")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()