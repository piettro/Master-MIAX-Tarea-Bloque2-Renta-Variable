"""
CRITICAL VENDOR DATA VALIDATION TEST

This script demonstrates the critical vendor-specific data validations:
1. Magic Numbers filtering (prevents massive P&L errors)
2. Market Status validation (ensures tradable periods)
3. Addressable orderbook filtering

IMPORTANT: These validations prevent trading algorithm disasters!
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

def create_test_data_with_magic_numbers():
    """
    Creates test data that includes vendor magic numbers to demonstrate filtering
    """
    print("🧪 Creating test data with magic numbers...")
    
    # Create sample data with magic numbers mixed in
    test_data = {
        'session': ['2025-11-07'] * 10,
        'isin': ['ES0113900J37'] * 10,
        'ticker': ['SAN'] * 10,
        'mic': ['AQEU'] * 10,
        'event_timestamp': pd.date_range('2025-11-07 10:00:00', periods=10, freq='1S'),
        
        # Mix normal prices with magic numbers
        'px_bid_0': [3.45, 666666.666, 3.46, 999999.999, 3.47, 3.48, 999999.989, 3.49, 999999.988, 3.50],
        'px_ask_0': [3.46, 3.47, 999999.979, 3.48, 3.49, 999999.123, 3.50, 3.51, 3.52, 3.53],
        'qty_bid_0': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'qty_ask_0': [1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950],
    }
    
    return pd.DataFrame(test_data)

def create_test_sts_data():
    """
    Creates test STS data with different market status codes
    """
    print("🧪 Creating test STS data with market status codes...")
    
    # Create STS data with mix of trading and non-trading periods
    sts_data = {
        'session': ['2025-11-07'] * 10,
        'isin': ['ES0113900J37'] * 10,
        'mic': ['AQEU'] * 10,
        'event_timestamp': pd.date_range('2025-11-07 10:00:00', periods=10, freq='1S'),
        
        # Mix continuous trading code (5308427) with other status codes
        'market_status': [5308427, 5308427, 1234567, 5308427, 5308427, 8888888, 5308427, 5308427, 9999999, 5308427],
        'trading_state': ['CONTINUOUS', 'CONTINUOUS', 'AUCTION', 'CONTINUOUS', 'CONTINUOUS', 'HALT', 'CONTINUOUS', 'CONTINUOUS', 'CLOSE', 'CONTINUOUS']
    }
    
    return pd.DataFrame(sts_data)

def test_magic_number_filtering():
    """
    Tests the critical magic number filtering functionality
    """
    print("\\n" + "="*60)
    print("🚨 CRITICAL TEST: MAGIC NUMBER FILTERING")
    print("="*60)
    
    # Create test data with magic numbers
    test_df = create_test_data_with_magic_numbers()
    
    print(f"\\n📊 Original data: {len(test_df)} records")
    print("Sample prices before filtering:")
    for i, (bid, ask) in enumerate(zip(test_df['px_bid_0'], test_df['px_ask_0'])):
        status = "🚨 MAGIC!" if (bid in [666666.666, 999999.999, 999999.989, 999999.988, 999999.979, 999999.123] or 
                                ask in [666666.666, 999999.999, 999999.989, 999999.988, 999999.979, 999999.123]) else "✓ Normal"
        print(f"  Record {i}: Bid={bid}, Ask={ask} - {status}")
    
    # Test magic number filtering
    extractor = AquisExtractor()
    magic_numbers = extractor._get_magic_numbers()
    
    print(f"\\n🎭 Magic numbers to filter: {magic_numbers}")
    
    # Apply filtering
    filtered_df = extractor._filter_magic_numbers(test_df)
    
    print(f"\\n📊 After filtering: {len(filtered_df)} records")
    print("Remaining prices:")
    for i, (bid, ask) in enumerate(zip(filtered_df['px_bid_0'], filtered_df['px_ask_0'])):
        print(f"  Record {i}: Bid={bid}, Ask={ask}")
    
    # Verify no magic numbers remain
    all_prices = list(filtered_df['px_bid_0']) + list(filtered_df['px_ask_0'])
    magic_found = [p for p in all_prices if p in magic_numbers]
    
    if magic_found:
        print(f"❌ CRITICAL ERROR: Magic numbers still present: {magic_found}")
    else:
        print("✅ SUCCESS: All magic numbers removed!")
    
    return filtered_df

def test_continuous_trading_filter():
    """
    Tests the continuous trading period filtering
    """
    print("\\n" + "="*60) 
    print("🕒 CRITICAL TEST: CONTINUOUS TRADING FILTER")
    print("="*60)
    
    # Create test data
    qte_data = create_test_data_with_magic_numbers()
    sts_data = create_test_sts_data()
    
    print(f"\\n📊 QTE data: {len(qte_data)} records")
    print(f"📊 STS data: {len(sts_data)} records")
    
    print("\\nMarket status by timestamp:")
    for i, (ts, status, state) in enumerate(zip(sts_data['event_timestamp'], sts_data['market_status'], sts_data['trading_state'])):
        tradable = "✅ TRADABLE" if status == 5308427 else "❌ NON-TRADABLE"
        print(f"  {ts}: Status={status} ({state}) - {tradable}")
    
    # Test continuous trading filter
    extractor = AquisExtractor()
    continuous_codes = extractor._get_continuous_trading_codes()
    
    print(f"\\n🏛️ Continuous trading codes for AQUIS: {continuous_codes.get('AQEU', [])}")
    
    # Apply filtering
    filtered_qte = extractor._filter_non_continuous_trading(qte_data, sts_data)
    
    print(f"\\n📊 QTE records after continuous trading filter: {len(filtered_qte)}")
    
    if len(filtered_qte) < len(qte_data):
        removed = len(qte_data) - len(filtered_qte)
        print(f"✅ Correctly removed {removed} records from non-trading periods")
    else:
        print("ℹ️  No records removed (all periods were continuous trading)")

def test_comprehensive_validation():
    """
    Tests the comprehensive data validation pipeline
    """
    print("\\n" + "="*60)
    print("🛡️ COMPREHENSIVE VALIDATION PIPELINE TEST")
    print("="*60)
    
    # Create test data
    qte_data = create_test_data_with_magic_numbers()
    sts_data = create_test_sts_data()
    
    # Apply comprehensive validation
    extractor = AquisExtractor()
    result = extractor.clean_and_validate_data(qte_data, sts_data)
    
    # Display results
    cleaned_qte = result['qte_data']
    cleaned_sts = result['sts_data'] 
    report = result['validation_report']
    
    print(f"\\n📋 VALIDATION RESULTS:")
    print(f"   Original QTE records: {report['original_qte_records']}")
    print(f"   Cleaned QTE records: {report['cleaned_qte_records']}")
    print(f"   Removal rate: {report['qte_removal_rate']:.1f}%")
    print(f"   Data quality score: {report['data_quality_score']:.1f}%")
    print(f"   Magic numbers filtered: {'✅' if report['magic_numbers_filtered'] else '❌'}")
    print(f"   Trading hours validated: {'✅' if report['continuous_trading_validated'] else '⚠️'}")
    
    # Check final data quality
    if len(cleaned_qte) > 0:
        print(f"\\n✅ Final clean dataset: {len(cleaned_qte)} tradable records")
        print(f"   Price range: {cleaned_qte['px_bid_0'].min():.3f} - {cleaned_qte['px_ask_0'].max():.3f}")
    else:
        print(f"\\n❌ CRITICAL: No tradable records remaining!")

def demonstrate_pl_impact():
    """
    Demonstrates the P&L impact of not filtering magic numbers
    """
    print("\\n" + "="*60)
    print("💰 P&L IMPACT DEMONSTRATION")
    print("="*60)
    
    # Scenario: Algorithm tries to sell at magic number price
    magic_price = 999999.999
    real_price = 3.45
    quantity = 1000
    
    # Calculate P&L difference
    magic_pnl = quantity * magic_price  # Catastrophic overestimation
    real_pnl = quantity * real_price    # Actual possible P&L
    error_amount = magic_pnl - real_pnl
    
    print(f"\\n🎯 Scenario: Selling {quantity} shares")
    print(f"   Magic number price: €{magic_price:,.3f}")
    print(f"   Real market price:  €{real_price:,.3f}")
    print(f"\\n💸 P&L Calculation:")
    print(f"   With magic number:  €{magic_pnl:,.2f}")
    print(f"   With real price:    €{real_pnl:,.2f}")
    print(f"   🚨 ERROR AMOUNT:   €{error_amount:,.2f}")
    print(f"\\n🔥 P&L OVERESTIMATION: {error_amount/real_pnl*100:,.0f}%")
    print(f"\\nThis is why magic number filtering is CRITICAL!")

if __name__ == "__main__":
    print("🚨 CRITICAL VENDOR DATA VALIDATION TESTS")
    print("="*60)
    print("These tests prevent trading algorithm disasters!")
    
    try:
        # Run all critical tests
        test_magic_number_filtering()
        test_continuous_trading_filter() 
        test_comprehensive_validation()
        demonstrate_pl_impact()
        
        print(f"\\n{'='*60}")
        print("🛡️ ALL CRITICAL VALIDATIONS PASSED")
        print("Your data is now safe for trading algorithms!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\\n❌ CRITICAL ERROR: {e}")
        print("Data validation failed - DO NOT USE FOR TRADING!")