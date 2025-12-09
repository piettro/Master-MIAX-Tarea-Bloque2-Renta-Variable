"""
Test Suite for Step 3: Arbitrage Signal Generation

This module provides comprehensive testing for the arbitrage signal generation system,
including unit tests, integration tests, and performance validation.

Author: GitHub Copilot
Date: December 2025
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.arbitrage_signals import (
    ArbitrageSignalGenerator, 
    ArbitrageOpportunity, 
    ArbitrageSignalManager,
    ArbitrageSignalFactory,
    OpportunityStatus
)


class TestArbitrageOpportunity(unittest.TestCase):
    """Test cases for ArbitrageOpportunity dataclass"""
    
    def test_valid_opportunity_creation(self):
        """Test creating a valid arbitrage opportunity"""
        opp = ArbitrageOpportunity(
            opportunity_id="TEST_001",
            detection_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(seconds=1),
            max_bid_venue="AQUIS",
            min_ask_venue="BME", 
            max_bid_price=100.05,
            min_ask_price=100.02,
            max_bid_quantity=1000,
            min_ask_quantity=500,
            profit_per_share=0.03,
            max_tradeable_quantity=500,
            total_profit=15.0
        )
        
        self.assertEqual(opp.opportunity_id, "TEST_001")
        self.assertEqual(opp.max_bid_venue, "AQUIS")
        self.assertEqual(opp.min_ask_venue, "BME")
        self.assertEqual(opp.profit_per_share, 0.03)
        self.assertEqual(opp.status, OpportunityStatus.ACTIVE)
    
    def test_invalid_opportunity_bid_less_than_ask(self):
        """Test that opportunity with bid <= ask raises error"""
        with self.assertRaises(ValueError):
            ArbitrageOpportunity(
                opportunity_id="INVALID_001",
                detection_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(seconds=1),
                max_bid_venue="AQUIS",
                min_ask_venue="BME",
                max_bid_price=100.02,  # Bid lower than ask
                min_ask_price=100.05,
                max_bid_quantity=1000,
                min_ask_quantity=500,
                profit_per_share=0.03,
                max_tradeable_quantity=500,
                total_profit=15.0
            )
    
    def test_invalid_opportunity_zero_quantity(self):
        """Test that opportunity with zero quantity raises error"""
        with self.assertRaises(ValueError):
            ArbitrageOpportunity(
                opportunity_id="INVALID_002",
                detection_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(seconds=1),
                max_bid_venue="AQUIS",
                min_ask_venue="BME",
                max_bid_price=100.05,
                min_ask_price=100.02,
                max_bid_quantity=1000,
                min_ask_quantity=500,
                profit_per_share=0.03,
                max_tradeable_quantity=0,  # Zero quantity
                total_profit=15.0
            )


class TestArbitrageSignalGenerator(unittest.TestCase):
    """Test cases for ArbitrageSignalGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ArbitrageSignalGenerator(
            persistence_snapshots=1000,
            min_profit_threshold=0.001
        )
    
    def test_calculate_profit(self):
        """Test profit calculation logic"""
        profit_per_share, max_quantity, total_profit = self.generator.calculate_profit(
            max_bid=100.05,
            min_ask=100.02,
            bid_qty=1000,
            ask_qty=500
        )
        
        self.assertAlmostEqual(profit_per_share, 0.03, places=4)
        self.assertEqual(max_quantity, 500)  # Min of bid/ask quantities
        self.assertAlmostEqual(total_profit, 15.0, places=2)
    
    def test_validate_opportunity_valid(self):
        """Test opportunity validation with valid opportunity"""
        opp = ArbitrageOpportunity(
            opportunity_id="VALID_001",
            detection_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(seconds=1),
            max_bid_venue="AQUIS",
            min_ask_venue="BME",
            max_bid_price=100.05,
            min_ask_price=100.02,
            max_bid_quantity=1000,
            min_ask_quantity=500,
            profit_per_share=0.03,
            max_tradeable_quantity=500,
            total_profit=15.0
        )
        
        self.assertTrue(self.generator.validate_opportunity(opp))
    
    def test_validate_opportunity_same_venue(self):
        """Test opportunity validation fails with same venue"""
        opp = ArbitrageOpportunity(
            opportunity_id="INVALID_VENUE",
            detection_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(seconds=1),
            max_bid_venue="AQUIS",
            min_ask_venue="AQUIS",  # Same venue
            max_bid_price=100.05,
            min_ask_price=100.02,
            max_bid_quantity=1000,
            min_ask_quantity=500,
            profit_per_share=0.03,
            max_tradeable_quantity=500,
            total_profit=15.0
        )
        
        self.assertFalse(self.generator.validate_opportunity(opp))
    
    def test_validate_opportunity_low_profit(self):
        """Test opportunity validation fails with profit below threshold"""
        self.generator.min_profit_threshold = 0.05  # Higher threshold
        
        opp = ArbitrageOpportunity(
            opportunity_id="LOW_PROFIT",
            detection_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(seconds=1),
            max_bid_venue="AQUIS",
            min_ask_venue="BME",
            max_bid_price=100.03,
            min_ask_price=100.02,
            max_bid_quantity=1000,
            min_ask_quantity=500,
            profit_per_share=0.01,  # Below threshold
            max_tradeable_quantity=500,
            total_profit=5.0
        )
        
        self.assertFalse(self.generator.validate_opportunity(opp))
    
    def test_create_opportunity_id(self):
        """Test opportunity ID creation"""
        opp_id = self.generator._create_opportunity_id("AQUIS", "BME", 100.05, 100.02)
        expected = "AQUIS_BME_100.0500_100.0200"
        self.assertEqual(opp_id, expected)
    
    def test_rising_edge_detection_new_opportunity(self):
        """Test rising edge detection for new opportunities"""
        opp_id = "NEW_OPPORTUNITY"
        timestamp = datetime.now()
        
        # First time seeing this opportunity should be rising edge
        self.assertTrue(self.generator._is_rising_edge(opp_id, timestamp))
        
        # Should now be in history
        self.assertIn(opp_id, self.generator.opportunity_history)
    
    def test_extract_venue_data(self):
        """Test extraction of venue data from consolidated tape row"""
        # Create mock consolidated tape row
        row_data = pd.Series({
            'AQUIS_bid': 100.05,
            'AQUIS_ask': 100.07,
            'AQUIS_bid_qty': 1000,
            'AQUIS_ask_qty': 800,
            'BME_bid': 100.03,
            'BME_ask': 100.02,  # Lower ask creates arbitrage
            'BME_bid_qty': 500,
            'BME_ask_qty': 600,
            'CBOE_bid': np.nan,  # Missing data
            'CBOE_ask': np.nan
        })
        
        venue_data = self.generator._extract_venue_data(row_data)
        
        # Should have AQUIS and BME data
        self.assertIn('AQUIS', venue_data)
        self.assertIn('BME', venue_data)
        self.assertNotIn('CBOE', venue_data)  # Missing data
        
        # Verify AQUIS data
        aquis_data = venue_data['AQUIS']
        self.assertEqual(aquis_data['bid_price'], 100.05)
        self.assertEqual(aquis_data['ask_price'], 100.07)
        self.assertEqual(aquis_data['bid_quantity'], 1000)
        self.assertEqual(aquis_data['ask_quantity'], 800)
    
    def test_detect_opportunities_with_arbitrage(self):
        """Test opportunity detection with valid arbitrage condition"""
        # Create consolidated data with arbitrage opportunity
        data = pd.DataFrame({
            'AQUIS_bid': [100.05],
            'AQUIS_ask': [100.07],
            'AQUIS_bid_qty': [1000],
            'AQUIS_ask_qty': [800],
            'BME_bid': [100.03],
            'BME_ask': [100.02],  # Lower ask creates arbitrage
            'BME_bid_qty': [500],
            'BME_ask_qty': [600]
        }, index=[datetime.now()])
        
        opportunities = self.generator.detect_opportunities(data)
        
        self.assertEqual(len(opportunities), 1)
        
        opp = opportunities[0]
        self.assertEqual(opp.max_bid_venue, 'AQUIS')
        self.assertEqual(opp.min_ask_venue, 'BME')
        self.assertAlmostEqual(opp.profit_per_share, 0.03, places=4)
    
    def test_detect_opportunities_no_arbitrage(self):
        """Test opportunity detection with no arbitrage condition"""
        # Create consolidated data with normal market (no arbitrage)
        data = pd.DataFrame({
            'AQUIS_bid': [100.03],
            'AQUIS_ask': [100.05],
            'BME_bid': [100.02],
            'BME_ask': [100.04]  # No arbitrage condition
        }, index=[datetime.now()])
        
        opportunities = self.generator.detect_opportunities(data)
        
        self.assertEqual(len(opportunities), 0)
    
    def test_cleanup_expired_opportunities(self):
        """Test cleanup of expired opportunities"""
        # Add an active opportunity
        old_time = datetime.now() - timedelta(seconds=2)
        opp = ArbitrageOpportunity(
            opportunity_id="EXPIRED_001",
            detection_time=old_time,
            expiry_time=old_time + timedelta(seconds=1),  # Already expired
            max_bid_venue="AQUIS",
            min_ask_venue="BME",
            max_bid_price=100.05,
            min_ask_price=100.02,
            max_bid_quantity=1000,
            min_ask_quantity=500,
            profit_per_share=0.03,
            max_tradeable_quantity=500,
            total_profit=15.0
        )
        
        self.generator.active_opportunities["EXPIRED_001"] = opp
        
        # Run cleanup
        current_time = datetime.now()
        self.generator._cleanup_expired_opportunities(current_time)
        
        # Opportunity should be removed
        self.assertNotIn("EXPIRED_001", self.generator.active_opportunities)
    
    def test_statistics_tracking(self):
        """Test statistics tracking functionality"""
        # Initial statistics
        stats = self.generator.get_statistics()
        self.assertEqual(stats['total_opportunities_detected'], 0)
        
        # Simulate processing some data
        self.generator.total_opportunities_detected = 5
        self.generator.total_profit_potential = 125.0
        self.generator.snapshot_count = 1000
        
        stats = self.generator.get_statistics()
        self.assertEqual(stats['total_opportunities_detected'], 5)
        self.assertEqual(stats['total_profit_potential'], 125.0)
        self.assertEqual(stats['total_snapshots_processed'], 1000)


class TestArbitrageSignalManager(unittest.TestCase):
    """Test cases for ArbitrageSignalManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ArbitrageSignalGenerator()
        self.manager = ArbitrageSignalManager(self.generator)
    
    def test_process_market_data(self):
        """Test processing market data through manager"""
        # Create test data
        data = pd.DataFrame({
            'AQUIS_bid': [100.05],
            'AQUIS_ask': [100.07],
            'BME_bid': [100.03],
            'BME_ask': [100.02]  # Arbitrage opportunity
        }, index=[datetime.now()])
        
        result = self.manager.process_market_data(data)
        
        self.assertIn('new_opportunities', result)
        self.assertIn('active_opportunities', result)
        self.assertIn('statistics', result)
    
    def test_opportunity_summary_empty(self):
        """Test opportunity summary with no opportunities"""
        summary = self.manager.get_opportunity_summary()
        self.assertTrue(summary.empty)
    
    def test_venue_performance_analysis_empty(self):
        """Test venue performance analysis with no opportunities"""
        analysis = self.manager.analyze_venue_performance()
        self.assertEqual(analysis, {})


class TestArbitrageSignalFactory(unittest.TestCase):
    """Test cases for ArbitrageSignalFactory"""
    
    def test_create_standard_generator(self):
        """Test creating standard generator"""
        generator = ArbitrageSignalFactory.create_standard_generator()
        
        self.assertIsInstance(generator, ArbitrageSignalGenerator)
        self.assertEqual(generator.persistence_snapshots, 1000)
        self.assertEqual(generator.min_profit_threshold, 0.001)
    
    def test_create_high_frequency_generator(self):
        """Test creating high frequency generator"""
        generator = ArbitrageSignalFactory.create_high_frequency_generator()
        
        self.assertIsInstance(generator, ArbitrageSignalGenerator)
        self.assertEqual(generator.persistence_snapshots, 100)
        self.assertEqual(generator.min_profit_threshold, 0.0001)
    
    def test_create_conservative_generator(self):
        """Test creating conservative generator"""
        generator = ArbitrageSignalFactory.create_conservative_generator()
        
        self.assertIsInstance(generator, ArbitrageSignalGenerator)
        self.assertEqual(generator.persistence_snapshots, 2000)
        self.assertEqual(generator.min_profit_threshold, 0.01)


class TestIntegrationSignalGeneration(unittest.TestCase):
    """Integration tests for complete signal generation workflow"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.generator = ArbitrageSignalFactory.create_standard_generator()
        self.manager = ArbitrageSignalManager(self.generator)
    
    def test_full_workflow_with_multiple_opportunities(self):
        """Test complete workflow with multiple arbitrage opportunities"""
        # Create time series data with multiple arbitrage opportunities
        timestamps = [datetime.now() + timedelta(milliseconds=i*10) for i in range(10)]
        
        data = pd.DataFrame({
            'AQUIS_bid': [100.05, 100.06, 100.04, 100.05, 100.07, 100.05, 100.06, 100.04, 100.05, 100.06],
            'AQUIS_ask': [100.07, 100.08, 100.06, 100.07, 100.09, 100.07, 100.08, 100.06, 100.07, 100.08],
            'BME_bid': [100.03, 100.04, 100.02, 100.03, 100.05, 100.03, 100.04, 100.02, 100.03, 100.04],
            'BME_ask': [100.02, 100.01, 100.03, 100.02, 100.01, 100.02, 100.01, 100.03, 100.02, 100.01]  # Multiple arbitrage opportunities
        }, index=timestamps)
        
        total_opportunities = 0
        
        # Process each row to simulate real-time processing
        for i in range(len(data)):
            current_data = data.iloc[:i+1]  # Cumulative data up to current point
            result = self.manager.process_market_data(current_data)
            total_opportunities += len(result['new_opportunities'])
        
        # Verify opportunities were detected
        self.assertGreater(total_opportunities, 0)
        
        # Check opportunity summary
        summary = self.manager.get_opportunity_summary()
        self.assertFalse(summary.empty)
        
        # Check venue analysis
        analysis = self.manager.analyze_venue_performance()
        self.assertIn('opportunity_counts_by_venue_pair', analysis)
    
    def test_rising_edge_prevention_duplicate_counting(self):
        """Test that rising edge detection prevents double counting"""
        # Create identical arbitrage conditions for multiple snapshots
        timestamps = [datetime.now() + timedelta(milliseconds=i) for i in range(5)]
        
        # Same arbitrage condition repeated
        data = pd.DataFrame({
            'AQUIS_bid': [100.05] * 5,
            'AQUIS_ask': [100.07] * 5,
            'BME_bid': [100.03] * 5,
            'BME_ask': [100.02] * 5  # Same arbitrage opportunity
        }, index=timestamps)
        
        opportunities_detected = []
        
        # Process each snapshot
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            opportunities = self.generator.detect_opportunities(current_data)
            opportunities_detected.extend(opportunities)
        
        # Should only detect opportunity once (rising edge)
        self.assertEqual(len(opportunities_detected), 1)
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset"""
        import time
        
        # Create larger dataset
        n_snapshots = 1000
        timestamps = [datetime.now() + timedelta(milliseconds=i) for i in range(n_snapshots)]
        
        # Random walk with occasional arbitrage opportunities
        np.random.seed(42)  # For reproducible results
        base_price = 100.0
        
        aquis_bid = base_price + np.cumsum(np.random.normal(0, 0.001, n_snapshots))
        aquis_ask = aquis_bid + 0.02
        bme_bid = base_price + np.cumsum(np.random.normal(0, 0.001, n_snapshots))
        bme_ask = bme_bid + 0.02
        
        # Inject some arbitrage opportunities
        for i in range(0, n_snapshots, 100):
            bme_ask[i] = aquis_bid[i] - 0.01  # Create arbitrage
        
        data = pd.DataFrame({
            'AQUIS_bid': aquis_bid,
            'AQUIS_ask': aquis_ask,
            'BME_bid': bme_bid,
            'BME_ask': bme_ask
        }, index=timestamps)
        
        # Time the processing
        start_time = time.time()
        
        for i in range(0, len(data), 10):  # Process every 10th snapshot
            current_data = data.iloc[:i+1]
            self.manager.process_market_data(current_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\\nProcessed {n_snapshots} snapshots in {processing_time:.4f} seconds")
        print(f"Processing rate: {n_snapshots/processing_time:.0f} snapshots/second")
        
        # Verify some opportunities were detected
        stats = self.generator.get_statistics()
        self.assertGreater(stats['total_opportunities_detected'], 0)


def run_test_suite():
    """Run the complete test suite for arbitrage signals"""
    
    print("="*80)
    print("RUNNING ARBITRAGE SIGNAL GENERATION TEST SUITE")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)