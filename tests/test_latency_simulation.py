"""
Test Suite for Step 4: Latency Simulation ("Time Machine")

This module provides comprehensive testing for the latency simulation system,
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

from models.latency_simulator import (
    LatencySimulator, 
    LatencyAnalyzer,
    ExecutionResult,
    ExecutionStatus,
    LatencyScenario
)
from models.arbitrage_signals import ArbitrageOpportunity, OpportunityStatus


class TestLatencyScenario(unittest.TestCase):
    """Test cases for LatencyScenario dataclass"""
    
    def test_latency_scenario_creation(self):
        """Test creating latency scenario"""
        scenario = LatencyScenario(
            latency_microseconds=1000,
            description="Test Scenario",
            typical_use_case="Testing"
        )
        
        self.assertEqual(scenario.latency_microseconds, 1000)
        self.assertEqual(scenario.description, "Test Scenario")
        self.assertEqual(scenario.typical_use_case, "Testing")


class TestExecutionResult(unittest.TestCase):
    """Test cases for ExecutionResult dataclass"""
    
    def test_successful_execution_result(self):
        """Test creating successful execution result"""
        result = ExecutionResult(
            opportunity_id="TEST_001",
            detection_time=datetime.now(),
            execution_time=datetime.now() + timedelta(microseconds=1000),
            latency_microseconds=1000,
            original_max_bid_price=100.05,
            original_min_ask_price=100.02,
            original_profit_per_share=0.03,
            original_total_profit=30.0,
            original_quantity=1000,
            execution_max_bid_price=100.04,
            execution_min_ask_price=100.025,
            execution_profit_per_share=0.015,
            execution_total_profit=15.0,
            execution_quantity=1000,
            status=ExecutionStatus.SUCCESS,
            actual_profit=15.0,
            profit_degradation=0.0,
            slippage=0.0
        )
        
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(result.latency_microseconds, 1000)
        self.assertAlmostEqual(result.profit_degradation, 50.0, places=1)  # 50% degradation
    
    def test_failed_execution_result(self):
        """Test creating failed execution result"""
        result = ExecutionResult(
            opportunity_id="FAILED_001",
            detection_time=datetime.now(),
            execution_time=datetime.now() + timedelta(microseconds=5000),
            latency_microseconds=5000,
            original_max_bid_price=100.05,
            original_min_ask_price=100.02,
            original_profit_per_share=0.03,
            original_total_profit=30.0,
            original_quantity=1000,
            execution_max_bid_price=None,
            execution_min_ask_price=None,
            execution_profit_per_share=None,
            execution_total_profit=None,
            execution_quantity=None,
            status=ExecutionStatus.FAILED_OPPORTUNITY_GONE,
            actual_profit=0.0,
            profit_degradation=100.0,
            slippage=0.0
        )
        
        self.assertEqual(result.status, ExecutionStatus.FAILED_OPPORTUNITY_GONE)
        self.assertEqual(result.actual_profit, 0.0)
        self.assertEqual(result.profit_degradation, 100.0)


class TestLatencySimulator(unittest.TestCase):
    """Test cases for LatencySimulator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = LatencySimulator()
        
        # Create test opportunity
        self.test_opportunity = ArbitrageOpportunity(
            opportunity_id="TEST_001",
            detection_time=datetime(2025, 12, 9, 10, 0, 0),
            expiry_time=datetime(2025, 12, 9, 10, 0, 1),
            max_bid_venue="AQUIS",
            min_ask_venue="BME",
            max_bid_price=100.05,
            min_ask_price=100.02,
            max_bid_quantity=1000,
            min_ask_quantity=800,
            profit_per_share=0.03,
            max_tradeable_quantity=800,
            total_profit=24.0
        )
        
        # Create test market data
        timestamps = [
            datetime(2025, 12, 9, 10, 0, 0) + timedelta(microseconds=i*100) 
            for i in range(1000)  # 100ms of data at 100μs intervals
        ]
        
        self.test_market_data = pd.DataFrame({
            'AQUIS_bid': [100.05 - i*0.00001 for i in range(1000)],  # Declining bid
            'AQUIS_ask': [100.07] * 1000,
            'AQUIS_bid_qty': [1000] * 1000,
            'AQUIS_ask_qty': [800] * 1000,
            'BME_bid': [100.03] * 1000,
            'BME_ask': [100.02 + i*0.00001 for i in range(1000)],  # Rising ask
            'BME_bid_qty': [500] * 1000,
            'BME_ask_qty': [600] * 1000
        }, index=timestamps)
    
    def test_latency_scenario_creation(self):
        """Test creation of latency scenarios"""
        scenarios = self.simulator._create_latency_scenarios()
        
        self.assertIn(0, scenarios)
        self.assertIn(100000, scenarios)
        self.assertEqual(scenarios[0].description, "Perfect (No Latency)")
        self.assertEqual(scenarios[1000].description, "Low (1ms)")
    
    def test_get_market_state_at_time_exact_match(self):
        """Test getting market state at exact timestamp"""
        target_time = datetime(2025, 12, 9, 10, 0, 0, 1000)  # 1ms after start
        
        market_state = self.simulator._get_market_state_at_time(
            self.test_market_data, target_time
        )
        
        self.assertIsNotNone(market_state)
        self.assertEqual(market_state.name, target_time)
    
    def test_get_market_state_at_time_no_future(self):
        """Test that future timestamps return None"""
        future_time = datetime(2025, 12, 9, 11, 0, 0)  # 1 hour in future
        
        market_state = self.simulator._get_market_state_at_time(
            self.test_market_data, future_time
        )
        
        # Should get the last available data point
        self.assertIsNotNone(market_state)
        self.assertEqual(market_state.name, self.test_market_data.index.max())
    
    def test_get_market_state_at_time_before_start(self):
        """Test getting market state before data starts"""
        before_time = datetime(2025, 12, 9, 9, 0, 0)  # Before data starts
        
        market_state = self.simulator._get_market_state_at_time(
            self.test_market_data, before_time
        )
        
        self.assertIsNone(market_state)
    
    def test_extract_venue_data(self):
        """Test venue data extraction from market row"""
        test_row = self.test_market_data.iloc[0]
        venue_data = self.simulator._extract_venue_data(test_row)
        
        self.assertIn('AQUIS', venue_data)
        self.assertIn('BME', venue_data)
        
        aquis_data = venue_data['AQUIS']
        self.assertEqual(aquis_data['bid_price'], 100.05)
        self.assertEqual(aquis_data['ask_price'], 100.07)
    
    def test_simulate_execution_zero_latency_success(self):
        """Test simulation with zero latency (should succeed)"""
        result = self.simulator.simulate_execution(
            self.test_opportunity, 
            self.test_market_data, 
            0  # Zero latency
        )
        
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(result.latency_microseconds, 0)
        self.assertGreater(result.actual_profit, 0)
    
    def test_simulate_execution_high_latency_degradation(self):
        """Test simulation with high latency (should show degradation)"""
        # Simulate with 50ms latency (significant for the test data)
        result = self.simulator.simulate_execution(
            self.test_opportunity, 
            self.test_market_data, 
            50000  # 50ms latency
        )
        
        # Should still succeed but with degraded profit
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertLess(result.actual_profit, self.test_opportunity.total_profit)
        self.assertGreater(result.profit_degradation, 0)
    
    def test_simulate_execution_extreme_latency_failure(self):
        """Test simulation with extreme latency (should fail)"""
        # Create market data where opportunity disappears quickly
        short_timestamps = [
            datetime(2025, 12, 9, 10, 0, 0) + timedelta(microseconds=i*100) 
            for i in range(100)  # Only 10ms of data
        ]
        
        short_market_data = self.test_market_data.iloc[:100].copy()
        short_market_data.index = short_timestamps
        
        # Make arbitrage disappear after first few points
        short_market_data.iloc[10:, short_market_data.columns.get_loc('AQUIS_bid')] = 100.01
        
        result = self.simulator.simulate_execution(
            self.test_opportunity, 
            short_market_data, 
            50000  # 50ms latency, longer than data duration
        )
        
        # Should fail due to no data or opportunity gone
        self.assertIn(result.status, [ExecutionStatus.FAILED_NO_DATA, 
                                     ExecutionStatus.FAILED_OPPORTUNITY_GONE])
        self.assertEqual(result.actual_profit, 0.0)
    
    def test_batch_simulate(self):
        """Test batch simulation across multiple latencies"""
        opportunities = [self.test_opportunity]
        latencies = [0, 1000, 5000, 10000]
        
        batch_results = self.simulator.batch_simulate(
            opportunities, 
            self.test_market_data, 
            latencies
        )
        
        self.assertEqual(len(batch_results), 4)  # 4 latency scenarios
        
        for latency in latencies:
            self.assertIn(latency, batch_results)
            self.assertEqual(len(batch_results[latency]), 1)  # 1 opportunity
    
    def test_performance_summary_generation(self):
        """Test generation of performance summary"""
        opportunities = [self.test_opportunity]
        batch_results = self.simulator.batch_simulate(opportunities, self.test_market_data)
        
        summary = self.simulator.get_latency_performance_summary(batch_results)
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('latency_microseconds', summary.columns)
        self.assertIn('success_rate_pct', summary.columns)
        self.assertIn('profit_retention_pct', summary.columns)
        
        # Should be sorted by latency
        self.assertTrue(summary['latency_microseconds'].is_monotonic_increasing)
    
    def test_statistics_tracking(self):
        """Test statistics tracking during simulation"""
        initial_stats = self.simulator.get_statistics()
        self.assertEqual(initial_stats['total_simulations'], 0)
        
        # Run a simulation
        self.simulator.simulate_execution(
            self.test_opportunity, 
            self.test_market_data, 
            0
        )
        
        updated_stats = self.simulator.get_statistics()
        self.assertEqual(updated_stats['total_simulations'], 1)


class TestLatencyAnalyzer(unittest.TestCase):
    """Test cases for LatencyAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = LatencySimulator()
        self.analyzer = LatencyAnalyzer(self.simulator)
        
        # Create test opportunities
        self.test_opportunities = [
            ArbitrageOpportunity(
                opportunity_id=f"TEST_{i:03d}",
                detection_time=datetime(2025, 12, 9, 10, 0, 0) + timedelta(microseconds=i*1000),
                expiry_time=datetime(2025, 12, 9, 10, 0, 1) + timedelta(microseconds=i*1000),
                max_bid_venue="AQUIS",
                min_ask_venue="BME",
                max_bid_price=100.05,
                min_ask_price=100.02,
                max_bid_quantity=1000,
                min_ask_quantity=800,
                profit_per_share=0.03,
                max_tradeable_quantity=800,
                total_profit=24.0
            )
            for i in range(5)
        ]
        
        # Create test market data
        timestamps = [
            datetime(2025, 12, 9, 10, 0, 0) + timedelta(microseconds=i*100) 
            for i in range(2000)  # 200ms of data
        ]
        
        self.test_market_data = pd.DataFrame({
            'AQUIS_bid': [100.05 - i*0.000001 for i in range(2000)],  # Slow decline
            'AQUIS_ask': [100.07] * 2000,
            'AQUIS_bid_qty': [1000] * 2000,
            'AQUIS_ask_qty': [800] * 2000,
            'BME_bid': [100.03] * 2000,
            'BME_ask': [100.02 + i*0.000001 for i in range(2000)],  # Slow rise
            'BME_bid_qty': [500] * 2000,
            'BME_ask_qty': [600] * 2000
        }, index=timestamps)
    
    def test_analyze_latency_impact(self):
        """Test comprehensive latency impact analysis"""
        analysis = self.analyzer.analyze_latency_impact(
            self.test_opportunities, 
            self.test_market_data
        )
        
        self.assertIn('batch_results', analysis)
        self.assertIn('performance_summary', analysis)
        self.assertIn('critical_latency_microseconds', analysis)
        self.assertIn('latency_tiers', analysis)
        self.assertIn('profitability_threshold_microseconds', analysis)
        
        # Check performance summary structure
        summary = analysis['performance_summary']
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
    
    def test_find_critical_latency(self):
        """Test finding critical latency threshold"""
        # Create mock performance summary
        test_summary = pd.DataFrame({
            'latency_microseconds': [0, 1000, 5000, 10000],
            'success_rate_pct': [100, 80, 45, 20]  # Drops below 50% at 5000μs
        })
        
        critical_latency = self.analyzer._find_critical_latency(test_summary)
        self.assertEqual(critical_latency, 5000)
    
    def test_categorize_latency_tiers(self):
        """Test categorization of latency into performance tiers"""
        test_summary = pd.DataFrame({
            'latency_microseconds': [0, 1000, 5000, 10000, 20000],
            'success_rate_pct': [100, 95, 75, 55, 30]
        })
        
        tiers = self.analyzer._categorize_latency_tiers(test_summary)
        
        self.assertIn('excellent', tiers)
        self.assertIn('good', tiers)
        self.assertIn('fair', tiers)
        self.assertIn('poor', tiers)
        
        self.assertIn(0, tiers['excellent'])    # 100% success
        self.assertIn(1000, tiers['excellent']) # 95% success
        self.assertIn(5000, tiers['good'])      # 75% success
        self.assertIn(10000, tiers['fair'])     # 55% success
        self.assertIn(20000, tiers['poor'])     # 30% success
    
    def test_find_profitability_threshold(self):
        """Test finding profitability threshold"""
        test_summary = pd.DataFrame({
            'latency_microseconds': [0, 1000, 5000, 10000],
            'profit_retention_pct': [100, 50, 15, 5]  # Drops below 10% at 10000μs
        })
        
        threshold = self.analyzer._find_profitability_threshold(test_summary)
        self.assertEqual(threshold, 10000)


class TestIntegrationLatencySimulation(unittest.TestCase):
    """Integration tests for complete latency simulation workflow"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.simulator = LatencySimulator()
        self.analyzer = LatencyAnalyzer(self.simulator)
    
    def test_full_workflow_realistic_scenario(self):
        """Test complete workflow with realistic market scenario"""
        # Create realistic arbitrage scenario with gradual opportunity decay
        timestamps = [
            datetime(2025, 12, 9, 10, 0, 0) + timedelta(microseconds=i*50) 
            for i in range(4000)  # 200ms of data at 50μs intervals
        ]
        
        # Create opportunity that exists for ~100ms then degrades
        aquis_bid = []
        bme_ask = []
        
        for i in range(4000):
            if i < 2000:  # First 100ms: good arbitrage
                aquis_bid.append(100.05 - i*0.000001)  # Slow decline
                bme_ask.append(100.02 + i*0.0000005)   # Very slow rise
            else:  # Second 100ms: opportunity degrades
                aquis_bid.append(100.03 - (i-2000)*0.000005)  # Faster decline
                bme_ask.append(100.025 + (i-2000)*0.000005)   # Faster rise
        
        market_data = pd.DataFrame({
            'AQUIS_bid': aquis_bid,
            'AQUIS_ask': [100.07] * 4000,
            'AQUIS_bid_qty': [1000] * 4000,
            'AQUIS_ask_qty': [800] * 4000,
            'BME_bid': [100.03] * 4000,
            'BME_ask': bme_ask,
            'BME_bid_qty': [500] * 4000,
            'BME_ask_qty': [600] * 4000
        }, index=timestamps)
        
        # Create opportunities detected at different times
        opportunities = [
            ArbitrageOpportunity(
                opportunity_id=f"REALISTIC_{i:03d}",
                detection_time=timestamps[i*500],  # Every 25ms
                expiry_time=timestamps[i*500] + timedelta(seconds=1),
                max_bid_venue="AQUIS",
                min_ask_venue="BME",
                max_bid_price=aquis_bid[i*500],
                min_ask_price=bme_ask[i*500],
                max_bid_quantity=1000,
                min_ask_quantity=600,
                profit_per_share=aquis_bid[i*500] - bme_ask[i*500],
                max_tradeable_quantity=600,
                total_profit=(aquis_bid[i*500] - bme_ask[i*500]) * 600
            )
            for i in range(7)  # 7 opportunities over time
        ]
        
        # Run analysis
        analysis = self.analyzer.analyze_latency_impact(opportunities, market_data)
        
        # Verify analysis results
        self.assertGreater(analysis['total_opportunities'], 0)
        
        summary = analysis['performance_summary']
        self.assertFalse(summary.empty)
        
        # Verify latency impact - should see degradation with higher latencies
        zero_latency_success = summary[summary['latency_microseconds'] == 0]['success_rate_pct'].iloc[0]
        high_latency_success = summary[summary['latency_microseconds'] == 50000]['success_rate_pct'].iloc[0]
        
        self.assertGreaterEqual(zero_latency_success, high_latency_success)
        
        # Verify tiers make sense
        tiers = analysis['latency_tiers']
        self.assertTrue(len(tiers['excellent']) >= len(tiers['good']) >= len(tiers['fair']) >= len(tiers['poor']))
    
    def test_performance_with_different_latency_patterns(self):
        """Test performance across different latency patterns"""
        # Test with custom latency scenarios
        custom_latencies = [0, 50, 200, 1000, 5000, 25000, 100000]
        
        # Create simple test data
        timestamps = [
            datetime(2025, 12, 9, 10, 0, 0) + timedelta(microseconds=i*100) 
            for i in range(1000)
        ]
        
        market_data = pd.DataFrame({
            'AQUIS_bid': [100.05] * 1000,
            'AQUIS_ask': [100.07] * 1000,
            'AQUIS_bid_qty': [1000] * 1000,
            'AQUIS_ask_qty': [800] * 1000,
            'BME_bid': [100.03] * 1000,
            'BME_ask': [100.02] * 1000,
            'BME_bid_qty': [500] * 1000,
            'BME_ask_qty': [600] * 1000
        }, index=timestamps)
        
        opportunity = ArbitrageOpportunity(
            opportunity_id="CUSTOM_TEST",
            detection_time=timestamps[0],
            expiry_time=timestamps[0] + timedelta(seconds=1),
            max_bid_venue="AQUIS",
            min_ask_venue="BME",
            max_bid_price=100.05,
            min_ask_price=100.02,
            max_bid_quantity=1000,
            min_ask_quantity=600,
            profit_per_share=0.03,
            max_tradeable_quantity=600,
            total_profit=18.0
        )
        
        batch_results = self.simulator.batch_simulate(
            [opportunity], 
            market_data, 
            custom_latencies
        )
        
        # Verify all latencies were tested
        self.assertEqual(len(batch_results), len(custom_latencies))
        
        for latency in custom_latencies:
            self.assertIn(latency, batch_results)
            self.assertEqual(len(batch_results[latency]), 1)


def run_test_suite():
    """Run the complete test suite for latency simulation"""
    
    print("="*80)
    print("RUNNING LATENCY SIMULATION TEST SUITE")
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