"""
Step 4: The "Time Machine" (Latency Simulation)

This module simulates real-world execution latencies to understand how delays
between signal detection and execution affect arbitrage profitability.

Key Features:
- Realistic latency simulation from 0 to 100,000 microseconds
- Time machine logic to look up future market state
- Profit degradation analysis due to execution delays
- Integration with existing arbitrage signals

Author: GitHub Copilot
Date: December 2025
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from enum import Enum
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.arbitrage_signals import ArbitrageOpportunity, ArbitrageSignalGenerator


class ExecutionStatus(Enum):
    """Status of execution attempts"""
    SUCCESS = "SUCCESS"
    FAILED_NO_DATA = "FAILED_NO_DATA"
    FAILED_OPPORTUNITY_GONE = "FAILED_OPPORTUNITY_GONE"
    FAILED_NEGATIVE_PROFIT = "FAILED_NEGATIVE_PROFIT"
    FAILED_INSUFFICIENT_QUANTITY = "FAILED_INSUFFICIENT_QUANTITY"


@dataclass
class LatencyScenario:
    """Configuration for latency simulation scenario"""
    latency_microseconds: int
    description: str
    typical_use_case: str


@dataclass
class ExecutionResult:
    """Result of executing an arbitrage opportunity with latency"""
    opportunity_id: str
    detection_time: datetime
    execution_time: datetime
    latency_microseconds: int
    
    # Original opportunity data
    original_max_bid_price: float
    original_min_ask_price: float
    original_profit_per_share: float
    original_total_profit: float
    original_quantity: float
    
    # Execution time market data
    execution_max_bid_price: Optional[float]
    execution_min_ask_price: Optional[float]
    execution_profit_per_share: Optional[float]
    execution_total_profit: Optional[float]
    execution_quantity: Optional[float]
    
    # Results
    status: ExecutionStatus
    actual_profit: float
    profit_degradation: float  # Percentage loss due to latency
    slippage: float  # Price movement against us
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        if self.status == ExecutionStatus.SUCCESS:
            # Calculate profit degradation
            if self.original_total_profit > 0:
                self.profit_degradation = (
                    (self.original_total_profit - self.actual_profit) / 
                    self.original_total_profit * 100
                )
            
            # Calculate slippage
            if (self.original_max_bid_price is not None and 
                self.execution_max_bid_price is not None):
                bid_slippage = self.original_max_bid_price - self.execution_max_bid_price
                ask_slippage = (self.execution_min_ask_price or 0) - self.original_min_ask_price
                self.slippage = max(bid_slippage, ask_slippage)


class LatencySimulatorBase(ABC):
    """Abstract base class for latency simulation"""
    
    @abstractmethod
    def simulate_execution(self, opportunity: ArbitrageOpportunity, 
                          market_data: pd.DataFrame, 
                          latency_microseconds: int) -> ExecutionResult:
        """Simulate execution with given latency"""
        pass
    
    @abstractmethod
    def batch_simulate(self, opportunities: List[ArbitrageOpportunity],
                      market_data: pd.DataFrame,
                      latency_scenarios: List[int]) -> Dict[int, List[ExecutionResult]]:
        """Simulate execution across multiple latency scenarios"""
        pass


class LatencySimulator(LatencySimulatorBase):
    """
    Main latency simulation engine implementing the "Time Machine" concept
    
    Simulates realistic execution delays and their impact on arbitrage profitability
    """
    
    def __init__(self, 
                 default_latencies: Optional[List[int]] = None,
                 execution_cost_per_share: float = 0.0001):
        """
        Initialize latency simulator
        
        Args:
            default_latencies: List of latencies in microseconds to test
            execution_cost_per_share: Fixed execution cost per share traded
        """
        self.default_latencies = default_latencies or [
            0, 100, 500, 1000, 2000, 3000, 4000, 5000, 10000, 
            15000, 20000, 30000, 50000, 100000
        ]
        self.execution_cost_per_share = execution_cost_per_share
        
        # Create latency scenario descriptions
        self.latency_scenarios = self._create_latency_scenarios()
        
        # Statistics tracking
        self.simulation_count = 0
        self.successful_executions = 0
        self.total_original_profit = 0.0
        self.total_actual_profit = 0.0
    
    def _create_latency_scenarios(self) -> Dict[int, LatencyScenario]:
        """Create descriptive scenarios for each latency level"""
        scenarios = {
            0: LatencyScenario(0, "Perfect (No Latency)", "Theoretical baseline"),
            100: LatencyScenario(100, "Ultra-Low (100μs)", "Co-located HFT"),
            500: LatencyScenario(500, "Very Low (500μs)", "Exchange proximity"),
            1000: LatencyScenario(1000, "Low (1ms)", "Local datacenter"),
            2000: LatencyScenario(2000, "Good (2ms)", "Same city"),
            3000: LatencyScenario(3000, "Fair (3ms)", "Regional network"),
            4000: LatencyScenario(4000, "Moderate (4ms)", "Cross-datacenter"),
            5000: LatencyScenario(5000, "Acceptable (5ms)", "Standard retail"),
            10000: LatencyScenario(10000, "High (10ms)", "Remote connection"),
            15000: LatencyScenario(15000, "Very High (15ms)", "International"),
            20000: LatencyScenario(20000, "Poor (20ms)", "Satellite/Poor network"),
            30000: LatencyScenario(30000, "Very Poor (30ms)", "Mobile/Rural"),
            50000: LatencyScenario(50000, "Extremely Poor (50ms)", "Dial-up equivalent"),
            100000: LatencyScenario(100000, "Unusable (100ms)", "Extreme latency")
        }
        return scenarios
    
    def simulate_execution(self, opportunity: ArbitrageOpportunity, 
                          market_data: pd.DataFrame, 
                          latency_microseconds: int) -> ExecutionResult:
        """
        Simulate execution with given latency using the "Time Machine" approach
        
        Args:
            opportunity: Original arbitrage opportunity detected
            market_data: Complete market data DataFrame with microsecond timestamps
            latency_microseconds: Execution delay in microseconds
            
        Returns:
            ExecutionResult with actual execution outcome
        """
        self.simulation_count += 1
        
        # Calculate execution time
        latency_delta = timedelta(microseconds=latency_microseconds)
        execution_time = opportunity.detection_time + latency_delta
        
        # Look up market state at execution time (Time Machine!)
        execution_data = self._get_market_state_at_time(market_data, execution_time)
        
        if execution_data is None:
            # No market data available at execution time
            return ExecutionResult(
                opportunity_id=opportunity.opportunity_id,
                detection_time=opportunity.detection_time,
                execution_time=execution_time,
                latency_microseconds=latency_microseconds,
                original_max_bid_price=opportunity.max_bid_price,
                original_min_ask_price=opportunity.min_ask_price,
                original_profit_per_share=opportunity.profit_per_share,
                original_total_profit=opportunity.total_profit,
                original_quantity=opportunity.max_tradeable_quantity,
                execution_max_bid_price=None,
                execution_min_ask_price=None,
                execution_profit_per_share=None,
                execution_total_profit=None,
                execution_quantity=None,
                status=ExecutionStatus.FAILED_NO_DATA,
                actual_profit=0.0,
                profit_degradation=100.0,
                slippage=0.0
            )
        
        # Extract execution time market data
        exec_venue_data = self._extract_venue_data(execution_data)
        
        if len(exec_venue_data) < 2:
            # Insufficient venues active
            return self._create_failed_result(
                opportunity, execution_time, latency_microseconds,
                ExecutionStatus.FAILED_NO_DATA
            )
        
        # Find max bid and min ask at execution time
        exec_max_bid_venue, exec_max_bid_info = max(
            exec_venue_data.items(), 
            key=lambda x: x[1]['bid_price'] if x[1]['bid_price'] is not None else -np.inf
        )
        exec_min_ask_venue, exec_min_ask_info = min(
            exec_venue_data.items(), 
            key=lambda x: x[1]['ask_price'] if x[1]['ask_price'] is not None else np.inf
        )
        
        exec_max_bid_price = exec_max_bid_info['bid_price']
        exec_min_ask_price = exec_min_ask_info['ask_price']
        
        # Check if arbitrage opportunity still exists
        if (exec_max_bid_price is None or exec_min_ask_price is None or
            exec_max_bid_price <= exec_min_ask_price or
            exec_max_bid_venue == exec_min_ask_venue):
            
            return self._create_failed_result(
                opportunity, execution_time, latency_microseconds,
                ExecutionStatus.FAILED_OPPORTUNITY_GONE,
                exec_max_bid_price, exec_min_ask_price
            )
        
        # Calculate execution metrics
        exec_profit_per_share = exec_max_bid_price - exec_min_ask_price
        exec_quantity = min(
            exec_max_bid_info['bid_quantity'], 
            exec_min_ask_info['ask_quantity'],
            opportunity.max_tradeable_quantity  # Don't exceed original planned quantity
        )
        
        # Apply execution costs
        net_profit_per_share = exec_profit_per_share - (2 * self.execution_cost_per_share)
        
        if net_profit_per_share <= 0:
            return self._create_failed_result(
                opportunity, execution_time, latency_microseconds,
                ExecutionStatus.FAILED_NEGATIVE_PROFIT,
                exec_max_bid_price, exec_min_ask_price
            )
        
        if exec_quantity <= 0:
            return self._create_failed_result(
                opportunity, execution_time, latency_microseconds,
                ExecutionStatus.FAILED_INSUFFICIENT_QUANTITY,
                exec_max_bid_price, exec_min_ask_price
            )
        
        # Successful execution
        actual_total_profit = net_profit_per_share * exec_quantity
        self.successful_executions += 1
        self.total_original_profit += opportunity.total_profit
        self.total_actual_profit += actual_total_profit
        
        return ExecutionResult(
            opportunity_id=opportunity.opportunity_id,
            detection_time=opportunity.detection_time,
            execution_time=execution_time,
            latency_microseconds=latency_microseconds,
            original_max_bid_price=opportunity.max_bid_price,
            original_min_ask_price=opportunity.min_ask_price,
            original_profit_per_share=opportunity.profit_per_share,
            original_total_profit=opportunity.total_profit,
            original_quantity=opportunity.max_tradeable_quantity,
            execution_max_bid_price=exec_max_bid_price,
            execution_min_ask_price=exec_min_ask_price,
            execution_profit_per_share=exec_profit_per_share,
            execution_total_profit=exec_profit_per_share * exec_quantity,
            execution_quantity=exec_quantity,
            status=ExecutionStatus.SUCCESS,
            actual_profit=actual_total_profit,
            profit_degradation=0.0,  # Will be calculated in __post_init__
            slippage=0.0  # Will be calculated in __post_init__
        )
    
    def _get_market_state_at_time(self, market_data: pd.DataFrame, 
                                  target_time: datetime) -> Optional[pd.Series]:
        """
        Get market state at specific time using Time Machine logic
        
        Args:
            market_data: Complete market data with timestamp index
            target_time: Target execution time
            
        Returns:
            Market state at target time, or None if not available
        """
        if market_data.empty:
            return None
        
        # Find closest timestamp <= target_time (cannot see future)
        valid_timestamps = market_data.index[market_data.index <= target_time]
        
        if len(valid_timestamps) == 0:
            return None
        
        # Get the most recent data point
        latest_timestamp = valid_timestamps.max()
        return market_data.loc[latest_timestamp]
    
    def _extract_venue_data(self, row_data: pd.Series) -> Dict[str, Dict]:
        """Extract venue bid/ask data from market data row"""
        venue_data = {}
        venue_patterns = ['AQUIS', 'BME', 'CBOE', 'TURQUOISE']
        
        for venue in venue_patterns:
            bid_col = f"{venue}_bid"
            ask_col = f"{venue}_ask" 
            bid_qty_col = f"{venue}_bid_qty"
            ask_qty_col = f"{venue}_ask_qty"
            
            if (bid_col in row_data.index and ask_col in row_data.index and
                pd.notna(row_data[bid_col]) and pd.notna(row_data[ask_col]) and
                row_data[bid_col] > 0 and row_data[ask_col] > 0):
                
                venue_data[venue] = {
                    'bid_price': row_data[bid_col],
                    'ask_price': row_data[ask_col],
                    'bid_quantity': row_data.get(bid_qty_col, 100.0) if pd.notna(row_data.get(bid_qty_col)) else 100.0,
                    'ask_quantity': row_data.get(ask_qty_col, 100.0) if pd.notna(row_data.get(ask_qty_col)) else 100.0
                }
        
        return venue_data
    
    def _create_failed_result(self, opportunity: ArbitrageOpportunity, 
                             execution_time: datetime, 
                             latency_microseconds: int,
                             status: ExecutionStatus,
                             exec_max_bid: Optional[float] = None,
                             exec_min_ask: Optional[float] = None) -> ExecutionResult:
        """Create a failed execution result"""
        return ExecutionResult(
            opportunity_id=opportunity.opportunity_id,
            detection_time=opportunity.detection_time,
            execution_time=execution_time,
            latency_microseconds=latency_microseconds,
            original_max_bid_price=opportunity.max_bid_price,
            original_min_ask_price=opportunity.min_ask_price,
            original_profit_per_share=opportunity.profit_per_share,
            original_total_profit=opportunity.total_profit,
            original_quantity=opportunity.max_tradeable_quantity,
            execution_max_bid_price=exec_max_bid,
            execution_min_ask_price=exec_min_ask,
            execution_profit_per_share=None,
            execution_total_profit=None,
            execution_quantity=None,
            status=status,
            actual_profit=0.0,
            profit_degradation=100.0,
            slippage=0.0
        )
    
    def batch_simulate(self, opportunities: List[ArbitrageOpportunity],
                      market_data: pd.DataFrame,
                      latency_scenarios: Optional[List[int]] = None) -> Dict[int, List[ExecutionResult]]:
        """
        Simulate execution across multiple latency scenarios
        
        Args:
            opportunities: List of arbitrage opportunities to test
            market_data: Complete market data DataFrame
            latency_scenarios: List of latencies in microseconds (uses default if None)
            
        Returns:
            Dictionary mapping latency to list of execution results
        """
        if latency_scenarios is None:
            latency_scenarios = self.default_latencies
        
        print(f"🕐 Simulating {len(opportunities)} opportunities across {len(latency_scenarios)} latency scenarios...")
        
        results = {}
        
        for latency in latency_scenarios:
            scenario = self.latency_scenarios.get(latency, 
                LatencyScenario(latency, f"Custom ({latency}μs)", "Custom scenario"))
            
            print(f"   Testing {scenario.description} ({latency}μs)...")
            
            latency_results = []
            for opportunity in opportunities:
                result = self.simulate_execution(opportunity, market_data, latency)
                latency_results.append(result)
            
            results[latency] = latency_results
            
            # Print summary for this latency
            successful = sum(1 for r in latency_results if r.status == ExecutionStatus.SUCCESS)
            success_rate = (successful / len(latency_results)) * 100 if latency_results else 0
            print(f"      Success rate: {success_rate:.1f}% ({successful}/{len(latency_results)})")
        
        return results
    
    def get_latency_performance_summary(self, batch_results: Dict[int, List[ExecutionResult]]) -> pd.DataFrame:
        """
        Generate performance summary across all latency scenarios
        
        Args:
            batch_results: Results from batch_simulate
            
        Returns:
            DataFrame with performance metrics by latency
        """
        summary_data = []
        
        for latency, results in batch_results.items():
            if not results:
                continue
            
            successful_results = [r for r in results if r.status == ExecutionStatus.SUCCESS]
            
            total_opportunities = len(results)
            successful_executions = len(successful_results)
            success_rate = (successful_executions / total_opportunities) * 100
            
            original_profit = sum(r.original_total_profit for r in results)
            actual_profit = sum(r.actual_profit for r in results)
            profit_retention = (actual_profit / original_profit) * 100 if original_profit > 0 else 0
            
            avg_profit_degradation = np.mean([r.profit_degradation for r in successful_results]) if successful_results else 100.0
            avg_slippage = np.mean([r.slippage for r in successful_results]) if successful_results else 0.0
            
            scenario = self.latency_scenarios.get(latency, 
                LatencyScenario(latency, f"Custom ({latency}μs)", "Custom"))
            
            summary_data.append({
                'latency_microseconds': latency,
                'latency_ms': latency / 1000,
                'description': scenario.description,
                'use_case': scenario.typical_use_case,
                'total_opportunities': total_opportunities,
                'successful_executions': successful_executions,
                'success_rate_pct': success_rate,
                'original_total_profit': original_profit,
                'actual_total_profit': actual_profit,
                'profit_retention_pct': profit_retention,
                'avg_profit_degradation_pct': avg_profit_degradation,
                'avg_slippage': avg_slippage
            })
        
        return pd.DataFrame(summary_data).sort_values('latency_microseconds')
    
    def get_statistics(self) -> Dict:
        """Get simulator performance statistics"""
        success_rate = (self.successful_executions / max(1, self.simulation_count)) * 100
        profit_retention = (self.total_actual_profit / max(1, self.total_original_profit)) * 100
        
        return {
            'total_simulations': self.simulation_count,
            'successful_executions': self.successful_executions,
            'success_rate_pct': success_rate,
            'total_original_profit': self.total_original_profit,
            'total_actual_profit': self.total_actual_profit,
            'profit_retention_pct': profit_retention,
            'latency_scenarios_tested': len(self.default_latencies)
        }


class LatencyAnalyzer:
    """
    High-level analyzer for latency impact on arbitrage strategies
    """
    
    def __init__(self, simulator: LatencySimulator):
        """
        Initialize analyzer with latency simulator
        
        Args:
            simulator: Configured LatencySimulator instance
        """
        self.simulator = simulator
        self.analysis_results = {}
    
    def analyze_latency_impact(self, opportunities: List[ArbitrageOpportunity],
                              market_data: pd.DataFrame) -> Dict:
        """
        Comprehensive latency impact analysis
        
        Args:
            opportunities: List of arbitrage opportunities
            market_data: Market data for simulation
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        print("📊 LATENCY IMPACT ANALYSIS")
        print("="*50)
        
        # Run batch simulation
        batch_results = self.simulator.batch_simulate(opportunities, market_data)
        
        # Generate performance summary
        performance_summary = self.simulator.get_latency_performance_summary(batch_results)
        
        # Calculate additional metrics
        critical_latency = self._find_critical_latency(performance_summary)
        latency_tiers = self._categorize_latency_tiers(performance_summary)
        profitability_threshold = self._find_profitability_threshold(performance_summary)
        
        analysis = {
            'batch_results': batch_results,
            'performance_summary': performance_summary,
            'critical_latency_microseconds': critical_latency,
            'latency_tiers': latency_tiers,
            'profitability_threshold_microseconds': profitability_threshold,
            'total_opportunities': len(opportunities),
            'simulator_stats': self.simulator.get_statistics()
        }
        
        self.analysis_results = analysis
        
        print(f"✓ Analysis complete: {len(opportunities)} opportunities across {len(batch_results)} latency scenarios")
        print(f"  Critical latency: {critical_latency}μs ({critical_latency/1000:.1f}ms)")
        print(f"  Profitability threshold: {profitability_threshold}μs ({profitability_threshold/1000:.1f}ms)")
        
        return analysis
    
    def _find_critical_latency(self, performance_summary: pd.DataFrame) -> int:
        """Find latency where success rate drops below 50%"""
        below_50_pct = performance_summary[performance_summary['success_rate_pct'] < 50]
        if not below_50_pct.empty:
            return below_50_pct.iloc[0]['latency_microseconds']
        return performance_summary.iloc[-1]['latency_microseconds']  # Return max latency if all above 50%
    
    def _categorize_latency_tiers(self, performance_summary: pd.DataFrame) -> Dict[str, List[int]]:
        """Categorize latencies into performance tiers"""
        excellent = []  # >90% success
        good = []       # 70-90% success  
        fair = []       # 50-70% success
        poor = []       # <50% success
        
        for _, row in performance_summary.iterrows():
            latency = row['latency_microseconds']
            success_rate = row['success_rate_pct']
            
            if success_rate >= 90:
                excellent.append(latency)
            elif success_rate >= 70:
                good.append(latency)
            elif success_rate >= 50:
                fair.append(latency)
            else:
                poor.append(latency)
        
        return {
            'excellent': excellent,
            'good': good,
            'fair': fair,
            'poor': poor
        }
    
    def _find_profitability_threshold(self, performance_summary: pd.DataFrame) -> int:
        """Find latency where profit retention drops below 10%"""
        below_10_pct = performance_summary[performance_summary['profit_retention_pct'] < 10]
        if not below_10_pct.empty:
            return below_10_pct.iloc[0]['latency_microseconds']
        return performance_summary.iloc[-1]['latency_microseconds']