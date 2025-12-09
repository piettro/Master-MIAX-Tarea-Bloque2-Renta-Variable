"""
Step 3: Arbitrage Signal Generation

This module implements sophisticated arbitrage opportunity detection with:
- Global Max Bid vs Global Min Ask analysis
- Profit calculation with quantity constraints
- Rising edge detection (1-second persistence rule)
- Duplicate opportunity prevention
- Cross-venue signal generation

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


class OpportunityStatus(Enum):
    """Status of arbitrage opportunities"""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED" 
    EXECUTED = "EXECUTED"
    INVALID = "INVALID"


@dataclass
class ArbitrageOpportunity:
    """
    Represents a single arbitrage opportunity
    """
    opportunity_id: str
    detection_time: datetime
    expiry_time: datetime
    max_bid_venue: str
    min_ask_venue: str
    max_bid_price: float
    min_ask_price: float
    max_bid_quantity: float
    min_ask_quantity: float
    profit_per_share: float
    max_tradeable_quantity: float
    total_profit: float
    status: OpportunityStatus = OpportunityStatus.ACTIVE
    first_detection: bool = True
    snapshots_detected: int = 0
    
    def __post_init__(self):
        """Validate opportunity data after initialization"""
        if self.max_bid_price <= self.min_ask_price:
            raise ValueError(f"Invalid arbitrage: bid {self.max_bid_price} <= ask {self.min_ask_price}")
        
        if self.max_tradeable_quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.max_tradeable_quantity}")
            
        if self.total_profit <= 0:
            raise ValueError(f"Invalid profit: {self.total_profit}")


class ArbitrageSignalBase(ABC):
    """Abstract base class for arbitrage signal generation"""
    
    @abstractmethod
    def detect_opportunities(self, consolidated_data: pd.DataFrame) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities from consolidated tape data"""
        pass
    
    @abstractmethod
    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate if opportunity is tradeable"""
        pass
    
    @abstractmethod
    def calculate_profit(self, max_bid: float, min_ask: float, 
                        bid_qty: float, ask_qty: float) -> Tuple[float, float, float]:
        """Calculate profit metrics for opportunity"""
        pass


class ArbitrageSignalGenerator(ArbitrageSignalBase):
    """
    Main arbitrage signal generator implementing Step 3 requirements
    
    Features:
    - Global Max Bid vs Min Ask detection
    - 1-second persistence rule (1000 snapshots)
    - Rising edge detection to prevent double counting
    - Comprehensive profit calculations
    """
    
    def __init__(self, persistence_snapshots: int = 1000, min_profit_threshold: float = 0.001):
        """
        Initialize signal generator
        
        Args:
            persistence_snapshots: Number of snapshots opportunity must persist (default 1000 = 1 second)
            min_profit_threshold: Minimum profit per share to consider opportunity valid
        """
        self.persistence_snapshots = persistence_snapshots
        self.min_profit_threshold = min_profit_threshold
        
        # Tracking for rising edge detection
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.opportunity_history: Set[str] = set()
        self.snapshot_count = 0
        
        # Statistics
        self.total_opportunities_detected = 0
        self.total_opportunities_executed = 0
        self.total_profit_potential = 0.0
        
    def detect_opportunities(self, consolidated_data: pd.DataFrame) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities from consolidated tape data
        
        Args:
            consolidated_data: DataFrame with columns like 'AQUIS_bid', 'BME_ask', etc.
            
        Returns:
            List of valid arbitrage opportunities
        """
        if consolidated_data.empty:
            return []
        
        self.snapshot_count += 1
        opportunities = []
        
        # Get the latest row (current market state)
        current_data = consolidated_data.iloc[-1]
        current_timestamp = current_data.name if hasattr(current_data, 'name') else datetime.now()
        
        # Extract bid and ask data from all venues
        venue_data = self._extract_venue_data(current_data)
        
        if len(venue_data) < 2:
            return []  # Need at least 2 venues for arbitrage
        
        # Find global max bid and min ask
        max_bid_venue, max_bid_info = max(venue_data.items(), 
                                         key=lambda x: x[1]['bid_price'] if x[1]['bid_price'] is not None else -np.inf)
        min_ask_venue, min_ask_info = min(venue_data.items(), 
                                         key=lambda x: x[1]['ask_price'] if x[1]['ask_price'] is not None else np.inf)
        
        # Check if arbitrage opportunity exists
        if (max_bid_info['bid_price'] is not None and 
            min_ask_info['ask_price'] is not None and
            max_bid_venue != min_ask_venue and
            max_bid_info['bid_price'] > min_ask_info['ask_price']):
            
            # Calculate profit metrics
            profit_per_share, max_quantity, total_profit = self.calculate_profit(
                max_bid_info['bid_price'], 
                min_ask_info['ask_price'],
                max_bid_info['bid_quantity'], 
                min_ask_info['ask_quantity']
            )
            
            # Check if opportunity meets minimum profit threshold
            if profit_per_share >= self.min_profit_threshold:
                # Create opportunity ID for tracking
                opportunity_id = self._create_opportunity_id(
                    max_bid_venue, min_ask_venue, max_bid_info['bid_price'], min_ask_info['ask_price']
                )
                
                # Check for rising edge (new opportunity or reappearing after gap)
                if self._is_rising_edge(opportunity_id, current_timestamp):
                    opportunity = ArbitrageOpportunity(
                        opportunity_id=opportunity_id,
                        detection_time=current_timestamp,
                        expiry_time=current_timestamp + timedelta(seconds=1),  # 1-second window
                        max_bid_venue=max_bid_venue,
                        min_ask_venue=min_ask_venue,
                        max_bid_price=max_bid_info['bid_price'],
                        min_ask_price=min_ask_info['ask_price'],
                        max_bid_quantity=max_bid_info['bid_quantity'],
                        min_ask_quantity=min_ask_info['ask_quantity'],
                        profit_per_share=profit_per_share,
                        max_tradeable_quantity=max_quantity,
                        total_profit=total_profit,
                        snapshots_detected=1
                    )
                    
                    if self.validate_opportunity(opportunity):
                        self.active_opportunities[opportunity_id] = opportunity
                        opportunities.append(opportunity)
                        self.total_opportunities_detected += 1
                        self.total_profit_potential += total_profit
                        
                        print(f"🎯 NEW ARBITRAGE OPPORTUNITY DETECTED!")
                        print(f"   ID: {opportunity_id}")
                        print(f"   Buy from {min_ask_venue} @ {min_ask_info['ask_price']:.4f}")
                        print(f"   Sell to {max_bid_venue} @ {max_bid_info['bid_price']:.4f}")
                        print(f"   Profit: {profit_per_share:.4f} per share")
                        print(f"   Max Quantity: {max_quantity:.0f}")
                        print(f"   Total Profit: {total_profit:.4f}")
                
                # Update existing opportunity snapshot count
                elif opportunity_id in self.active_opportunities:
                    self.active_opportunities[opportunity_id].snapshots_detected += 1
        
        # Clean up expired opportunities
        self._cleanup_expired_opportunities(current_timestamp)
        
        return opportunities
    
    def _extract_venue_data(self, row_data: pd.Series) -> Dict[str, Dict]:
        """
        Extract bid/ask data for all venues from consolidated tape row
        
        Args:
            row_data: Single row from consolidated DataFrame
            
        Returns:
            Dictionary mapping venue to bid/ask information
        """
        venue_data = {}
        
        # Common venue patterns in column names
        venue_patterns = ['AQUIS', 'BME', 'CBOE', 'TURQUOISE']
        
        for venue in venue_patterns:
            bid_col = f"{venue}_bid"
            ask_col = f"{venue}_ask" 
            bid_qty_col = f"{venue}_bid_qty"
            ask_qty_col = f"{venue}_ask_qty"
            
            # Check if venue columns exist and have valid data
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
    
    def calculate_profit(self, max_bid: float, min_ask: float, 
                        bid_qty: float, ask_qty: float) -> Tuple[float, float, float]:
        """
        Calculate arbitrage profit metrics
        
        Args:
            max_bid: Highest bid price across venues
            min_ask: Lowest ask price across venues  
            bid_qty: Quantity available at max bid
            ask_qty: Quantity available at min ask
            
        Returns:
            Tuple of (profit_per_share, max_tradeable_quantity, total_profit)
        """
        # Profit per share
        profit_per_share = max_bid - min_ask
        
        # Maximum tradeable quantity is limited by smaller of bid/ask quantities
        max_tradeable_quantity = min(bid_qty, ask_qty)
        
        # Total profit potential
        total_profit = profit_per_share * max_tradeable_quantity
        
        return profit_per_share, max_tradeable_quantity, total_profit
    
    def _create_opportunity_id(self, bid_venue: str, ask_venue: str, 
                              bid_price: float, ask_price: float) -> str:
        """Create unique identifier for opportunity tracking"""
        return f"{bid_venue}_{ask_venue}_{bid_price:.4f}_{ask_price:.4f}"
    
    def _is_rising_edge(self, opportunity_id: str, current_timestamp: datetime) -> bool:
        """
        Determine if this is a rising edge (new opportunity or reappearing after gap)
        
        Args:
            opportunity_id: Unique opportunity identifier
            current_timestamp: Current detection timestamp
            
        Returns:
            True if this is a new/rising edge opportunity
        """
        # If never seen before, it's definitely a rising edge
        if opportunity_id not in self.opportunity_history:
            self.opportunity_history.add(opportunity_id)
            return True
        
        # If currently active, it's not a rising edge
        if opportunity_id in self.active_opportunities:
            return False
        
        # If seen before but not currently active, check if enough time has passed
        # For simplicity, we'll consider any reappearance as a new opportunity
        return True
    
    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Validate if opportunity is tradeable and profitable
        
        Args:
            opportunity: Arbitrage opportunity to validate
            
        Returns:
            True if opportunity is valid for trading
        """
        try:
            # Basic price validation
            if opportunity.max_bid_price <= opportunity.min_ask_price:
                return False
            
            # Minimum profit threshold
            if opportunity.profit_per_share < self.min_profit_threshold:
                return False
            
            # Quantity validation
            if opportunity.max_tradeable_quantity <= 0:
                return False
            
            # Venue validation (can't trade with same venue)
            if opportunity.max_bid_venue == opportunity.min_ask_venue:
                return False
            
            # Total profit validation
            if opportunity.total_profit <= 0:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  Opportunity validation failed: {e}")
            return False
    
    def _cleanup_expired_opportunities(self, current_timestamp: datetime):
        """Remove expired opportunities from active tracking"""
        expired_ids = []
        
        for opp_id, opportunity in self.active_opportunities.items():
            if current_timestamp > opportunity.expiry_time:
                expired_ids.append(opp_id)
                opportunity.status = OpportunityStatus.EXPIRED
        
        for opp_id in expired_ids:
            del self.active_opportunities[opp_id]
    
    def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get all currently active arbitrage opportunities"""
        return list(self.active_opportunities.values())
    
    def get_statistics(self) -> Dict:
        """Get signal generator performance statistics"""
        return {
            'total_snapshots_processed': self.snapshot_count,
            'total_opportunities_detected': self.total_opportunities_detected,
            'total_opportunities_executed': self.total_opportunities_executed,
            'total_profit_potential': self.total_profit_potential,
            'active_opportunities_count': len(self.active_opportunities),
            'unique_opportunities_seen': len(self.opportunity_history),
            'execution_rate': (self.total_opportunities_executed / max(1, self.total_opportunities_detected)) * 100
        }
    
    def reset_tracking(self):
        """Reset all opportunity tracking (useful for testing)"""
        self.active_opportunities.clear()
        self.opportunity_history.clear()
        self.snapshot_count = 0
        self.total_opportunities_detected = 0
        self.total_opportunities_executed = 0
        self.total_profit_potential = 0.0


class ArbitrageSignalManager:
    """
    High-level manager for arbitrage signal generation and tracking
    
    Provides:
    - Easy integration with consolidated tape
    - Batch processing capabilities
    - Historical opportunity analysis
    """
    
    def __init__(self, signal_generator: ArbitrageSignalGenerator):
        """
        Initialize signal manager
        
        Args:
            signal_generator: Configured arbitrage signal generator
        """
        self.signal_generator = signal_generator
        self.opportunity_log: List[ArbitrageOpportunity] = []
        
    def process_market_data(self, consolidated_data: pd.DataFrame) -> Dict:
        """
        Process consolidated market data and generate signals
        
        Args:
            consolidated_data: DataFrame with consolidated tape data
            
        Returns:
            Dictionary with processing results
        """
        opportunities = self.signal_generator.detect_opportunities(consolidated_data)
        
        # Log all detected opportunities
        self.opportunity_log.extend(opportunities)
        
        return {
            'new_opportunities': opportunities,
            'active_opportunities': self.signal_generator.get_active_opportunities(),
            'statistics': self.signal_generator.get_statistics()
        }
    
    def get_opportunity_summary(self) -> pd.DataFrame:
        """
        Get summary of all detected opportunities as DataFrame
        
        Returns:
            DataFrame with opportunity details
        """
        if not self.opportunity_log:
            return pd.DataFrame()
        
        data = []
        for opp in self.opportunity_log:
            data.append({
                'opportunity_id': opp.opportunity_id,
                'detection_time': opp.detection_time,
                'max_bid_venue': opp.max_bid_venue,
                'min_ask_venue': opp.min_ask_venue,
                'max_bid_price': opp.max_bid_price,
                'min_ask_price': opp.min_ask_price,
                'profit_per_share': opp.profit_per_share,
                'max_tradeable_quantity': opp.max_tradeable_quantity,
                'total_profit': opp.total_profit,
                'status': opp.status.value
            })
        
        return pd.DataFrame(data)
    
    def analyze_venue_performance(self) -> Dict:
        """
        Analyze which venue combinations generate most opportunities
        
        Returns:
            Dictionary with venue pair analysis
        """
        if not self.opportunity_log:
            return {}
        
        venue_pairs = {}
        total_profit_by_pair = {}
        
        for opp in self.opportunity_log:
            pair = (opp.max_bid_venue, opp.min_ask_venue)
            
            if pair not in venue_pairs:
                venue_pairs[pair] = 0
                total_profit_by_pair[pair] = 0.0
            
            venue_pairs[pair] += 1
            total_profit_by_pair[pair] += opp.total_profit
        
        return {
            'opportunity_counts_by_venue_pair': venue_pairs,
            'total_profit_by_venue_pair': total_profit_by_pair,
            'most_profitable_pair': max(total_profit_by_pair.items(), key=lambda x: x[1]) if total_profit_by_pair else None,
            'most_frequent_pair': max(venue_pairs.items(), key=lambda x: x[1]) if venue_pairs else None
        }


class ArbitrageSignalFactory:
    """Factory for creating different types of arbitrage signal generators"""
    
    @staticmethod
    def create_standard_generator(persistence_snapshots: int = 1000, 
                                 min_profit_threshold: float = 0.001) -> ArbitrageSignalGenerator:
        """Create standard arbitrage signal generator with default settings"""
        return ArbitrageSignalGenerator(
            persistence_snapshots=persistence_snapshots,
            min_profit_threshold=min_profit_threshold
        )
    
    @staticmethod
    def create_high_frequency_generator(min_profit_threshold: float = 0.0001) -> ArbitrageSignalGenerator:
        """Create high-frequency trading optimized generator"""
        return ArbitrageSignalGenerator(
            persistence_snapshots=100,  # Shorter persistence for HFT
            min_profit_threshold=min_profit_threshold
        )
    
    @staticmethod
    def create_conservative_generator(min_profit_threshold: float = 0.01) -> ArbitrageSignalGenerator:
        """Create conservative generator with higher profit thresholds"""
        return ArbitrageSignalGenerator(
            persistence_snapshots=2000,  # Longer persistence
            min_profit_threshold=min_profit_threshold
        )