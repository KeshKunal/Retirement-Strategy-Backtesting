import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSVs
spy = pd.read_csv('DataSet/spy.csv')
spy['Date'] = pd.to_datetime(spy['Date'])
spy['Year'] = spy['Date'].dt.year
SPY_prices = spy.groupby('Year')['Close'].last().to_dict()

googl = pd.read_csv('DataSet/googl.csv')
googl['Date'] = pd.to_datetime(googl['Date'])
googl['Year'] = googl['Date'].dt.year
GOOGL_prices = googl.groupby('Year')['Close'].last().to_dict()  # makes dictionary of end of year closing prices

def simulate(start_year):
    """
    CPWR + Firewall Strategy Implementation
    
    Strategy Definition:
    "Spending is set by CPWR with a hard floor and funded from a cash firewall; 
    equity is sold only in positive-return years to refill the buffer, and any 
    need to sell during drawdowns constitutes strategy failure."
    
    Rules:
    1. Determine market regime FIRST (UP/DOWN year)
    2. Spending = max(CPWR × Equity, $250k floor) - funded from cash ONLY
    3. Apply market returns (Equity + Cash 2%)
    4. Refill cash buffer ONLY in UP years
    5. NEVER sell equity in DOWN years - immediate failure if needed
    """
    #Constants
    cpwr=0.04
    min_spend=250000
    buffer_years=5
    
    # Initialize portfolio
    cash_target = buffer_years * min_spend
    initial_total = 10000000
    cash = cash_target
    
    # Get initial prices (year BEFORE start)
    prev_year = start_year - 1
    if prev_year not in SPY_prices:
        return None, f"No data for year {prev_year}"
    
    spy_price_prev = SPY_prices[prev_year]
    googl_price_prev = GOOGL_prices.get(prev_year, None)
    
    # Determine initial allocation
    initial_equity = initial_total - cash
    if prev_year >= 2004 and googl_price_prev is not None:
        # 80/20 split if GOOGL data exists
        shares_spy = (initial_equity * 0.8) / spy_price_prev
        shares_googl = (initial_equity * 0.2) / googl_price_prev
    else:
        # 100% SPY before 2004 or if no GOOGL data
        shares_spy = initial_equity / spy_price_prev
        shares_googl = 0
    
    results = []
    success = True
    current_year = start_year
    failure_reason = None
    
    while current_year <= max(int(k) for k in SPY_prices.keys()):
        # Check if we have data for current year
        if current_year not in SPY_prices:
            break
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Read prices and determine market regime
        # ═══════════════════════════════════════════════════════════════
        spy_price_current = SPY_prices[current_year]
        googl_price_current = GOOGL_prices.get(current_year, None)
        
        # Calculate equity value at START of year (using previous year's prices)
        equity_start = shares_spy * spy_price_prev + shares_googl * (googl_price_prev if googl_price_prev else 0)
        
        # Calculate annual equity return to determine market regime
        spy_return = (spy_price_current - spy_price_prev) / spy_price_prev if spy_price_prev > 0 else 0
        
        # CRITICAL: Determine if this is a DOWN year BEFORE any other decisions
        is_down_year = spy_return < 0
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Determine spending (CPWR base = equity value)
        # ═══════════════════════════════════════════════════════════════
        # CPWR × Equity
        proposed = cpwr * equity_start
        spending = max(proposed, min_spend)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Attempt to withdraw spending from cash
        # ═══════════════════════════════════════════════════════════════
        buffer_depleted = False
        
        if cash >= spending:
            # Normal case: cash covers spending
            cash -= spending
        else:
            # CRITICAL: Insufficient cash
            if is_down_year:
                # FAILURE: Cannot sell equity in DOWN years
                success = False
                failure_reason = f"Insufficient cash in DOWN year {current_year} (needed ${spending:,.0f}, had ${cash:,.0f})"
                spending = cash  # Spend what we have
                cash = 0
                
                # Record this year and TERMINATE
                results.append({
                    'Year': current_year,
                    'Equity': round(equity_start, 2),
                    'Cash': round(cash, 2),
                    'Total': round(equity_start + cash, 2),
                    'Proposed': round(proposed, 2),
                    'Spending': round(spending, 2),
                    'Down Year': is_down_year,
                    'Buffer Depleted': True,
                    'Refill Event': False,
                    'Equity Return': round(spy_return * 100, 2)
                })
                break
            else:
                # UP year but insufficient cash - mark buffer depletion
                buffer_depleted = True
                spending = cash  # Spend what we have for now
                cash = 0
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Apply market returns
        # ═══════════════════════════════════════════════════════════════
        # Update prices to end-of-year
        spy_price_prev = spy_price_current
        googl_price_prev = googl_price_current
        
        # Equity value after market returns
        equity_after_return = shares_spy * spy_price_current + shares_googl * (googl_price_current if googl_price_current else 0)
        
        # Apply 2% annual return to cash
        cash *= 1.02
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Refill cash buffer (ONLY in UP years)
        # ═══════════════════════════════════════════════════════════════
        refill_event = False
        
        if not is_down_year and cash < cash_target:
            # UP year and cash below target - refill allowed
            refill_needed = cash_target - cash
            
            if equity_after_return >= refill_needed:
                # Sell equity proportionally to refill buffer
                sell_fraction = refill_needed / equity_after_return
                shares_spy -= shares_spy * sell_fraction
                shares_googl -= shares_googl * sell_fraction
                
                cash += refill_needed
                equity_after_return -= refill_needed
                refill_event = True
            else:
                # Not enough equity to fully refill - this could lead to failure
                # Sell all equity if desperate, but this is likely terminal
                if equity_after_return > 0:
                    success = False
                    failure_reason = f"Insufficient equity to refill buffer in year {current_year}"
                    break
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Rebalance to 80/20 (if GOOGL available and UP year)
        # ═══════════════════════════════════════════════════════════════
        # First-time allocation to GOOGL if we just crossed into 2004+
        if current_year >= 2004 and googl_price_current is not None and shares_googl == 0 and not is_down_year:
            equity_value = shares_spy * spy_price_current
            if equity_value > 0:
                target_googl_value = 0.2 * equity_value
                shares_googl = target_googl_value / googl_price_current
                shares_spy = (equity_value - target_googl_value) / spy_price_current
                equity_after_return = equity_value  # No change in total
        
        # Annual rebalance to maintain 80/20 (only in UP years to avoid trading costs in down markets)
        if googl_price_current is not None and shares_googl > 0 and not is_down_year:
            spy_value = shares_spy * spy_price_current
            googl_value = shares_googl * googl_price_current
            total_equity = spy_value + googl_value
            
            if total_equity > 0:
                target_spy_value = 0.8 * total_equity
                drift = abs(target_spy_value - spy_value) / total_equity
                
                # Only rebalance if drift > 1%
                if drift > 0.01:
                    rebalance_amount = target_spy_value - spy_value
                    shares_spy = target_spy_value / spy_price_current
                    shares_googl = (total_equity - target_spy_value) / googl_price_current
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Calculate final values and check failure conditions
        # ═══════════════════════════════════════════════════════════════
        equity_end = shares_spy * spy_price_current + shares_googl * (googl_price_current if googl_price_current else 0)
        total_end = equity_end + cash
        
        # Record results
        results.append({
            'Year': current_year,
            'Equity': round(equity_end, 2),
            'Cash': round(cash, 2),
            'Total': round(total_end, 2),
            'Proposed': round(proposed, 2),
            'Spending': round(spending, 2),
            'Down Year': is_down_year,
            'Buffer Depleted': buffer_depleted,
            'Refill Event': refill_event,
            'Equity Return': round(spy_return * 100, 2)
        })
        
        # Check failure conditions
        if total_end <= 0:
            success = False
            failure_reason = f"Portfolio depleted in year {current_year}"
            break
        
        if total_end < min_spend:
            success = False
            failure_reason = f"Portfolio below minimum spend threshold in year {current_year}"
            break
        
        current_year += 1
    
    # Add failure reason to results if failed
    if not success and failure_reason:
        print(f"  ⚠️  {failure_reason}")
    
    return results, success

def visualize_simulation(df, start_year, success):
    """Create comprehensive visualization of the simulation"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color scheme
    equity_color = '#2E86AB'
    cash_color = '#A23B72'
    total_color = '#F18F01'
    spending_color = '#C73E1D'
    
    # 1. Portfolio Value Over Time (Stacked Area)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(df['Year'], 0, df['Equity']/1e6, alpha=0.7, color=equity_color, label='Equity')
    ax1.fill_between(df['Year'], df['Equity']/1e6, df['Total']/1e6, alpha=0.7, color=cash_color, label='Cash')
    ax1.plot(df['Year'], df['Total']/1e6, color=total_color, linewidth=2.5, label='Total Portfolio')
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Initial $10M')
    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($M)', fontsize=11, fontweight='bold')
    ax1.set_title(f'CPWR + Firewall Strategy - Starting {start_year} | {"✓ SUCCESS" if success else "✗ FAILED"}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(df['Year'].min(), df['Year'].max())
    
    # 2. Equity vs Cash Balance
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['Year'], df['Equity']/1e6, color=equity_color, linewidth=2, marker='o', markersize=3, label='Equity')
    ax2.plot(df['Year'], df['Cash']/1e6, color=cash_color, linewidth=2, marker='s', markersize=3, label='Cash')
    ax2.axhline(y=1.25, color=cash_color, linestyle=':', alpha=0.6, label='Target Cash ($1.25M)')
    ax2.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Value ($M)', fontsize=10, fontweight='bold')
    ax2.set_title('Equity vs Cash Balance', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Spending Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['#8B0000' if depleted else spending_color for depleted in df['Buffer Depleted']]
    ax3.bar(df['Year'], df['Spending']/1000, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.plot(df['Year'], df['Proposed']/1000, color='green', linewidth=2, marker='d', markersize=4, label='Proposed (CPWR)')
    ax3.axhline(y=250, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Floor ($250k)')
    ax3.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Spending ($k)', fontsize=10, fontweight='bold')
    ax3.set_title('Annual Spending (Dark Red = Buffer Depleted)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Market Regime & Actions
    ax4 = fig.add_subplot(gs[2, 0])
    down_years = df[df['Down Year'] == True]['Year']
    buffer_depleted_years = df[df['Buffer Depleted'] == True]['Year']
    refill_years = df[df['Refill Event'] == True]['Year']
    
    ax4.bar(df['Year'], df['Equity Return'], color=['red' if x < 0 else 'green' for x in df['Equity Return']], 
            alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Mark critical events
    for year in buffer_depleted_years:
        ax4.axvline(x=year, color='darkred', linestyle='--', alpha=0.8, linewidth=2)
    
    for year in refill_years:
        year_data = df[df['Year'] == year]
        ax4.scatter(year, year_data['Equity Return'].values[0], color='blue', s=80, zorder=5, marker='o')
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Equity Return (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Market Returns (Dashed = Buffer Depleted, Blue = Refill)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Summary Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate statistics
    total_spending = df['Spending'].sum()
    avg_spending = df['Spending'].mean()
    min_portfolio = df['Total'].min()
    max_portfolio = df['Total'].max()
    final_portfolio = df['Total'].iloc[-1]
    years_survived = len(df)
    buffer_depleted_count = df['Buffer Depleted'].sum()
    refill_count = df['Refill Event'].sum()
    down_year_count = df['Down Year'].sum()
    
    stats_text = f"""
    SIMULATION SUMMARY
    {'='*40}
    
    Duration: {years_survived} years ({df['Year'].min()}-{df['Year'].max()})
    Status: {"✓ SUCCESS" if success else "✗ FAILED"}
    
    PORTFOLIO METRICS
    {'─'*40}
    Initial Value:      ${10_000_000:>12,.0f}
    Final Value:        ${final_portfolio:>12,.0f}
    Min Value:          ${min_portfolio:>12,.0f}
    Max Value:          ${max_portfolio:>12,.0f}
    Growth:             {((final_portfolio/10_000_000 - 1) * 100):>11.1f}%
    
    SPENDING METRICS
    {'─'*40}
    Total Spent:        ${total_spending:>12,.0f}
    Average/Year:       ${avg_spending:>12,.0f}
    Min Spending:       ${df['Spending'].min():>12,.0f}
    Max Spending:       ${df['Spending'].max():>12,.0f}
    
    RISK EVENTS
    {'─'*40}
    Down Years:         {down_year_count:>12}
    Buffer Depleted:    {buffer_depleted_count:>12}
    Refill Events:      {refill_count:>12}
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('CPWR + Firewall Strategy: Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

# Run backtests
print("="*80)
print("CPWR + FIREWALL STRATEGY BACKTEST")
print("Initial Portfolio: $10,000,000 | CPWR: 4% | Floor: $250,000 | Buffer: 5 years")
print("="*80)
print()

start_years = [1999, 2000]
all_results = {}

for start in start_years:
    print(f"\n{'─'*80}")
    print(f"SIMULATION STARTING {start}")
    print(f"{'─'*80}")
    
    results, success = simulate(start)
    
    if results is None:
        print(f"❌ {success}")
        continue
    
    df = pd.DataFrame(results)
    all_results[start] = (df, success)
    
    # Print summary table
    print(f"\nStatus: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"Years Survived: {len(df)}")
    print(f"Final Portfolio: ${df['Total'].iloc[-1]:,.0f}")
    print(f"Buffer Depleted Events: {df['Buffer Depleted'].sum()}")
    print(f"Refill Events: {df['Refill Event'].sum()}")
    print(f"Down Years: {df['Down Year'].sum()}")
    
    # Print detailed table
    print("\nDetailed Results:")
    display_df = df.copy()
    display_df[['Equity', 'Cash', 'Total', 'Proposed', 'Spending']] = display_df[['Equity', 'Cash', 'Total', 'Proposed', 'Spending']].round(0)
    print(display_df.to_markdown(index=False, numalign="right", stralign="right", floatfmt=".0f"))
    
    # Create visualization
    fig = visualize_simulation(df, start, success)
    plt.savefig(f'simulation_{start}.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Chart saved as: simulation_{start}.png")
    plt.close(fig)

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

# Create comparison chart
if len(all_results) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('CPWR + Firewall Strategy: Multi-Scenario Comparison', fontsize=16, fontweight='bold')
    
    for idx, (start, (df, success)) in enumerate(all_results.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Plot total portfolio value
        ax.plot(df['Year'], df['Total']/1e6, linewidth=2.5, color='#F18F01', label='Total')
        ax.fill_between(df['Year'], 0, df['Equity']/1e6, alpha=0.5, color='#2E86AB', label='Equity')
        ax.fill_between(df['Year'], df['Equity']/1e6, df['Total']/1e6, alpha=0.5, color='#A23B72', label='Cash')
        
        # Mark sold low events
        buffer_depleted_years = df[df['Buffer Depleted'] == True]['Year']
        for year in buffer_depleted_years:
            year_data = df[df['Year'] == year]
            ax.scatter(year, year_data['Total'].values[0]/1e6, color='red', s=100, zorder=5, marker='X')
        
        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Portfolio Value ($M)', fontweight='bold')
        ax.set_title(f'Starting {start} | {"✓ SUCCESS" if success else "✗ FAILED"}', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(df['Year'].min(), df['Year'].max())
    
    plt.tight_layout()
    plt.savefig('comparison_all_scenarios.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comparison chart saved as: comparison_all_scenarios.png")
    plt.close(fig)