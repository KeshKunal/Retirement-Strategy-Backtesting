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
GOOGL_prices = googl.groupby('Year')['Close'].last().to_dict()

def simulate(start_year, cpwr=0.03, min_spend=250000, buffer_years=5):
    """
    CPWR + Firewall Strategy Implementation
    
    Rules:
    1. Spending = max(CPWR × Portfolio, $250k floor)
    2. Fund spending from cash FIRST
    3. Apply market returns
    4. Refill cash buffer ONLY in UP years
    5. Never sell equity in DOWN years (unless forced emergency)
    """
    # Initialize portfolio
    cash_target = buffer_years * min_spend
    initial_total = 10000000
    cash = cash_target
    
    # Get initial prices
    prev_year = start_year - 1
    if prev_year not in SPY_prices:
        return None, f"No data for year {prev_year}"
    
    spy_price_prev = SPY_prices[prev_year]
    googl_price_prev = GOOGL_prices.get(prev_year, None)
    
    # Determine initial allocation
    initial_equity = initial_total - cash
    if googl_price_prev is not None:
        # 80/20 split
        shares_spy = (initial_equity * 0.8) / spy_price_prev
        shares_googl = (initial_equity * 0.2) / googl_price_prev
    else:
        # 100% SPY before 2004
        shares_spy = initial_equity / spy_price_prev
        shares_googl = 0
    
    results = []
    success = True
    current_year = start_year
    
    while current_year <= max(int(k) for k in SPY_prices.keys()):
        # Check if we have data for current year
        if current_year not in SPY_prices:
            break
        
        # Get current year prices
        spy_price = SPY_prices[current_year]
        googl_price = GOOGL_prices.get(current_year, None)
        
        # Calculate equity value at start of year (before any transactions)
        equity_start = shares_spy * spy_price_prev + shares_googl * (googl_price_prev if googl_price_prev else 0)
        total_start = equity_start + cash
        
        # STEP 1: Determine spending amount
        proposed = cpwr * total_start
        spending = max(proposed, min_spend)
        
        # STEP 2: Withdraw spending from cash
        sold_low = False
        forced_sale = False
        
        if cash >= spending:
            # Normal case: cash covers spending
            cash -= spending
        else:
            # Emergency: insufficient cash
            deficit = spending - cash
            cash = 0
            
            # Must sell equity (forced sale)
            if equity_start >= deficit:
                # Sell proportionally from holdings
                sell_fraction = deficit / equity_start
                shares_spy -= shares_spy * sell_fraction
                shares_googl -= shares_googl * sell_fraction
                forced_sale = True
            else:
                # Cannot even meet spending - FAILURE
                spending = cash + equity_start
                shares_spy = 0
                shares_googl = 0
                success = False
        
        # STEP 3: Apply market returns
        # Calculate equity return for market regime
        spy_return = (spy_price - spy_price_prev) / spy_price_prev if spy_price_prev > 0 else 0
        equity_return = spy_return  # Simplified: use SPY as proxy for portfolio return
        
        # Mark if this was a down year
        is_down_year = equity_return < 0
        if forced_sale and is_down_year:
            sold_low = True
        
        # Update share prices to end-of-year
        spy_price_prev = spy_price
        googl_price_prev = googl_price
        
        # Calculate equity value after returns
        equity_after_return = shares_spy * spy_price + shares_googl * (googl_price if googl_price else 0)
        
        # Apply 2% return to cash
        cash *= 1.02
        
        # STEP 4: Refill cash buffer (ONLY in UP years)
        if not is_down_year and cash < cash_target:
            refill_needed = cash_target - cash
            
            if equity_after_return >= refill_needed:
                # Sell equity proportionally to refill
                sell_fraction = refill_needed / equity_after_return
                shares_spy -= shares_spy * sell_fraction
                shares_googl -= shares_googl * sell_fraction
                cash += refill_needed
                equity_after_return -= refill_needed
        
        # STEP 5: Rebalance to 80/20 (if GOOGL exists and we just crossed into 2004)
        if googl_price is not None and shares_googl == 0 and current_year >= 2004:
            # First year with GOOGL available - rebalance to 80/20
            equity_value = shares_spy * spy_price
            target_googl = 0.2 * equity_value
            shares_googl = target_googl / googl_price
            shares_spy -= target_googl / spy_price
        
        # Annual rebalance to maintain 80/20 (if GOOGL exists)
        if googl_price is not None and shares_googl > 0:
            spy_value = shares_spy * spy_price
            googl_value = shares_googl * googl_price
            total_equity = spy_value + googl_value
            
            target_spy_value = 0.8 * total_equity
            rebalance_amount = target_spy_value - spy_value
            
            if abs(rebalance_amount) > 0.01 * total_equity:  # Only rebalance if >1% drift
                if rebalance_amount > 0:
                    # Buy SPY, sell GOOGL
                    shares_spy += rebalance_amount / spy_price
                    shares_googl -= rebalance_amount / googl_price
                else:
                    # Sell SPY, buy GOOGL
                    shares_spy += rebalance_amount / spy_price
                    shares_googl -= rebalance_amount / googl_price
        
        # Calculate final values
        equity_end = shares_spy * spy_price + shares_googl * (googl_price if googl_price else 0)
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
            'Sold Low': sold_low,
            'Forced Sale': forced_sale,
            'Equity Return': round(equity_return * 100, 2)
        })
        
        # STEP 6: Check failure conditions
        if total_end < min_spend or spending < min_spend:
            success = False
            break
        
        current_year += 1
    
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
    colors = [spending_color if not sold else '#8B0000' for sold in df['Sold Low']]
    ax3.bar(df['Year'], df['Spending']/1000, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.plot(df['Year'], df['Proposed']/1000, color='green', linewidth=2, marker='d', markersize=4, label='Proposed (CPWR)')
    ax3.axhline(y=250, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Floor ($250k)')
    ax3.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Spending ($k)', fontsize=10, fontweight='bold')
    ax3.set_title('Annual Spending (Red bars = Sold Low)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Market Regime & Actions
    ax4 = fig.add_subplot(gs[2, 0])
    down_years = df[df['Down Year'] == True]['Year']
    sold_low_years = df[df['Sold Low'] == True]['Year']
    forced_years = df[df['Forced Sale'] == True]['Year']
    
    ax4.bar(df['Year'], df['Equity Return'], color=['red' if x < 0 else 'green' for x in df['Equity Return']], 
            alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Mark problematic events
    for year in sold_low_years:
        ax4.axvline(x=year, color='darkred', linestyle='--', alpha=0.8, linewidth=2)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Equity Return (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Market Returns (Dashed lines = Sold Low)', fontsize=12, fontweight='bold')
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
    sold_low_count = df['Sold Low'].sum()
    forced_sale_count = df['Forced Sale'].sum()
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
    Sold Low:           {sold_low_count:>12}
    Forced Sales:       {forced_sale_count:>12}
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

start_years = [2010]
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
    print(f"Sold Low Events: {df['Sold Low'].sum()}")
    print(f"Forced Sales: {df['Forced Sale'].sum()}")
    
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
        sold_low_years = df[df['Sold Low'] == True]['Year']
        for year in sold_low_years:
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