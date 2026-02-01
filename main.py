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

    # Constants
    cpwr = 0.04
    min_spend = 250000
    buffer_years = 5
    alpha_up_year = 0.03 
    alpha_down_year = 0.0 
    smooth_cap = 1.10
    
    # Initialize portfolio
    cash_target = buffer_years * min_spend
    initial_total = 10000000
    cash = cash_target
    last_spending = None  

    # Initialize portfolio
    if start_year not in SPY_prices:
        return None, f"No data for year {start_year}"
    
    spy_price_prev = SPY_prices[start_year]
    googl_price_prev = GOOGL_prices.get(start_year, None)
    
    # Determine initial allocation
    initial_equity = initial_total - cash
    if start_year >= 2004 and googl_price_prev is not None:
        # 80/20 split if GOOGL data exists
        shares_spy = (initial_equity * 0.8) / spy_price_prev
        shares_googl = (initial_equity * 0.2) / googl_price_prev
    else:
        # 100% SPY before 2004 or if no GOOGL data
        shares_spy = initial_equity / spy_price_prev
        shares_googl = 0
    
    results = []
    success = True
    current_year = start_year + 1  # Start simulation from NEXT year
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
        # STEP 2: Determine spending (liquidity-constrained)
        # ═══════════════════════════════════════════════════════════════
        # 2A: DESIRED - What you'd like to spend (CPWR aspiration)
        desired = max(cpwr * equity_start, min_spend)
        
        # 2B: LIQUID - What you can actually access this year
        alpha = alpha_down_year if is_down_year else alpha_up_year
        liquid = cash + alpha * equity_start
        
        # 2C: SMOOTH - What's reasonable year-over-year (lifestyle stability)
        if last_spending is None:
            smooth = desired  # First year: no constraint
        else:
            smooth = last_spending * smooth_cap  # Cap year-over-year growth
        
        # 2D: FINAL SPENDING - Take minimum of all constraints
        spending = min(desired, liquid, smooth)
        
        # Store for tracking
        proposed = desired  # For reporting
        binding_constraint = 'desired' if spending == desired else ('liquid' if spending == liquid else 'smooth')
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Execute spending (withdraw from cash and/or equity)
        # ═══════════════════════════════════════════════════════════════
        # Check for true failure: liquidity < floor
        if liquid < min_spend:
            success = False
            failure_reason = f"Liquidity ${liquid:,.0f} below floor ${min_spend:,.0f} in year {current_year}"
            spending = liquid  # Spend what we can
            
            # Record failure year and terminate
            results.append({
                'Year': current_year,
                'Equity': round(equity_start, 2),
                'Cash': round(cash, 2),
                'Total': round(equity_start + cash, 2),
                'Proposed': round(proposed, 2),
                'Spending': round(spending, 2),
                'Binding': binding_constraint,
                'Down Year': is_down_year,
                'Buffer Depleted': True,
                'Sold Equity': False,
                'Refill Event': False,
                'Equity Return': round(spy_return * 100, 2)
            })
            break
        
        # Normal case: execute spending from cash and/or equity
        buffer_depleted = False
        sold_equity_for_spending = False
        
        # First, withdraw from cash
        amount_from_cash = min(spending, cash)
        cash -= amount_from_cash
        
        # If spending > cash, sell equity (only possible in UP years due to alpha=0 in down years)
        if spending > amount_from_cash:
            amount_from_equity = spending - amount_from_cash
            
            # Sell equity proportionally
            if equity_start > 0:
                sell_fraction = amount_from_equity / equity_start
                shares_spy -= shares_spy * sell_fraction
                shares_googl -= shares_googl * sell_fraction
                sold_equity_for_spending = True
            
            # Mark buffer depletion if we needed to sell equity for spending
            if amount_from_cash < min_spend:
                buffer_depleted = True
        
        # Update last_spending for next year's smoothing
        last_spending = spending
        
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
        # STEP 6: Rebalance to 80/20
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
            'Binding': binding_constraint,
            'Down Year': is_down_year,
            'Buffer Depleted': buffer_depleted,
            'Sold Equity': sold_equity_for_spending,
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
    
    # Add failure reason
    if not success and failure_reason:
        print(f"  ⚠️  {failure_reason}")
    
    return results, success

def visualize_simulation(df, start_year, success):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color scheme
    equity_color = '#2E86AB'
    cash_color = '#A23B72'
    total_color = '#F18F01'
    spending_color = '#C73E1D'
    
    # 1. Portfolio Value Over Time (Stacked Area)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(df['Year'], 0, df['Equity']/1e6, alpha=0.7, color=equity_color, label='Equity (Invested)')
    ax1.fill_between(df['Year'], df['Equity']/1e6, df['Total']/1e6, alpha=0.7, color=cash_color, label='Cash (Buffer)')
    ax1.plot(df['Year'], df['Total']/1e6, color=total_color, linewidth=2.5, label='Total Portfolio', zorder=5)
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Initial $10M')
    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($M)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Liquidity-Constrained CPWR Strategy - Starting {start_year} | {"✓ SUCCESS" if success else "✗ FAILED"}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', framealpha=0.95, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(df['Year'].min(), df['Year'].max())
    
    # Add annotations for major events
    if success:
        growth_pct = ((df['Total'].iloc[-1] / 10_000_000 - 1) * 100)
        ax1.text(0.98, 0.02, f'Growth: +{growth_pct:.1f}%', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                ha='right', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 2. Equity vs Cash Balance
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['Year'], df['Equity']/1e6, color=equity_color, linewidth=2.5, marker='o', markersize=4, label='Equity Portfolio')
    ax2.plot(df['Year'], df['Cash']/1e6, color=cash_color, linewidth=2.5, marker='s', markersize=4, label='Cash Buffer')
    ax2.axhline(y=1.25, color=cash_color, linestyle=':', alpha=0.7, linewidth=2, label='Buffer Target ($1.25M)')
    ax2.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Value ($M)', fontsize=10, fontweight='bold')
    ax2.set_title('Asset Allocation: Equity Growth vs Cash Buffer', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.95, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Spending Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['#8B0000' if depleted else spending_color for depleted in df['Buffer Depleted']]
    bars = ax3.bar(df['Year'], df['Spending']/1000, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.plot(df['Year'], df['Proposed']/1000, color='green', linewidth=2.5, marker='d', markersize=5, 
             label='Desired (CPWR × Equity)', zorder=5)
    ax3.axhline(y=250, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Floor ($250k)')
    
    # Add legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        ax3.plot([], [], color='green', linewidth=2.5, marker='d', markersize=5, label='Desired (CPWR × Equity)')[0],
        ax3.plot([], [], color='red', linestyle='--', linewidth=2, label='Floor ($250k)')[0],
        Patch(facecolor=spending_color, alpha=0.7, label='Normal Spending'),
        Patch(facecolor='#8B0000', alpha=0.7, label='Buffer Depleted (Sold Equity)')
    ]
    
    ax3.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Spending ($k)', fontsize=10, fontweight='bold')
    ax3.set_title('Annual Spending: Desired vs Actual', fontsize=12, fontweight='bold')
    ax3.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 4. Market Regime & Actions
    ax4 = fig.add_subplot(gs[2, 0])
    down_years = df[df['Down Year'] == True]['Year']
    buffer_depleted_years = df[df['Buffer Depleted'] == True]['Year']
    refill_years = df[df['Refill Event'] == True]['Year']
    
    # Bar chart for returns
    bars = ax4.bar(df['Year'], df['Equity Return'], 
                   color=['#DC143C' if x < 0 else '#228B22' for x in df['Equity Return']], 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Mark buffer depletion years with background shading
    for year in buffer_depleted_years:
        year_data = df[df['Year'] == year]
        ret = year_data['Equity Return'].values[0]
        ax4.axvspan(year-0.4, year+0.4, alpha=0.2, color='orange', zorder=0)
    
    # Mark refill events with prominent markers
    for year in refill_years:
        year_data = df[df['Year'] == year]
        ax4.scatter(year, year_data['Equity Return'].values[0], 
                   color='blue', s=150, zorder=5, marker='*', 
                   edgecolors='white', linewidths=1.5)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax4.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Annual Return (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Market Returns & Strategy Actions', fontsize=12, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#228B22', alpha=0.7, label='UP Year (Gains)'),
        Patch(facecolor='#DC143C', alpha=0.7, label='DOWN Year (Losses)'),
        Patch(facecolor='orange', alpha=0.2, label='Buffer Depleted'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', 
               markersize=12, label='Buffer Refilled', markeredgecolor='white')
    ]
    ax4.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
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
    
    # Count binding constraints
    binding_counts = df['Binding'].value_counts()
    desired_count = binding_counts.get('desired', 0)
    liquid_count = binding_counts.get('liquid', 0)
    smooth_count = binding_counts.get('smooth', 0)
    
    stats_text = f"""
    {'SIMULATION RESULTS':^40}
    {'='*40}
    
    Period: {df['Year'].min()}-{df['Year'].max()} ({years_survived} years)
    Status: {"✓ SUCCESS" if success else "✗ FAILED"}
    
    {'PORTFOLIO PERFORMANCE':^40}
    {'─'*40}
    Initial Value:      ${10_000_000:>12,.0f}
    Final Value:        ${final_portfolio:>12,.0f}
    Peak Value:         ${max_portfolio:>12,.0f}
    Lowest Value:       ${min_portfolio:>12,.0f}
    Total Growth:       {((final_portfolio/10_000_000 - 1) * 100):>11.1f}%
    
    {'SPENDING BEHAVIOR':^40}
    {'─'*40}
    Total Spent:        ${total_spending:>12,.0f}
    Avg Per Year:       ${avg_spending:>12,.0f}
    Minimum Spend:      ${df['Spending'].min():>12,.0f}
    Maximum Spend:      ${df['Spending'].max():>12,.0f}
    
    {'CONSTRAINT ANALYSIS':^40}
    {'─'*40}
    Desired Binding:    {desired_count:>12} years
    Liquid Binding:     {liquid_count:>12} years
    Smooth Binding:     {smooth_count:>12} years
    
    {'RISK EVENTS':^40}
    {'─'*40}
    Market Crashes:     {down_year_count:>12} years
    Buffer Depleted:    {buffer_depleted_count:>12} times
    Buffer Refilled:    {refill_count:>12} times
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=0.8))
    
    plt.suptitle('Complete Portfolio Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

# Run backtests
print("="*80)
print("CPWR + FIREWALL STRATEGY BACKTEST")
print("Initial Portfolio: $10,000,000 | CPWR: 4% | Floor: $250,000 | Buffer: 5 years")
print("="*80)
print()

start_years = [2003]
all_results = {}

for start in start_years:
    print(f"\n{'─'*80}")
    print(f"SIMULATION STARTING {start}")
    print(f"{'─'*80}")
    
    results, success = simulate(start)
    
    if results is None:
        print(f"❌ {success}")
        continue
    
    # Check if results is empty (happens when no simulation years available)
    if not results:
        print(f"❌ No simulation years available after initialization year {start}")
        continue
    
    df = pd.DataFrame(results)
    all_results[start] = (df, success)
    
    # Print summary table
    print(f"\nStatus: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"Years Survived: {len(df)}")
    print(f"Final Portfolio: ${df['Total'].iloc[-1]:,.0f}")
    print(f"Buffer Depleted Events: {df['Buffer Depleted'].sum()}")
    print(f"Sold Equity for Spending: {df['Sold Equity'].sum()}")
    print(f"Refill Events: {df['Refill Event'].sum()}")
    print(f"Down Years: {df['Down Year'].sum()}")
    
    # Save detailed results to text file
    with open(f'results/results_{start}_{'✓' if success else '✗'}.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"CPWR + FIREWALL STRATEGY - DETAILED RESULTS\n")
        f.write(f"Starting Year: {start}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}\n")
        f.write(f"Years Survived: {len(df)}\n")
        f.write(f"Final Portfolio: ${df['Total'].iloc[-1]:,.0f}\n")
        f.write(f"Buffer Depleted Events: {df['Buffer Depleted'].sum()}\n")
        f.write(f"Sold Equity for Spending: {df['Sold Equity'].sum()}\n")
        f.write(f"Refill Events: {df['Refill Event'].sum()}\n")
        f.write(f"Down Years: {df['Down Year'].sum()}\n\n")
        
        f.write("─"*80 + "\n")
        f.write("YEAR-BY-YEAR RESULTS\n")
        f.write("─"*80 + "\n\n")
        
        display_df = df.copy()
        display_df[['Equity', 'Cash', 'Total', 'Proposed', 'Spending']] = \
            display_df[['Equity', 'Cash', 'Total', 'Proposed', 'Spending']].round(0)
        f.write(display_df.to_markdown(index=False, numalign="right", stralign="right", floatfmt=".0f"))
        f.write("\n")
    
    # Create visualization
    fig = visualize_simulation(df, start, success)
    plt.savefig(f'results/simulation_{start}_{'✓' if success else '✗'}.png', dpi=150, bbox_inches='tight')
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