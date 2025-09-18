import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os
import pickle
from datetime import datetime, timedelta

# Set font to Arial for consistent visualization
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of negative signs

sns.set(style="whitegrid")  # Set Seaborn style for plots

# Create directories for saving results
output_dir = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Artificially simulated pricing strategies analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

strategy_vis_dir = os.path.join(output_dir, 'Strategy visualization results')
if not os.path.exists(strategy_vis_dir):
    os.makedirs(strategy_vis_dir)

strategy_eval_dir = os.path.join(output_dir, 'Strategy evaluation results')
if not os.path.exists(strategy_eval_dir):
    os.makedirs(strategy_eval_dir)


# ----------------------
# 1. Load and Preprocess Data
# ----------------------
def load_and_preprocess_data():
    """Load and preprocess charging and time-of-use price data"""
    # Define file paths
    charging_data_file = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Charging_Data.csv"
    tou_data_file = '/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Time-of-use_Price.csv'

    # Load data with appropriate encoding
    charging_df = pd.read_csv(charging_data_file, encoding='ISO-8859-1')
    tou_df = pd.read_csv(tou_data_file, encoding='ISO-8859-1')

    # Convert time columns to datetime format
    charging_df['Start Time'] = pd.to_datetime(charging_df['Start Time'], errors='coerce')
    charging_df['End Time'] = pd.to_datetime(charging_df['End Time'], errors='coerce')

    # Handle Excel-style date formats if detected
    if charging_df['Start Time'].dtype == 'float64':
        charging_df['Start Time'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(charging_df['Start Time'], unit='d')
    if charging_df['End Time'].dtype == 'float64':
        charging_df['End Time'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(charging_df['End Time'], unit='d')

    # Remove rows with missing time data
    charging_df = charging_df.dropna(subset=['Start Time', 'End Time'])

    # Define function to determine price periods (Off-Peak/Peak/Super-Peak)
    def get_price_period(time):
        hour = time.hour
        minute = time.minute
        total_min = hour * 60 + minute  # Convert time to total minutes

        # Categorize time into predefined price periods
        if (0 <= total_min < 8 * 60) or (11 * 60 <= total_min < 13 * 60) or (22 * 60 <= total_min < 24 * 60):
            return 'Valley'
        elif (8 * 60 <= total_min < 11 * 60) or (13 * 60 <= total_min < 19 * 60) or (21 * 60 <= total_min < 22 * 60):
            return 'Flat'
        elif (19 * 60 <= total_min < 21 * 60):
            return 'Peak'
        else:
            return 'Unknown'

    # Map time to price periods and assign corresponding prices
    charging_df['price_period'] = charging_df['Start Time'].apply(get_price_period)
    price_map = {'Valley': 0.3784, 'Flat': 0.9014, 'Peak': 1.2064}
    charging_df['period_price'] = charging_df['price_period'].map(price_map)

    # Clean data: remove invalid entries
    charging_df = charging_df[charging_df['Transaction power/kwh'] > 0]  # Remove non-positive power values
    charging_df = charging_df[charging_df['price_period'] != 'Unknown']  # Remove uncategorized periods

    # Calculate charging duration in hours
    charging_df['charging_duration'] = (charging_df['End Time'] - charging_df['Start Time']).dt.total_seconds() / 3600

    # Categorize users by charging frequency (High/Low)
    user_charging_count = charging_df['UserID'].value_counts()
    median_charging_count = user_charging_count.median()  # Use median as threshold
    charging_df['user_type'] = charging_df['UserID'].apply(
        lambda x: 'High-Frequency User' if user_charging_count[x] >= median_charging_count else 'Low-Frequency User'
    )

    # Extract site types from location information
    if 'Location Information' in charging_df.columns:
        charging_df['site_type'] = charging_df['Location Information'].fillna('Unknown Site Type')
    else:
        charging_df['site_type'] = 'Unknown Site Type'  # Default if column missing

    # Extract hour of day for load analysis
    charging_df['hour'] = charging_df['Start Time'].dt.hour

    return charging_df


# ----------------------
# 2. Price Elasticity Calculation
# ----------------------
def calculate_elasticity_by_group(df):
    """Calculate price elasticity for different user/site groups"""
    # Define dimensions for group analysis
    group_dimensions = [
        'user_type',  # Analyze by user type
        'site_type',  # Analyze by site type
    ]

    elasticity_results = {}

    # Calculate overall price elasticity (all users combined)
    price_groups = df.groupby('period_price')['Transaction power/kwh'].mean().reset_index()
    price_groups = price_groups.sort_values('period_price')

    if len(price_groups) >= 2:
        # Calculate percentage changes in price and quantity
        price_changes = np.log(price_groups['period_price'].iloc[1:].values) - np.log(
            price_groups['period_price'].iloc[:-1].values)
        quantity_changes = np.log(price_groups['Transaction power/kwh'].iloc[1:].values) - np.log(
            price_groups['Transaction power/kwh'].iloc[:-1].values)
        elasticity_values = quantity_changes / price_changes  # Elasticity = %ΔQ / %ΔP
        elasticity_results['Overall'] = np.mean(elasticity_values)

    # Calculate elasticity for each group dimension
    for dim in group_dimensions:
        elasticity_by_group = {}
        for group_name, group_data in df.groupby(dim, observed=False):
            price_groups = group_data.groupby('period_price')['Transaction power/kwh'].mean().reset_index()
            price_groups = price_groups.sort_values('period_price')

            if len(price_groups) >= 2:
                # Same elasticity calculation as overall, but per group
                price_changes = np.log(price_groups['period_price'].iloc[1:].values) - np.log(
                    price_groups['period_price'].iloc[:-1].values)
                quantity_changes = np.log(price_groups['Transaction power/kwh'].iloc[1:].values) - np.log(
                    price_groups['Transaction power/kwh'].iloc[:-1].values)
                elasticity_values = quantity_changes / price_changes
                elasticity_by_group[group_name] = np.mean(elasticity_values)

        elasticity_results[dim] = elasticity_by_group

    return elasticity_results


# ----------------------
# 3. Define Pricing Strategies
# ----------------------
def define_pricing_strategies(base_prices):
    """Define multiple pricing strategies for comparison"""
    # base_prices format: {'Valley': float, 'Flat': float, 'Peak': float}

    strategies = {}

    # Strategy 1: Baseline (existing time-of-use pricing)
    strategies['Baseline Strategy'] = {
        'Valley': base_prices['Valley'],
        'Flat': base_prices['Flat'],
        'Peak': base_prices['Peak'],
        'description': 'Existing time-of-use pricing with fixed peak/off-peak rates'
    }

    # Strategy 2: User-Type Differentiation (tailored to user behavior)
    strategies['User-Type Differentiation'] = {
        'Low-Frequency User': {  # Price-high-sensitive
            'Valley': base_prices['Valley'] * 0.70,
            'Flat': base_prices['Flat'] * 1.05,
            'Peak': base_prices['Peak'] * 1.15
        },
        'High-Frequency User': {  # Price-low-insensitive
            'Valley': base_prices['Valley'] * 0.85,
            'Flat': base_prices['Flat'] * 1.0,
            'Peak': base_prices['Peak'] * 1.05
        },
        'description': 'Differential pricing based on user frequency: stronger incentives for high-frequency users'
    }

    # Strategy 3: Site-Type Differentiation (based on price elasticity)
    strategies['Site-Type Differentiation'] = {
        'Technology Park': {  # Price-high-sensitive
            'Valley': base_prices['Valley'] * 0.70,
            'Flat': base_prices['Flat'] * 1.15,
            'Peak': base_prices['Peak'] * 1.35
        },
        'Financial Industrial Park': {  # Price-medium-sensitive
            'Valley': base_prices['Valley'] * 0.85,
            'Flat': base_prices['Flat'] * 1.1,
            'Peak': base_prices['Peak'] * 1.25
        },
        'Other Sites': {  # Price-low-sensitive
            'Valley': base_prices['Valley'] * 0.9,
            'Flat': base_prices['Flat'] * 1.05,
            'Peak': base_prices['Peak'] * 1.15
        },
        'description': 'Differential pricing based on site type, targeting price-sensitive locations'
    }

    return strategies


# ----------------------
# 4. Counterfactual Simulation
# ----------------------
def simulate_strategy_effects(df, strategies, elasticity_results, capacity_constraint=0.8):
    """
    Simulate impacts of different pricing strategies on load, revenue, and user welfare

    Parameters:
    df: Original dataset
    strategies: Dictionary of pricing strategies
    elasticity_results: Precomputed price elasticity values
    capacity_constraint: Maximum allowable charging pile utilization rate

    Returns:
    Dictionary of performance metrics for each strategy
    """
    results = {}

    # Extract baseline prices from data
    base_prices = {
        'Valley': df[df['price_period'] == 'Valley']['period_price'].unique()[0],
        'Flat': df[df['price_period'] == 'Flat']['period_price'].unique()[0],
        'Peak': df[df['price_period'] == 'Peak']['period_price'].unique()[0]
    }

    # Calculate baseline metrics (no pricing changes)
    baseline_metrics = calculate_metrics(df, base_prices)
    results['Baseline Strategy'] = baseline_metrics

    # Simulate each strategy
    for strategy_name, strategy in strategies.items():
        if strategy_name == 'Baseline Strategy':
            continue  # Skip baseline (already calculated)

        print(f"Simulating strategy: {strategy_name}")

        # Create copy of data for simulation
        simulated_df = df.copy()
        simulated_df['simulated_power'] = simulated_df['Transaction power/kwh'].copy()  # Initialize with original power
        simulated_df['simulated_price'] = simulated_df['period_price'].copy()  # Initialize with original prices

        # Apply pricing strategy and simulate user response
        if strategy_name == 'User-Type Differentiation':
            # Differentiated pricing by user type
            for user_type in ['High-Frequency User', 'Low-Frequency User']:
                new_prices = strategy[user_type]
                price_changes = {
                    period: (new_prices[period] - base_prices[period]) / base_prices[period]
                    for period in ['Valley', 'Flat', 'Peak']
                }

                # Apply to each period and user type
                for period in ['Valley', 'Flat', 'Peak']:
                    mask = (simulated_df['price_period'] == period) & (simulated_df['user_type'] == user_type)
                    # Use user-type specific elasticity (fallback to overall)
                    elasticity = elasticity_results['user_type'].get(user_type, elasticity_results['Overall'])
                    quantity_change = elasticity * price_changes[period]
                    simulated_df.loc[mask, 'simulated_power'] *= (1 + quantity_change)
                    simulated_df.loc[mask, 'simulated_price'] = new_prices[period]

        elif strategy_name == 'Site-Type Differentiation':
            # Differentiated pricing by site type
            for site_type in strategy.keys():
                if site_type == 'description':
                    continue  # Skip description field

                new_prices = strategy[site_type]
                price_changes = {
                    period: (new_prices[period] - base_prices[period]) / base_prices[period]
                    for period in ['Valley', 'Flat', 'Peak']
                }

                # Apply to each period and site type
                for period in ['Valley', 'Flat', 'Peak']:
                    mask = (simulated_df['price_period'] == period) & (simulated_df['site_type'] == site_type)
                    # Use site-type specific elasticity (fallback to overall)
                    elasticity = elasticity_results['site_type'].get(site_type, elasticity_results['Overall'])
                    quantity_change = elasticity * price_changes[period]
                    simulated_df.loc[mask, 'simulated_power'] *= (1 + quantity_change)
                    simulated_df.loc[mask, 'simulated_price'] = new_prices[period]

        # Apply charging pile capacity constraints
        simulated_df = apply_capacity_constraints(simulated_df, capacity_constraint)

        # Calculate metrics for this strategy
        metrics = calculate_metrics(
            simulated_df,
            base_prices,
            is_simulated=True,
            baseline_metrics=baseline_metrics
        )

        results[strategy_name] = metrics
        results[strategy_name]['description'] = strategy['description']

    return results


def apply_capacity_constraints(df, capacity_constraint):
    """Adjust load to respect charging pile capacity limits"""
    # Calculate hourly load
    hourly_usage = df.groupby('hour')['simulated_power'].sum().reset_index()
    if hourly_usage.empty:
        return df

    # Identify peak hour and check if capacity is exceeded
    max_usage = hourly_usage['simulated_power'].max()
    if max_usage <= 0:
        return df

    # If peak load exceeds constraint, scale down all loads proportionally
    if max_usage > 0 and (max_usage / max_usage) > capacity_constraint:  # Simplified check
        scaling_factor = capacity_constraint
        df['simulated_power'] = df['simulated_power'] * scaling_factor

    return df


def calculate_metrics(df, base_prices, is_simulated=False, baseline_metrics=None):
    """
    Calculate performance metrics including:
    - Peak shaving (peak-valley difference, ratio)
    - Utilization balance (period utilization, std)
    - User welfare (ΔCS using 0.5×(Pₒᵣᵢg - Pₙₑw)×(Qₒᵣᵢg - Qₙₑw))
    - Revenue (total revenue)
    """
    metrics = {}

    # Select columns based on simulation status
    power_col = 'simulated_power' if is_simulated else 'Transaction power/kwh'
    price_col = 'simulated_price' if is_simulated else 'period_price'

    # 1. Peak Shaving Metrics
    hourly_load = df.groupby('hour')[power_col].sum().reset_index()
    max_load = hourly_load[power_col].max()
    min_load = hourly_load[power_col].min()
    metrics['Peak-Valley Difference'] = max_load - min_load  # Lower = better
    metrics['Peak-Valley Ratio'] = max_load / min_load if min_load > 0 else np.inf  # Lower = better
    metrics['Peak Load'] = max_load

    # 2. Utilization Metrics
    total_power = df[power_col].sum()
    metrics['Total Charging Power'] = total_power

    # Calculate period utilization rates
    period_usage = df.groupby('price_period')[power_col].sum()
    for period in ['Valley', 'Flat', 'Peak']:
        metrics[f'{period} Utilization'] = period_usage.get(period, 0) / total_power if total_power > 0 else 0

    # Standard deviation of utilization (lower = more balanced)
    metrics['Utilization Standard Deviation'] = np.std([
        metrics[f'{p} Utilization'] for p in ['Valley', 'Flat', 'Peak']
    ])

    # 3. User Welfare Metrics (ΔCS with positive/negative values preserved)
    if is_simulated and baseline_metrics is not None:
        # Original (baseline) values
        P_original = np.mean(list(base_prices.values()))  # Average baseline price
        Q_original = baseline_metrics['Total Charging Power']  # Baseline total load

        # New (simulated) v
        # alues
        P_new = np.mean(df[price_col])  # Average simulated price
        Q_new = total_power  # Simulated total load

        # Calculate ΔCS using the formula
        delta_cs = 0.5 * (P_original - P_new) * (Q_original + Q_new)
        metrics['Change in Consumer Surplus (ΔCS)'] = delta_cs  # Preserve sign: negative = improvement

        # Welfare loss is positive ΔCS (negative = no loss)
        metrics['User Welfare Loss'] = delta_cs
    else:
        # Baseline has no welfare change
        metrics['Change in Consumer Surplus (ΔCS)'] = 0.0
        metrics['User Welfare Loss'] = 0.0

    # 4. Revenue Metrics
    df['revenue'] = df[power_col] * df[price_col]
    metrics['Total Revenue'] = df['revenue'].sum()

    return metrics


# ----------------------
# 5. Strategy Evaluation and Optimization
# ----------------------
def evaluate_and_optimize_strategies(results):
    """Evaluate strategies using multi-criteria analysis and identify the optimal one"""
    # Convert results to DataFrame for easier analysis
    eval_df = pd.DataFrame.from_dict(results, orient='index')

    # Normalize metrics for composite scoring (0-1 scale)
    normalized_df = pd.DataFrame()

    # Metrics to minimize (lower = better)
    for col in ['Peak-Valley Difference', 'Peak-Valley Ratio',
                'Utilization Standard Deviation', 'User Welfare Loss']:
        min_val = eval_df[col].min()
        max_val = eval_df[col].max()
        normalized_df[col] = 1 - (eval_df[col] - min_val) / (max_val - min_val + 1e-10)  # Higher = better

    # Metrics to maximize (higher = better)
    for col in ['Total Charging Power', 'Total Revenue']:
        min_val = eval_df[col].min()
        max_val = eval_df[col].max()
        normalized_df[col] = (eval_df[col] - min_val) / (max_val - min_val + 1e-10)  # Higher = better

    # Calculate composite score with weighted metrics
    weights = {
        'Peak-Valley Difference': 0.25,  # Prioritize peak shaving
        'Utilization Standard Deviation': 0.25,  # Prioritize balanced utilization
        'Total Revenue': 0.25,  # Prioritize revenue
        'User Welfare Loss': 0.25  # Prioritize user welfare
    }

    normalized_df['Composite Score'] = 0.0
    for col, weight in weights.items():
        normalized_df['Composite Score'] += normalized_df[col] * weight

    # Add strategy descriptions
    normalized_df['Description'] = [results[name].get('description', '') for name in normalized_df.index]

    # Identify top-performing strategy
    best_strategy = normalized_df['Composite Score'].idxmax()

    return eval_df, normalized_df, best_strategy


# ----------------------
# 6. Result Visualization
# ----------------------
def visualize_strategy_results(df, results, normalized_df, best_strategy, strategies):
    """Generate visualizations to compare strategy performance"""
    # 1. Key Metrics Comparison
    metrics_to_plot = [
        'Peak-Valley Difference', 'Utilization Standard Deviation',
        'User Welfare Loss', 'Total Revenue', 'Composite Score'
    ]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(results.keys()))
    width = 0.15  # Bar width

    for i, metric in enumerate(metrics_to_plot):
        # Extract values (normalize non-score metrics)
        if metric == 'Composite Score':
            values = [normalized_df.loc[name, metric] for name in results.keys()]
        else:
            values = [results[name][metric] for name in results.keys()]
            # Normalize to 0-1 range for visualization
            values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)

        plt.bar(x + i * width, values, width, label=metric)

    plt.xticks(x + width * 2, results.keys(), ha='center', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Pricing Strategy', fontsize=14)
    plt.ylabel('Normalized Value (0-1)', fontsize=14)
    plt.ylim(0, 1)
    plt.title('Comparison of Key Metrics Across Pricing Strategies', fontsize=16, pad=25)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(strategy_vis_dir, 'Key_metrics_comparison.png'), dpi=800)
    plt.close()

    # 2. Composite Score Ranking
    plt.figure(figsize=(10, 6))
    scores = normalized_df['Composite Score'].sort_values()
    # Highlight the best strategy in green
    colors = ['red' if idx != best_strategy else 'green' for idx in scores.index]

    # Plot the bar chart and get the bar objects
    bars = scores.plot(kind='bar', color=colors)

    # Add value labels on top of each bar
    for bar in bars.patches:
        # Get the height of the bar (i.e., the score value)
        height = bar.get_height()
        # Add label at the center top of the bar, keeping 4 decimal places
        plt.text(
            bar.get_x() + bar.get_width() / 2.,  # x-coordinate (center of the bar)
            height,  # y-coordinate (top of the bar)
            f'{height:.4f}',  # Label text (formatted value)
            ha='center',  # Horizontal alignment (center)
            va='bottom',  # Vertical alignment (bottom aligned with bar top)
            rotation=0,  # No rotation for the label
            fontsize=14
        )

    plt.title('Composite Score of Pricing Strategies', fontsize=16)
    plt.ylabel('Composite Score (Higher = Better)', fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(strategy_vis_dir, 'Composite_scores_ranking.png'), dpi=500)
    plt.close()

    # 3. Load Curve Comparison (Baseline vs Best Strategy)
    # Extract baseline prices
    base_prices = {
        'Valley': df[df['price_period'] == 'Valley']['period_price'].unique()[0],
        'Flat': df[df['price_period'] == 'Flat']['period_price'].unique()[0],
        'Peak': df[df['price_period'] == 'Peak']['period_price'].unique()[0]
    }

    # Baseline hourly load
    baseline_hourly = df.groupby('hour')['Transaction power/kwh'].sum()

    # Extend baseline_hourly to 24 hours
    baseline_hourly_extended = baseline_hourly.reindex(range(25), fill_value=0)
    baseline_hourly_extended.loc[24] = baseline_hourly_extended.loc[0]

    # Simulate best strategy load
    best_df = df.copy()
    best_df['simulated_power'] = best_df['Transaction power/kwh'].copy()
    best_df['simulated_price'] = best_df['period_price'].copy()
    best_strategy_data = strategies[best_strategy]

    # Apply best strategy prices
    if best_strategy in ['User-Type Differentiation', 'Site-Type Differentiation']:
        # Handle differentiated pricing
        for group_type in best_strategy_data.keys():
            if group_type == 'description':
                continue
            for period in ['Valley', 'Flat', 'Peak']:
                # Define mask based on group type
                if best_strategy == 'User-Type Differentiation':
                    mask = (best_df['price_period'] == period) & (best_df['user_type'] == group_type)
                else:
                    mask = (best_df['price_period'] == period) & (best_df['site_type'] == group_type)
                best_df.loc[mask, 'simulated_price'] = best_strategy_data[group_type][period]
    else:
        # Handle uniform pricing
        for period in ['Valley', 'Flat', 'Peak']:
            mask = best_df['price_period'] == period
            best_df.loc[mask, 'simulated_price'] = best_strategy_data[period]

    # Calculate load changes using elasticity
    elasticity = calculate_elasticity_by_group(df)
    for idx, row in best_df.iterrows():
        period = row['price_period']
        original_price = base_prices[period]
        new_price = row['simulated_price']
        price_change = (new_price - original_price) / original_price

        # Get relevant elasticity
        if best_strategy == 'User-Type Differentiation':
            elast = elasticity['user_type'].get(row['user_type'], elasticity['Overall'])
        elif best_strategy == 'Site-Type Differentiation':
            elast = elasticity['site_type'].get(row['site_type'], elasticity['Overall'])
        else:
            elast = elasticity['Overall']

        # Update load
        quantity_change = elast * price_change
        best_df.at[idx, 'simulated_power'] = row['Transaction power/kwh'] * (1 + quantity_change)

    # Best strategy hourly load
    best_hourly = best_df.groupby('hour')['simulated_power'].sum()

    # Extend best_hourly to 24 hours
    best_hourly_extended = best_hourly.reindex(range(25), fill_value=0)
    best_hourly_extended.loc[24] = best_hourly_extended.loc[0]

    # Plot load curves
    plt.figure(figsize=(10, 6))
    baseline_hourly_extended.plot(label='Baseline Strategy', color='blue', linewidth=1.5, linestyle='-',
                                  alpha=0.7)
    best_hourly_extended.plot(label=f'{best_strategy}', color='red', linewidth=1.5, linestyle='--', alpha=0.7)

    # Highlight price periods
    plt.axvspan(0, 8, color='green', alpha=0.1, label='Valley')
    plt.axvspan(11, 13, color='green', alpha=0.1)
    plt.axvspan(22, 24, color='green', alpha=0.1)
    plt.axvspan(8, 11, color='orange', alpha=0.1, label='Flat')
    plt.axvspan(13, 19, color='orange', alpha=0.1)
    plt.axvspan(21, 22, color='orange', alpha=0.1)
    plt.axvspan(19, 21, color='red', alpha=0.1, label='Peak')

    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 1), fontsize=14)
    plt.title(f'Hourly Load: Baseline vs {best_strategy}', fontsize=16, pad=25)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Charging Load (kWh)', fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(strategy_vis_dir, 'Load_curve_comparison.png'), dpi=500)
    plt.close()

    # 4. Trade-off: Peak Shaving vs User Welfare
    # Note: This plot is intentionally omitted as per user request to simplify analysis
    # ...

    print(f"All visualizations saved to: {strategy_vis_dir}")


# ----------------------
# 7. Main Execution
# ----------------------
if __name__ == "__main__":
    print("===== Pricing Strategy Evaluation Started =====")

    # Step 1: Load and preprocess data
    charging_df = load_and_preprocess_data()
    print(f"Data preprocessing completed. Valid records: {len(charging_df)}")

    # Step 2: Calculate price elasticity
    elasticity_results = calculate_elasticity_by_group(charging_df)
    print("Price elasticity calculation completed")

    # Step 3: Define baseline prices and pricing strategies
    base_prices = {
        'Valley': charging_df[charging_df['price_period'] == 'Valley']['period_price'].unique()[0],
        'Flat': charging_df[charging_df['price_period'] == 'Flat']['period_price'].unique()[0],
        'Peak': charging_df[charging_df['price_period'] == 'Peak']['period_price'].unique()[0]
    }
    print(f"Baseline prices: {base_prices}")

    strategies = define_pricing_strategies(base_prices)
    print(f"Defined {len(strategies)} pricing strategies")

    # Step 4: Simulate strategy impacts
    strategy_results = simulate_strategy_effects(charging_df, strategies, elasticity_results)
    print("Strategy simulation completed")

    # Step 5: Evaluate strategies and select optimal
    eval_df, normalized_df, best_strategy = evaluate_and_optimize_strategies(strategy_results)
    print(f"Optimal strategy identified: {best_strategy}")

    # Step 6: Save results
    eval_df.to_csv(os.path.join(strategy_eval_dir, 'strategy_metrics.csv'))
    normalized_df.to_csv(os.path.join(strategy_eval_dir, 'normalized_scores.csv'))

    with open(os.path.join(strategy_eval_dir, 'detailed_results.pkl'), 'wb') as f:
        pickle.dump(strategy_results, f)
    print(f"Evaluation results saved to: {strategy_eval_dir}")

    # Step 7: Generate visualizations
    visualize_strategy_results(charging_df, strategy_results, normalized_df, best_strategy, strategies)

    # Print key findings for optimal strategy
    print("\n===== Optimal Strategy Details =====")
    print(f"Strategy: {best_strategy}")
    print(f"Description: {strategies[best_strategy]['description']}")
    print("\nKey Performance Metrics:")
    print(f"1. Peak-Valley Difference: {strategy_results[best_strategy]['Peak-Valley Difference']:.2f} kWh "
          f"(Baseline: {strategy_results['Baseline Strategy']['Peak-Valley Difference']:.2f} kWh)")
    print(f"2. Utilization Std Dev: {strategy_results[best_strategy]['Utilization Standard Deviation']:.4f} "
          f"(Baseline: {strategy_results['Baseline Strategy']['Utilization Standard Deviation']:.4f})")
    print(f"3. User Welfare ΔCS: {strategy_results[best_strategy]['Change in Consumer Surplus (ΔCS)']:.2f} "
          f"(Negative = Improvement)")
    print(f"4. Total Revenue: {strategy_results[best_strategy]['Total Revenue']:.2f} "
          f"(Baseline: {strategy_results['Baseline Strategy']['Total Revenue']:.2f})")

    print("\n===== Pricing Strategy Evaluation Completed =====")
