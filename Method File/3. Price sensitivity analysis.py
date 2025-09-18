import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pickle
from datetime import datetime, timedelta

# Configure font settings to ensure proper display of negative signs and characters
plt.rcParams['axes.unicode_minus'] = False  # Ensure correct display of negative signs
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial font
plt.rcParams['font.family'] = 'Arial'  # Set font family

# Create directory for saving results (create if not exists)
output_dir = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Price sensitivity analysis"
os.makedirs(output_dir, exist_ok=True)  # Add directory creation to avoid "file not found" errors


# ----------------------
# 1. Data Loading and Preprocessing
# ----------------------
def load_and_preprocess_data():
    """Load and preprocess charging data and time-of-use (TOU) price data"""
    # Load data (replace with actual file paths if needed)
    charging_data_file = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Charging_Data.csv"
    tou_data_file = '/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Time-of-use_Price.csv'

    # Try multiple encodings to handle compatibility issues
    try:
        charging_df = pd.read_csv(charging_data_file, encoding='ISO-8859-1')
    except:
        charging_df = pd.read_csv(charging_data_file, encoding='utf-8')

    try:
        tou_df = pd.read_csv(tou_data_file, encoding='ISO-8859-1')
    except:
        tou_df = pd.read_csv(tou_data_file, encoding='utf-8')

    # Process datetime columns and remove invalid records
    charging_df['Start Time'] = pd.to_datetime(charging_df['Start Time'], errors='coerce')
    charging_df['End Time'] = pd.to_datetime(charging_df['End Time'], errors='coerce')
    charging_df = charging_df.dropna(subset=['Start Time', 'End Time'])  # Drop rows with missing datetime

    # Define function to classify TOU price periods based on start time
    def get_price_period(time):
        hour = time.hour
        minute = time.minute
        total_min = hour * 60 + minute  # Convert time to total minutes in a day

        # Classify periods based on TOU price rules
        if (0 <= total_min < 8 * 60) or (11 * 60 <= total_min < 13 * 60) or (22 * 60 <= total_min < 24 * 60):
            return 'Valley'
        elif (8 * 60 <= total_min < 11 * 60) or (13 * 60 <= total_min < 19 * 60) or (21 * 60 <= total_min < 22 * 60):
            return 'Flat'
        elif (19 * 60 <= total_min < 21 * 60):
            return 'Peak'
        else:
            return 'Unknown'

    # Map charging records to price periods and assign corresponding prices
    charging_df['price_period'] = charging_df['Start Time'].apply(get_price_period)
    price_map = {'Valley': 0.3784, 'Flat': 0.9014, 'Peak': 1.2064}  # CNY/kWh
    charging_df['period_price'] = charging_df['price_period'].map(price_map)

    # Data cleaning: Remove abnormal records
    charging_df = charging_df[charging_df['Transaction power/kwh'] > 0]  # Drop zero/negative charging volume
    charging_df = charging_df[charging_df['price_period'] != 'Unknown']  # Drop records with unknown price periods

    # Calculate charging duration (in hours)
    charging_df['charging_duration'] = (charging_df['End Time'] - charging_df['Start Time']).dt.total_seconds() / 3600

    # Classify users into high/low frequency based on charging count (median split)
    user_charging_count = charging_df['UserID'].value_counts()  # Count charging times per user
    median_charging_count = user_charging_count.median()  # Use median as split threshold
    charging_df['user_type'] = charging_df['UserID'].apply(
        lambda x: 'High-Frequency User' if user_charging_count[x] >= median_charging_count else 'Low-Frequency User'
    )

    # Extract site type from location information (fill missing values with 'Unknown')
    if 'Location Information' in charging_df.columns:
        charging_df['site_type'] = charging_df['Location Information'].fillna('Unknown')
    else:
        charging_df['site_type'] = 'Unknown'

    # Extract hour of charging start and process temperature data
    charging_df['hour'] = charging_df['Start Time'].dt.hour  # Hour of day (0-23) for charging start
    if 'Temperature' not in charging_df.columns:
        np.random.seed(42)  # Set random seed for reproducibility of mock data
        charging_df['Temperature'] = np.random.normal(25, 5, len(charging_df))  # Mock temp: mean=25°C, std=5°C
    else:
        # Handle temperature outliers using 3σ rule
        temp_mean = charging_df['Temperature'].mean()
        temp_std = charging_df['Temperature'].std()
        charging_df['Temperature'] = charging_df['Temperature'].clip(
            lower=temp_mean - 3 * temp_std,
            upper=temp_mean + 3 * temp_std
        )

    return charging_df


# ----------------------
# 2. Price Elasticity Calculation with Log-Log Regression R-squared
# ----------------------
def calculate_price_elasticity(df, groupby_col=None):
    """
    Calculate price elasticity of charging demand using the midpoint method,
    including R-squared value for log-log regression model.

    Parameters:
    df: DataFrame containing 'period_price' (price) and 'Transaction power/kwh' (demand).
    groupby_col: Column name for grouped analysis (e.g., 'user_type'). If None, calculate overall elasticity.

    Returns:
    DataFrame with elasticity values, p-values, R-squared values, and sample sizes.
    """
    # Calculate overall elasticity (no grouping)
    if groupby_col is None:
        # Aggregate average charging volume by price level
        price_groups = df.groupby('period_price')['Transaction power/kwh'].agg(['mean', 'count']).reset_index()
        price_groups.columns = ['price', 'quantity', 'count']  # Rename columns for clarity

        # Return NaN if fewer than 2 price levels (insufficient variation for elasticity)
        if len(price_groups) < 2:
            return pd.DataFrame({
                'group': ['Overall'],
                'elasticity': [np.nan],
                'p_value': [np.nan],
                'r_squared': [np.nan],
                'sample_size': [len(df)]
            })

        # Calculate elasticity using midpoint method (avoids bias from price direction)
        price_groups = price_groups.sort_values('price')  # Sort prices in ascending order
        elasticity_values = []
        for i in range(1, len(price_groups)):
            # Extract consecutive price and quantity pairs
            P1, P2 = price_groups['price'].iloc[i - 1], price_groups['price'].iloc[i]
            Q1, Q2 = price_groups['quantity'].iloc[i - 1], price_groups['quantity'].iloc[i]

            # Midpoint formula: [(ΔQ / Q_avg) / (ΔP / P_avg)]
            percent_change_q = (Q2 - Q1) / ((Q1 + Q2) / 2)
            percent_change_p = (P2 - P1) / ((P1 + P2) / 2)
            elasticity = percent_change_q / percent_change_p
            elasticity_values.append(elasticity)

        avg_elasticity = np.mean(elasticity_values)  # Average elasticity across price pairs

        # Log-log regression (ln(Q) ~ ln(P)) with R-squared calculation
        log_price = np.log(price_groups['price'])
        log_quantity = np.log(price_groups['quantity'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_quantity)
        r_squared = r_value ** 2  # Calculate R-squared

        return pd.DataFrame({
            'group': ['Overall'],
            'elasticity': [avg_elasticity],
            'p_value': [p_value],
            'r_squared': [r_squared],
            'sample_size': [len(df)]
        })

    # Calculate elasticity for grouped data (e.g., by user_type or site_type)
    else:
        results = []
        for group_name, group_data in df.groupby(groupby_col):
            # Aggregate average charging volume by price within the group
            price_groups = group_data.groupby('period_price')['Transaction power/kwh'].agg(
                ['mean', 'count']
            ).reset_index()
            price_groups.columns = ['price', 'quantity', 'count']

            # Skip groups with insufficient price variation
            if len(price_groups) < 2:
                results.append({
                    'group': group_name,
                    'elasticity': np.nan,
                    'p_value': np.nan,
                    'r_squared': np.nan,
                    'sample_size': len(group_data)
                })
                continue

            # Calculate elasticity using midpoint method
            price_groups = price_groups.sort_values('price')
            elasticity_values = []
            for i in range(1, len(price_groups)):
                P1, P2 = price_groups['price'].iloc[i - 1], price_groups['price'].iloc[i]
                Q1, Q2 = price_groups['quantity'].iloc[i - 1], price_groups['quantity'].iloc[i]

                percent_change_q = (Q2 - Q1) / ((Q1 + Q2) / 2)
                percent_change_p = (P2 - P1) / ((P1 + P2) / 2)
                elasticity = percent_change_q / percent_change_p
                elasticity_values.append(elasticity)

            avg_elasticity = np.mean(elasticity_values)

            # Log-log regression with R-squared calculation for the group
            log_price = np.log(price_groups['price'])
            log_quantity = np.log(price_groups['quantity'])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_quantity)
            r_squared = r_value ** 2  # Calculate R-squared

            results.append({
                'group': group_name,
                'elasticity': avg_elasticity,
                'p_value': p_value,
                'r_squared': r_squared,
                'sample_size': len(group_data)
            })

        return pd.DataFrame(results)


# ----------------------
# 3. K-S Test for Charging Hour Distribution
# ----------------------
def ks_test_charging_hour(charging_df):
    """
    Perform two-sample Kolmogorov-Smirnov (K-S) Test to verify if high-frequency and low-frequency users
    have significantly different distributions of charging start hours (0-23).
    """
    # Extract charging hours for high/low frequency users
    high_freq_hours = charging_df[charging_df['user_type'] == 'High-Frequency User']['hour'].values
    low_freq_hours = charging_df[charging_df['user_type'] == 'Low-Frequency User']['hour'].values

    # Perform two-sample K-S test
    ks_stat, p_value = stats.ks_2samp(
        data1=high_freq_hours,
        data2=low_freq_hours,
        alternative='two-sided'
    )

    # Calculate descriptive statistics
    group_stats = charging_df.groupby('user_type')['hour'].agg([
        'count', 'mean', 'std', 'median'
    ]).round(2)

    # Define significance level and conclusion
    alpha = 0.05
    ks_results = {
        'K-S Statistic (D)': round(ks_stat, 4),
        'p-value': round(p_value, 4),
        'Significance Level (α)': alpha,
        'Significant Difference': 'Yes' if p_value < alpha else 'No',
        'Conclusion': (
            "Reject null hypothesis: High-frequency and low-frequency users have significantly different "
            "charging hour distributions." if p_value < alpha else
            "Fail to reject null hypothesis: No significant difference in charging hour distributions "
            "between high-frequency and low-frequency users."
        ),
        'Group Statistics': group_stats
    }

    # Prepare grouped data for visualization
    grouped_hours = {
        'High-Frequency User': high_freq_hours,
        'Low-Frequency User': low_freq_hours
    }

    return ks_results, grouped_hours


# ----------------------
# 4. Visualize K-S Test Results
# ----------------------
def plot_ks_results(grouped_hours, output_dir):
    """Visualize charging hour distributions for high/low frequency users"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color scheme
    colors = {'High-Frequency User': '#006400', 'Low-Frequency User': '#90EE90'}
    labels = {'High-Frequency User': 'High-Frequency\n(≥ Median Charges)',
              'Low-Frequency User': 'Low-Frequency\n(< Median Charges)'}

    # Subplot 1: Histogram of charging hours
    for user_type, hours in grouped_hours.items():
        # Duplicate 0-hour data for 24-hour mark to represent a full cycle
        hours_with_24 = np.append(hours, hours[hours == 0] + 24)
        ax1.hist(
            x=hours_with_24,
            bins=np.arange(26),
            alpha=0.6,
            color=colors[user_type],
            label=labels[user_type],
            density=True
        )
    ax1.set_xlabel('Charging Start Hour (0-24)', fontsize=12)
    ax1.set_ylabel('Normalized Frequency', fontsize=12)
    ax1.set_title('Distribution of Charging Start Hours', fontsize=14, pad=5)
    ax1.set_xticks(range(0, 25, 1))
    ax1.set_xlim(0, 24)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Subplot 2: CDF Curve
    for user_type, hours in grouped_hours.items():
        sorted_hours = np.sort(hours)
        cdf = np.arange(1, len(sorted_hours) + 1) / len(sorted_hours)

        # Add a point at x=24 with a CDF value of 1.0 to close the cycle
        extended_hours = np.append(sorted_hours, 24)
        extended_cdf = np.append(cdf, 1.0)

        ax2.plot(
            extended_hours,
            extended_cdf,
            color=colors[user_type],
            linewidth=2,
            label=labels[user_type]
        )
    ax2.set_xlabel('Charging Start Hour (0-24)', fontsize=12)
    ax2.set_ylabel('Cumulative Distribution Function (CDF)', fontsize=12)
    ax2.set_title('CDF of Charging Start Hours', fontsize=14, pad=5)
    ax2.set_xticks(range(0, 25, 1))
    ax2.set_xlim(0, 24)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'Charging_Hour_Distribution_KS_Test.png'),
        dpi=500,
        bbox_inches='tight'
    )
    plt.close()
    print(f"K-S test visualization saved to: {os.path.join(output_dir, 'Charging_Hour_Distribution_KS_Test.png')}")


# ----------------------
# 5. Visualization Functions
# ----------------------
def visualize_additional_plots(charging_df, output_dir):
    """Generate additional visualizations"""
    # Green-themed colormaps
    green_cmap = plt.cm.YlGn
    green_cmap2 = plt.cm.Greens

    # Plot 1: Box plot of charging volume across price periods
    plt.figure(figsize=(10, 6))
    period_order = ['Valley', 'Flat', 'Peak']
    sns.boxplot(
        x='price_period',
        y='Transaction power/kwh',
        data=charging_df,
        order=period_order,
        palette=[green_cmap2(0.3), green_cmap2(0.6), green_cmap2(0.9)],
        hue='price_period',
        legend=False
    )
    # Add jittered points
    sns.stripplot(
        x='price_period',
        y='Transaction power/kwh',
        data=charging_df,
        order=period_order,
        color='#1B5E20',
        alpha=0.2,
        size=3
    )
    plt.title('Distribution of Charging Volume Across Price Periods', fontsize=16)
    plt.xlabel('Price Period', fontsize=14)
    plt.ylabel('Charging Volume (kWh)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Charging_Quantity_by_Period.png'), dpi=500)
    plt.close()

    # Plot 2: Hourly trend of charging volume and electricity price
    plt.figure(figsize=(12, 6))
    # Aggregate hourly averages
    hourly_data = charging_df.groupby('hour').agg({
        'Transaction power/kwh': 'mean',
        'period_price': 'mean'
    }).reset_index()

    # Extend hourly_data to include a 24-hour mark (as a duplicate of 0-hour)
    hourly_data.loc[len(hourly_data)] = [24, hourly_data.loc[0, 'Transaction power/kwh'],
                                         hourly_data.loc[0, 'period_price']]

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot charging volume
    charging_line, = ax1.plot(
        hourly_data['hour'],
        hourly_data['Transaction power/kwh'],
        color=green_cmap(0.5),
        marker='o',
        linewidth=2,
        label='Avg. Charging Volume'
    )
    # Add filled area
    min_charge = hourly_data['Transaction power/kwh'].min()
    ax1.fill_between(
        hourly_data['hour'],
        hourly_data['Transaction power/kwh'],
        y2=min_charge,
        color=green_cmap(0.5),
        alpha=0.3
    )

    # Plot electricity price
    price_line, = ax2.plot(
        hourly_data['hour'],
        hourly_data['period_price'],
        color=green_cmap(0.8),
        marker='s',
        linewidth=2,
        linestyle='--',
        label='Avg. Electricity Price'
    )

    # Axis settings
    ax1.set_xlabel('Hour', fontsize=14, color='black')
    ax1.set_ylabel('Avg. Charging Volume (kWh)', fontsize=14, color='black')
    ax2.set_ylabel('Avg. Electricity Price (CNY/kWh)', fontsize=14, color='black')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25))
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)

    # Combine legends
    lines = [charging_line, price_line]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2,
               fontsize=14)

    plt.title('Hourly Trend of Charging Volume and Electricity Price', fontsize=16, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Hourly_Trend_Charging_Price.png'), dpi=500, bbox_inches='tight')
    plt.close()


def visualize_results(elasticity_results, original_data):
    """Visualize price elasticity results"""
    group_dimensions = ['user_type', 'site_type']

    for dim in group_dimensions:
        plt.figure(figsize=(12, 7))
        results_df = elasticity_results[dim].dropna()
        results_df = results_df.sort_values('elasticity')

        # Green gradient colors for bars
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(results_df)))
        bars = plt.bar(results_df['group'], results_df['elasticity'], color=colors, alpha=0.7)

        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Add elasticity values as labels
        for bar, elasticity in zip(bars, results_df['elasticity']):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{elasticity:.3f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=14,
            )

        plt.title(f'Price Elasticity by {dim.replace("_", " ").title()}',
                  fontsize=16, pad=25)

        plt.ylabel('Price Elasticity Value', fontsize=14)
        plt.yticks(fontsize=14)

        if dim == 'user_type':
            plt.xticks(rotation=0, ha='center', fontsize=14)
        else:
            plt.xticks(rotation=45, ha='right', fontsize=14)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Price_Elasticity_by_{dim}.png'), dpi=500)
        plt.close()

    # Call additional plots
    visualize_additional_plots(original_data, output_dir)
    print("All visualization results saved to the output folder.")


# ----------------------
# 6. Main Workflow
# ----------------------
def analyze_elasticity_and_ks_test():
    """Main function to run the full workflow"""
    # Step 1: Load and preprocess data
    charging_df = load_and_preprocess_data()
    print(f"Data preprocessing completed. Valid charging records: {len(charging_df)}")

    # Step 2: Calculate user charging frequency statistics
    user_charging_count = charging_df['UserID'].value_counts()
    total_users = len(user_charging_count)
    median_freq = user_charging_count.median()
    high_freq_user_count = sum(user_charging_count >= median_freq)
    low_freq_user_count = total_users - high_freq_user_count

    print("\n=== User Charging Frequency Statistics ===")
    print(f"Total unique users: {total_users}")
    print(f"Median charging frequency (split threshold): {median_freq:.1f}")
    print(f"Number of high-frequency users: {high_freq_user_count}")
    print(f"Number of low-frequency users: {low_freq_user_count}")

    # Step 3: Save preprocessed data
    charging_df.to_csv(os.path.join(output_dir, 'Preprocessed_Data.csv'), index=False)
    print("\nPreprocessed data saved to: Preprocessed_Data.csv")

    # Step 4: Run price elasticity analysis
    print("\n=== Starting Price Elasticity Analysis ===")
    # Overall elasticity
    overall_elasticity = calculate_price_elasticity(charging_df)
    print("\nOverall Price Elasticity:")
    print(overall_elasticity.round(4))

    # Grouped elasticity
    group_dimensions = ['user_type', 'site_type']
    elasticity_results = {'Overall': overall_elasticity}

    for dim in group_dimensions:
        grouped_elasticity = calculate_price_elasticity(charging_df, groupby_col=dim)
        elasticity_results[dim] = grouped_elasticity
        print(f"\nPrice Elasticity by {dim.replace('_', ' ').title()}:")
        print(grouped_elasticity.round(4))

    # Save elasticity results
    with open(os.path.join(output_dir, 'Elasticity_Analysis_Results.pkl'), 'wb') as f:
        pickle.dump(elasticity_results, f)
    print("\nElasticity results saved to: Elasticity_Analysis_Results.pkl")

    # Step 5: Run K-S test
    print("\n=== Starting K-S Test (Charging Hour Distribution) ===")
    ks_results, grouped_hours = ks_test_charging_hour(charging_df)

    # Print K-S test results
    print("\nK-S Test Results (High vs. Low Frequency Users):")
    for key, value in ks_results.items():
        if key != 'Group Statistics':
            print(f"{key}: {value}")
    print("\nGroup Statistics (Charging Hours):")
    print(ks_results['Group Statistics'])

    # Save K-S test results
    with open(os.path.join(output_dir, 'KS_Test_Results.pkl'), 'wb') as f:
        pickle.dump(ks_results, f)
    print("\nK-S test results saved to: KS_Test_Results.pkl")

    # Step 6: Generate all visualizations
    print("\n=== Generating Visualizations ===")
    visualize_results(elasticity_results, charging_df)
    plot_ks_results(grouped_hours, output_dir)

    # Step 7: Compile all results into Excel with R-squared values
    combined_results = pd.DataFrame(columns=[
        'Group', 'Elasticity', 'P-Value (Elasticity)',
        'R-squared (Log-Log Model)', 'Sample Size', 'Dimension'
    ])

    for dim, result_df in elasticity_results.items():
        # Rename columns for consistency
        result_renamed = result_df.rename(columns={
            'group': 'Group',
            'elasticity': 'Elasticity',
            'p_value': 'P-Value (Elasticity)',
            'r_squared': 'R-squared (Log-Log Model)',
            'sample_size': 'Sample Size'
        })
        result_renamed['Dimension'] = dim.replace('_', ' ').title()
        # Keep only relevant columns
        result_renamed = result_renamed[combined_results.columns]
        # Remove empty rows/columns
        result_renamed = result_renamed.dropna(how='all').dropna(axis=1, how='all')
        # Append to combined results
        combined_results = pd.concat([combined_results, result_renamed], ignore_index=True)

    # Add K-S test summary to Excel
    ks_summary = pd.DataFrame({
        'Group': ['K-S Test (High vs. Low Frequency)'],
        'Elasticity': [np.nan],
        'P-Value (Elasticity)': [np.nan],
        'R-squared (Log-Log Model)': [np.nan],
        'Sample Size': [
            f"High: {len(grouped_hours['High-Frequency User'])}, Low: {len(grouped_hours['Low-Frequency User'])}"],
        'Dimension': f"K-S Statistic: {ks_results['K-S Statistic (D)']}, P-Value: {ks_results['p-value']}"
    })
    combined_results = pd.concat([combined_results, ks_summary], ignore_index=True)

    # Save combined Excel file
    combined_results.to_excel(os.path.join(output_dir, 'Combined_Analysis_Results.xlsx'), index=False)
    print("\nCombined results (elasticity + K-S test) saved to: Combined_Analysis_Results.xlsx")

    return elasticity_results, ks_results, charging_df


# ----------------------
# 7. Program Entry Point
# ----------------------
if __name__ == "__main__":
    print("===== Starting Integrated Analysis: Price Elasticity + K-S Test =====")
    elasticity_results, ks_results, charging_data = analyze_elasticity_and_ks_test()
    print("\n===== Integrated Analysis Completed Successfully =====")

    # Print key findings summary
    print("\n=== Key Findings Summary ===")
    # 1. Overall elasticity
    overall_elasticity_val = elasticity_results['Overall']['elasticity'].values[0]
    overall_r_squared = elasticity_results['Overall']['r_squared'].values[0]
    print(f"1. Overall Price Elasticity: {overall_elasticity_val:.2f}")
    print(f"   Log-Log Regression R-squared: {overall_r_squared:.4f}")

    # 2. Most/least price-sensitive groups
    max_elasticity = -np.inf
    min_elasticity = np.inf
    max_group, min_group = "", ""
    max_dim, min_dim = "", ""
    max_r_squared, min_r_squared = 0, 1

    for dim in elasticity_results:
        if dim == 'Overall':
            continue
        for _, row in elasticity_results[dim].iterrows():
            if not np.isnan(row['elasticity']):
                # Most sensitive
                if abs(row['elasticity']) > abs(max_elasticity):
                    max_elasticity = row['elasticity']
                    max_group = row['group']
                    max_dim = dim
                    max_r_squared = row['r_squared']
                # Least sensitive
                if abs(row['elasticity']) < abs(min_elasticity):
                    min_elasticity = row['elasticity']
                    min_group = row['group']
                    min_dim = dim
                    min_r_squared = row['r_squared']

    print(
        f"2. Most Price-Sensitive Group: {max_dim.replace('_', ' ').title()} - {max_group} "
        f"(Elasticity: {max_elasticity:.2f}, R²: {max_r_squared:.4f})"
    )
    print(
        f"3. Least Price-Sensitive Group: {min_dim.replace('_', ' ').title()} - {min_group} "
        f"(Elasticity: {min_elasticity:.2f}, R²: {min_r_squared:.4f})"
    )
    # 4. K-S test conclusion
    print(f"4. K-S Test Conclusion: {ks_results['Conclusion']}")
