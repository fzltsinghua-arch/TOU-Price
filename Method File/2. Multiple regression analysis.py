import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap


# Set font to Arial
plt.rcParams['font.family'] = 'Arial'  # Use Arial font
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of negative signs

sns.set(style="whitegrid")  # Set Seaborn style

# Create green gradient color mapping
green_colors = sns.color_palette("Greens", 10)
green_cmap = LinearSegmentedColormap.from_list("custom_greens", green_colors)

# ===== 1. Load charging data =====
data_file = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Charging_Data.csv"
try:
    df = pd.read_csv(data_file, encoding='ISO-8859-1')
    print(f"Original data count: {len(df)}")
    df = df[df['Transaction power/kwh'] > 0]  # Remove abnormal values
    print(f"Cleaned data count: {len(df)}")
except FileNotFoundError:
    print(f"Error: The file '{data_file}' was not found. Please check the file path.")
    exit()

# Convert time format
df['Start Time'] = pd.to_datetime(df['Start Time'])  # Directly convert string to datetime
df['hour'] = df['Start Time'].dt.hour
df['minute'] = df['Start Time'].dt.minute
df['time_str'] = df['hour'].astype(str).str.zfill(2) + ":" + df['minute'].astype(str).str.zfill(2)

# Rename columns to avoid spaces
df.rename(columns={'Location Information': 'station_type'}, inplace=True)

# Check unique values of station_type
unique_station_types = df['station_type'].unique()
print(f"Unique Station Types: {unique_station_types}")
print(f"Number of unique Station Types: {len(unique_station_types)}")


# ===== 2. Match time-of-use electricity prices =====
def get_price_period(hour):
    """
    Determines the electricity price based on the hour.
    This function is more robust as it uses the integer hour directly,
    avoiding the ValueError associated with parsing '24:00'.
    """
    if 0 <= hour < 8:
        return 0.3784
    elif 8 <= hour < 11:
        return 0.9014
    elif 11 <= hour < 13:
        return 0.3784
    elif 13 <= hour < 19:
        return 0.9014
    elif 19 <= hour < 21:
        return 1.2064
    elif 21 <= hour < 22:
        return 0.9014
    elif 22 <= hour <= 23: # Covers the 22:00 to 24:00 period
        return 0.3784
    return np.nan

df['electricity_price'] = df['hour'].apply(get_price_period)

# ===== 3. Define user types (High-frequency / Low-frequency) =====
user_freq = df['UserID'].value_counts()
threshold_freq = user_freq.median()  # Use median for division
user_type_map = {uid: ("High-Frequency User" if count > threshold_freq else "Low-Frequency User")
                 for uid, count in user_freq.items()}
df['user_type'] = df['UserID'].map(user_type_map)

# ===== 4. Multi-factor ANOVA =====
model = ols('Q("Transaction power/kwh") ~ C(electricity_price) * C(user_type) * C(station_type)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # typ=2 is commonly used for multi-factor ANOVA

print("\nMulti-factor ANOVA results:")
print(anova_table)

# Save multi-factor ANOVA results to Excel file
anova_file = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Multivariate analysis/anova_results.xlsx"
anova_dir = os.path.dirname(anova_file)

if not os.path.exists(anova_dir):
    os.makedirs(anova_dir)
    print(f"Directory {anova_dir} has been created")

anova_table.to_excel(anova_file, index=True)
print(f"Multi-factor ANOVA results have been saved to {anova_file}")

# ===== 5. Visualization =====

# 1. Combined plot for user types and station types in 3x5 layout
# Function: Each subplot uses bar charts for charging energy (green gradient) and outline lines for charging count (dual Y-axis)
all_categories = ['User: High-Frequency User', 'User: Low-Frequency User'] + [f'Station: {s}' for s in unique_station_types]
num_plots = len(all_categories)
fig, axes = plt.subplots(5, 3, figsize=(25, 15))  # Fixed 3x5 layout
axes = axes.flatten()  # Flatten to 1D array for easy indexing

# Adjust colormap to use darker green tones (modify range as needed)
green_cmap = plt.colormaps.get_cmap('Greens')  # Updated to use modern Matplotlib API
# Generate darker green palette (increase proportion of dark colors)
green_colors = [green_cmap(0.6), green_cmap(0.7), green_cmap(0.8), green_cmap(0.9), green_cmap(1.0)]

# Iterate through each category to plot subgraphs
for i, category in enumerate(all_categories):
    ax = axes[i]

    # 1. Filter data based on category
    if category == 'User: High-Frequency User':
        data = df[df['user_type'] == 'High-Frequency User']
    elif category == 'User: Low-Frequency User':
        data = df[df['user_type'] == 'Low-Frequency User']
    else:
        # Extract station type from category name
        station_name = category.replace('Station: ', '')
        data = df[df['station_type'] == station_name]

    # 2. Group by 0~24 hours and calculate core metrics (charging energy + charging count)
    hours = np.arange(25)  # 0,1,...,24
    hourly_metrics = data.groupby('hour').agg(
        total_energy=('Transaction power/kwh', 'sum'),
        charge_count=('Transaction power/kwh', 'count')
    ).reindex(hours, fill_value=0)
    # Make the value at hour 24 the same as hour 0 to represent a cycle
    hourly_metrics.loc[24] = hourly_metrics.loc[0]

    # 3. Plot "charging energy bar chart" (darker green gradient)
    bars = ax.bar(
        x=hourly_metrics.index,
        height=hourly_metrics['total_energy'],
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5,
        label='Charging Load'
    )
    # Adjust bar color depth: use range 0.4-1.0 to skip light tones
    for j, bar in enumerate(bars):
        adjusted_index = 0.4 + (j / len(bars)) * 0.6
        bar.set_facecolor(green_cmap(adjusted_index))

    # 4. Plot "charging count outline" (use darker line color)
    ax2 = ax.twinx()
    # Adjust line color: use darkest green in green_colors
    line, = ax2.plot(
        hourly_metrics.index,
        hourly_metrics['charge_count'],
        color=green_colors[-1],
        linewidth=2.5,
        linestyle='-',
        marker='o',
        markersize=4,
        label='Charging Count'
    )

    # 5. Subplot style configuration (keep other styles unchanged)
    ax.set_title(category, fontsize=16, pad=15, color='black')
    ax.set_ylabel('Total Charging load (kWh)', fontsize=14, color='black')
    ax.tick_params(axis='y', labelsize=14, colors='black')
    ax2.set_ylabel('Charging Count', fontsize=14, color='black')
    ax2.tick_params(axis='y', labelsize=14, colors='black')
    ax.set_xlabel('Hour', fontsize=14, color='black')
    ax.set_xticks(range(0, 25, 1))
    ax.set_xticklabels(range(0, 25, 1))
    ax.tick_params(axis='x', labelsize=14, colors='black')
    # Set x-limits to go from 0 to 24
    ax.set_xlim(-0.5, 24.5)

    # 6. Add legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='upper right',
        fontsize=14,
        labelcolor='black',
        frameon=True,
        framealpha=0.9,
        edgecolor='gray'
    )

    # 7. Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

# Hide unused subplots
for i in range(num_plots, 3 * 5):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(
    "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Multivariate analysis/combined_load_count_distributions.png",
    dpi=1000,
    bbox_inches='tight'
)
plt.close()

# 2. Relationship between time-of-use electricity prices and charging power for different user types (violin plot)
plt.figure(figsize=(10, 6))
sns.violinplot(x='electricity_price', y='Transaction power/kwh', hue='user_type', data=df, palette='viridis', split=True)
plt.title('Relationship Between Electricity Prices and Charging Power for Different User Types', fontsize=16)
plt.xlabel('Electricity Price', fontsize=14)
plt.ylabel('Charging Power (kWh)', fontsize=14)
plt.legend(title='User Type', loc='upper right', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Multivariate analysis/user_price_power_relationship_violinplot.png", dpi=500)
plt.close()

# 3. Relationship between time-of-use electricity prices and charging power for different charging stations (heatmap)
pivot_table = df.pivot_table(index='electricity_price', columns='station_type', values='Transaction power/kwh', aggfunc="mean")
plt.figure(figsize=(12, 6))
ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
cbar = ax.collections[0].colorbar
cbar.set_label('Average Charging Power (kWh)', fontsize=14)
plt.title('Relationship Between Time-of-Use Electricity Prices and Charging Power for Different Charging Stations', fontsize=16)
plt.xlabel('Charging Station Type', fontsize=14)
plt.ylabel('Electricity Price', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Multivariate analysis/station_price_power_relationship_heatmap.png", dpi=500)
plt.close()

# 4. Create a pivot table containing all electricity prices, user types, and station types
pivot_table = df.pivot_table(index='station_type', columns=['electricity_price', 'user_type'],
                             values='Transaction power/kwh', aggfunc="mean")

# Plot three heatmaps, each corresponding to a different electricity price
electricity_prices = pivot_table.columns.levels[0]  # Get all unique electricity prices

fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Create a 1x3 subplot layout
for i, price in enumerate(electricity_prices):
    ax = axes[i]
    # Generate heatmap for current price with annotations and viridis color map
    sns.heatmap(pivot_table[price], annot=True, fmt=".2f", cmap="viridis", ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_label(f'Electricity Price: {price}', fontsize=14)
    ax.set_title(f'Electricity Price: {price}', fontsize=16)
    ax.set_xlabel('User Type', fontsize=14)
    ax.set_ylabel('Station Type', fontsize=14)

    # 获取当前x轴刻度标签文本（用户类型）
    xtick_labels = [label.get_text() for label in ax.get_xticklabels()]

    # 对每个标签进行拆分，插入换行符（根据实际标签文本调整拆分逻辑）
    wrapped_labels = []
    for label in xtick_labels:
        # 示例1：按空格拆分，在第一个空格后换行（适用于"High-Frequency Users"等）
        if ' ' in label:
            parts = label.split(' ', 1)  # 只拆分为两部分
            wrapped = f"{parts[0]}\n{parts[1]}"
        # 示例2：按括号拆分（适用于"High-Frequency (≥ Median)"等）
        elif '(' in label:
            parts = label.split('(', 1)
            wrapped = f"{parts[0]}(\n{parts[1]}"  # 在括号后换行
        else:
            wrapped = label  # 无需拆分的标签保持原样
        wrapped_labels.append(wrapped)

    # 应用处理后的标签，设置字体大小和旋转角度（避免重叠）
    ax.set_xticklabels(wrapped_labels, fontsize=12, rotation=0, ha='center')

    ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()  # Adjust layout to prevent overlap
# Corrected the path to avoid FileNotFoundError
plt.savefig("/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Multivariate analysis/charging_power_heatmap_by_price.png", dpi=500)


print("All visualizations have been generated and saved.")
