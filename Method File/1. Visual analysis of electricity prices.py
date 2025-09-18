import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'  # Use Arial font
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of negative signs

sns.set(style="whitegrid")  # Set Seaborn style

# Read CSV file
data_file = '/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Time-of-use_Price.csv'
df = pd.read_csv(data_file, encoding='ISO-8859-1')

# Define colors for each electricity price period
colors = {
    0.3784: 'green',  # Off-Peak period
    0.9014: 'yellow', # Peak period
    1.2064: 'red'     # Super-Peak period
}

# Create plot
plt.figure(figsize=(10, 6), dpi=500)

# Iterate through each time period and plot
for index, row in df.iterrows():
    start_hour = int(row['Time Period'].split('-')[0].split(':')[0])
    end_hour = int(row['Time Period'].split('-')[1].split(':')[0])
    price = row['Electricity Price(Yuan/kWh)']
    color = colors[price]
    # Plot semi-transparent filled area
    plt.fill_between([start_hour, end_hour], [price, price], color=color, alpha=0.3, label=f'{start_hour}-{end_hour} {price} Yuan/kWh' if start_hour == 0 else "")
    # Plot boundary line
    plt.plot([start_hour, end_hour], [price, price], color=color, linewidth=2)

# Set x-axis to 24-hour format
plt.xticks(range(25), fontsize=14)
plt.yticks(fontsize=14)

# Add title and labels
plt.title('Time-of-Use Electricity Prices', fontsize=16)
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Electricity Price (Yuan/kWh)', fontsize=14)

# Remove empty space on both sides of x-axis
plt.xlim(0, 24)

# Add legend
patch1 = mpatches.Patch(color='green', label='Valley (0.3784 Yuan/kWh)')
patch2 = mpatches.Patch(color='yellow', label='Flat (0.9014 Yuan/kWh)')
patch3 = mpatches.Patch(color='red', label='Peak (1.2064 Yuan/kWh)')
plt.legend(handles=[patch1, patch2, patch3], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=14)

# Display grid
plt.grid(True)

# Ensure the directory exists
save_path = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Electricity price analysis/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Show and save figure
plt.tight_layout()
plt.savefig(os.path.join(save_path, "time_of_electricity_pricing.png"), dpi=500)