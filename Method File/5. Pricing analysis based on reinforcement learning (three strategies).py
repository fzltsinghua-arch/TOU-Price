import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import warnings
import os
import pickle
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime

# Global Settings
warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
torch.set_num_threads(mp.cpu_count() // 2)

# Ensure standard output supports Chinese characters
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Result saving directory
output_dir = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/V2G_论文程序/处理结果/Reinforcement learning pricing analysis"
for subdir in ['site_differentiation_model', 'user_differentiation_model', 'site_user_differentiation_model']:
    if not os.path.exists(os.path.join(output_dir, subdir)):
        os.makedirs(os.path.join(output_dir, subdir))

# Font settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")


# ----------------------
# 1. Data Loading and Preprocessing
# ----------------------
def load_and_preprocess_data():
    """Load and preprocess data, integrating steps required for elasticity calculation"""
    charging_data_file = "/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Charging_Data.csv"
    tou_data_file = '/Users/bush/PycharmProjects/山东V2G项目/在线V2G数据的跑通/Dataset/Time-of-use_Price.csv'

    # Load data (compatible with different encodings)
    try:
        charging_df = pd.read_csv(charging_data_file, encoding='ISO-8859-1')
    except:
        charging_df = pd.read_csv(charging_data_file, encoding='utf-8')
    try:
        tou_df = pd.read_csv(tou_data_file, encoding='ISO-8859-1')
    except:
        tou_df = pd.read_csv(tou_data_file, encoding='utf-8')

    # Time column processing
    charging_df['Start Time'] = pd.to_datetime(charging_df['Start Time'], errors='coerce')
    charging_df['End Time'] = pd.to_datetime(charging_df['End Time'], errors='coerce')
    charging_df = charging_df.dropna(subset=['Start Time', 'End Time'])

    # Electricity price period division
    def get_price_period(time):
        total_min = time.hour * 60 + time.minute
        if (0 <= total_min < 480) or (660 <= total_min < 780) or (1320 <= total_min < 1440):
            return 'Valley'
        elif (480 <= total_min < 660) or (780 <= total_min < 1140) or (1260 <= total_min < 1320):
            return 'Flat'
        else:
            return 'Peak'

    # Price mapping and data cleaning
    charging_df['price_period'] = charging_df['Start Time'].apply(get_price_period)
    price_map = {'Valley': 0.3784, 'Flat': 0.9014, 'Peak': 1.2064}
    charging_df['period_price'] = charging_df['price_period'].map(price_map)
    charging_df = charging_df[(charging_df['Transaction power/kwh'] > 0) &
                              (charging_df['price_period'] != 'Unknown')]

    # Derived feature calculation
    charging_df['charging_duration'] = (charging_df['End Time'] - charging_df['Start Time']).dt.total_seconds() / 3600
    user_charging_count = charging_df['UserID'].value_counts()
    median_count = np.median(user_charging_count.values)
    charging_df['user_type'] = charging_df['UserID'].apply(
        lambda x: 'High-Frequency' if user_charging_count[x] >= median_count else 'Low-Frequency'
    )
    charging_df['site_type'] = charging_df.get('Location Information', 'Unknown').fillna('Unknown')
    charging_df['hour'] = charging_df['Start Time'].dt.hour

    # Temperature feature processing (generate reasonable random values if no data)
    if 'Temperature' not in charging_df.columns:
        np.random.seed(42)
        charging_df['Temperature'] = np.random.normal(25, 5, len(charging_df))
    else:
        temp_mean, temp_std = charging_df['Temperature'].mean(), charging_df['Temperature'].std()
        charging_df['Temperature'] = charging_df['Temperature'].clip(temp_mean - 3 * temp_std, temp_mean + 3 * temp_std)

    # Save preprocessing results
    charging_df.to_csv(os.path.join(output_dir, 'Preprocessed_Data.csv'), index=False)
    print("Preprocessed data has been saved to 'Preprocessed_Data.csv'")
    return charging_df


# ----------------------
# 2. Price Elasticity Calculation Functions
# ----------------------
def calculate_price_elasticity(df, groupby_col=None):
    """Calculate price elasticity (supports single/multi-column grouping for dual-dimensional differentiation)"""

    def _format_group_name(group_key, groupby_cols):
        """Format group name (multi-column groups joined with | to avoid conflicts with - in names)"""
        if isinstance(group_key, tuple):
            return "|".join([str(k) for k in group_key])  # Use | instead of -
        return str(group_key)

    # 1. Overall elasticity calculation
    if groupby_col is None:
        price_groups = df.groupby('period_price')['Transaction power/kwh'].agg(['mean', 'count']).reset_index()
        price_groups.columns = ['price', 'quantity', 'count']
        if len(price_groups) < 2:
            return pd.DataFrame({
                'group': ['Overall'], 'elasticity': [np.nan], 'p_value': [np.nan], 'sample_size': [len(df)]
            })

        price_groups = price_groups.sort_values('price')
        elasticity_values = []
        for i in range(1, len(price_groups)):
            P1, P2 = price_groups['price'].iloc[i - 1:i + 1].values
            Q1, Q2 = price_groups['quantity'].iloc[i - 1:i + 1].values
            elasticity = ((Q2 - Q1) / ((Q1 + Q2) / 2)) / ((P2 - P1) / ((P1 + P2) / 2))
            elasticity_values.append(elasticity)

        slope, _, r_value, p_value, _ = stats.linregress(np.log(price_groups['price']),
                                                         np.log(price_groups['quantity']))
        return pd.DataFrame({
            'group': ['Overall'], 'elasticity': [np.mean(elasticity_values)],
            'p_value': [p_value], 'sample_size': [len(df)]
        })

    # 2. Calculate elasticity by group
    else:
        results = []
        valid_group_cols = groupby_col if isinstance(groupby_col, list) else [groupby_col]
        valid_group_cols = [col for col in valid_group_cols if col in df.columns]
        if not valid_group_cols:
            raise ValueError(f"Grouping column(s) {groupby_col} not in dataset")

        for group_key, group_data in df.groupby(valid_group_cols):
            price_groups = group_data.groupby('period_price')['Transaction power/kwh'].agg(
                ['mean', 'count']).reset_index()
            price_groups.columns = ['price', 'quantity', 'count']
            group_name = _format_group_name(group_key, valid_group_cols)

            if len(price_groups) < 2:
                results.append({
                    'group': group_name, 'elasticity': [np.nan], 'p_value': [np.nan], 'sample_size': [len(group_data)]
                })
                continue

            # Midpoint formula for elasticity calculation
            price_groups = price_groups.sort_values('price')
            elasticity_values = []
            for i in range(1, len(price_groups)):
                P1, P2 = price_groups['price'].iloc[i - 1:i + 1].values
                Q1, Q2 = price_groups['quantity'].iloc[i - 1:i + 1].values
                elasticity = ((Q2 - Q1) / ((Q1 + Q2) / 2)) / ((P2 - P1) / ((P1 + P2) / 2))
                elasticity_values.append(elasticity)

            slope, _, r_value, p_value, _ = stats.linregress(np.log(price_groups['price']),
                                                             np.log(price_groups['quantity']))
            results.append({
                'group': group_name, 'elasticity': np.mean(elasticity_values),
                'p_value': p_value, 'sample_size': len(group_data)
            })
        return pd.DataFrame(results)


def analyze_price_elasticity(df):
    """Main function: perform price elasticity analysis (including single/dual dimensions)"""
    print(f"\nStarting price elasticity analysis, valid data volume: {len(df)} records")
    group_dimensions = ['user_type', 'site_type', ['site_type', 'user_type']]
    elasticity_results = {'Overall': calculate_price_elasticity(df)}

    for dim in group_dimensions:
        dim_name = 'site_user' if isinstance(dim, list) and dim == ['site_type', 'user_type'] else dim
        elasticity_results[dim_name] = calculate_price_elasticity(df, groupby_col=dim)
        print(f"\nPrice elasticity grouped by {dim_name.upper()}:")
        print(elasticity_results[dim_name])

    # Save elasticity results
    with open(os.path.join(output_dir, 'Elasticity_Analysis_Results.pkl'), 'wb') as f:
        pickle.dump(elasticity_results, f)
    print("\nAll elasticity analysis results saved to 'Elasticity_Analysis_Results.pkl'")
    return elasticity_results


# ----------------------
# 3. PPO Agent
# ----------------------
class FixedPPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound=0.3):
        super(FixedPPOAgent, self).__init__()
        self.action_bound = action_bound

        # Shared feature network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )

        # Actor network (outputs action distribution parameters)
        self.actor_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network (outputs state value)
        self.critic_head = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.shared(state)

    def get_action(self, state):
        """Generate action and log probability"""
        features = self.forward(state)
        mu = torch.tanh(self.actor_head(features)) * 5.0  # Limit mean range
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)

        # Outlier handling
        mu = torch.nan_to_num(mu, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)

        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Map action to [1-action_bound, 1+action_bound]
        action = torch.tanh(action) * self.action_bound + 1.0
        return action, log_prob

    def evaluate_action(self, state, action):
        """Evaluate log probability of action and state value"""
        features = self.forward(state)
        mu = torch.tanh(self.actor_head(features)) * 5.0
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)

        # Outlier handling
        mu = torch.nan_to_num(mu, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)

        # Action denormalization
        action_normalized = (action - 1.0) / self.action_bound
        action_normalized = torch.clamp(action_normalized, -1.0 + 1e-6, 1.0 - 1e-6)
        action_normalized = torch.atanh(action_normalized)

        dist = Normal(mu, std)
        log_prob = dist.log_prob(action_normalized)
        value = self.critic_head(features)

        # Log probability outlier handling
        log_prob = torch.nan_to_num(log_prob, nan=0.0)
        return log_prob, value


# ----------------------
# 4. Environment Definition
# ----------------------
class TargetedV2GEnv:
    def __init__(self, df, elasticity_results, differentiation_type='site'):
        self.df = df.copy()
        self.base_prices = {'Valley': 0.3784, 'Flat': 0.9014, 'Peak': 1.2064}
        self.period_map = {'Valley': 0, 'Flat': 1, 'Peak': 2}
        self.differentiation_type = differentiation_type

        # Format elasticity results
        self.elasticity_results = self._format_elasticity_results(elasticity_results)

        # Initialize entity types (compound entities separated by |)
        if self.differentiation_type == 'site_user':
            self.site_user_types = self._get_valid_site_user_combinations()
            self.entity_types = self.site_user_types
        else:
            self.site_types = self.df['site_type'].unique()
            self.user_types = self.df['user_type'].unique()
            self.entity_types = self.site_types if differentiation_type == 'site' else self.user_types

        # Precompute baseline metrics and hourly masks
        self.baseline_metrics = self.calculate_metrics(is_simulated=False)
        self._precompute_hourly_masks()
        self.baseline_peak_diff = self.baseline_metrics['Peak-Valley Difference']
        self.baseline_util_std = self.baseline_metrics['Utilization Standard Deviation']
        self.baseline_revenue = self.baseline_metrics['Total Revenue']

        # Metrics for visualization
        self.baseline_metrics['peak_valley'] = self.baseline_peak_diff
        self.baseline_metrics['util_std'] = self.baseline_util_std
        self.baseline_metrics['welfare_change'] = 0.0
        self.baseline_metrics['revenue'] = self.baseline_revenue

    def _get_valid_site_user_combinations(self):
        """Generate valid compound entities (separated by |)"""
        all_combinations = [(site, user) for site in self.df['site_type'].unique()
                            for user in self.df['user_type'].unique()]
        valid_combinations = []
        for site, user in all_combinations:
            mask = (self.df['site_type'] == site) & (self.df['user_type'] == user)
            if mask.sum() > 0:
                valid_combinations.append(f"{site}|{user}")  # Use | instead of - to avoid split conflicts
        if not valid_combinations:
            raise ValueError("No valid 'site|user' combinations, please check site_type and user_type columns")
        return valid_combinations

    def _format_elasticity_results(self, elasticity_results):
        """Format elasticity results (adapt to compound entities)"""
        formatted = {'site': {}, 'user': {}, 'site_user': {}, 'overall': None}
        formatted['overall'] = elasticity_results['Overall']['elasticity'].values[0]

        # Single-dimensional elasticity
        if 'site_type' in elasticity_results:
            for _, row in elasticity_results['site_type'].iterrows():
                formatted['site'][row['group']] = row['elasticity']
        if 'user_type' in elasticity_results:
            for _, row in elasticity_results['user_type'].iterrows():
                formatted['user'][row['group']] = row['elasticity']

        # Dual-dimensional elasticity (matching |-separated entity names)
        if 'site_user' in elasticity_results:
            for _, row in elasticity_results['site_user'].iterrows():
                formatted['site_user'][row['group']] = row['elasticity']
        else:
            print("No dual-dimensional elasticity found, filling with single-dimensional average")
            for combo in self.site_user_types:
                site, user = combo.split('|')  # Split by |, ensuring exactly 2 values
                site_elastic = formatted['site'].get(site, formatted['overall'])
                user_elastic = formatted['user'].get(user, formatted['overall'])
                formatted['site_user'][combo] = (site_elastic + user_elastic) / 2

        # Fill NaN values
        for group_type in ['site', 'user', 'site_user']:
            for group_name in formatted[group_type]:
                if np.isnan(formatted[group_type][group_name]):
                    formatted[group_type][group_name] = formatted['overall']
        if np.isnan(formatted['overall']):
            formatted['overall'] = -0.25
            print("Warning: Overall elasticity invalid, using default value -0.25")
        return formatted

    def _precompute_hourly_masks(self):
        self.hour_masks = {hour: self.df['hour'] == hour for hour in range(24)}

    def calculate_metrics(self, is_simulated=False, simulated_df=None):
        """Calculate evaluation metrics (peak-valley, utilization, welfare, revenue)"""
        df = simulated_df if simulated_df is not None else self.df
        power_col = 'simulated_power' if is_simulated else 'Transaction power/kwh'
        price_col = 'simulated_price' if is_simulated else 'period_price'
        metrics = {}

        # 1. Peak-valley metrics
        hourly_load = df.groupby('hour')[power_col].sum()
        max_load, min_load = hourly_load.max(), hourly_load.min()
        metrics['Peak-Valley Difference'] = max_load - min_load
        metrics['Peak Load'] = max_load
        metrics['hourly_load'] = hourly_load

        # 2. Utilization metrics
        total_power = df[power_col].sum()
        period_usage = df.groupby('price_period')[power_col].sum()
        metrics['Total Charging Power'] = total_power
        for period in ['Valley', 'Flat', 'Peak']:
            metrics[f'{period} Utilization'] = period_usage.get(period, 0) / (total_power + 1e-8)
        metrics['Utilization Standard Deviation'] = np.std([
            metrics[f'{p} Utilization'] for p in ['Valley', 'Flat', 'Peak']
        ])

        # 3. User welfare metrics (ΔCS: negative = improvement, positive = loss)
        if is_simulated:
            P_original = np.mean(list(self.base_prices.values()))
            Q_original = self.baseline_metrics['Total Charging Power']
            P_new = np.mean(df[price_col])
            Q_new = df[power_col].sum()
            delta_cs = 0.5 * (P_original - P_new) * (Q_original + Q_new)
            metrics['User Welfare Change'] = delta_cs
        else:
            metrics['User Welfare Change'] = 0.0

        # 4. Revenue metrics
        df['revenue'] = df[power_col] * df[price_col]
        metrics['Total Revenue'] = df['revenue'].sum()

        # Derived improvement rate metrics
        if is_simulated:
            metrics['peak_valley_reduction'] = (self.baseline_peak_diff - metrics['Peak-Valley Difference']) / (
                    self.baseline_peak_diff + 1e-8)
            metrics['util_std_reduction'] = (self.baseline_util_std - metrics['Utilization Standard Deviation']) / (
                    self.baseline_util_std + 1e-8)
            metrics['revenue_increase'] = (metrics['Total Revenue'] - self.baseline_revenue) / (
                    self.baseline_revenue + 1e-8)
            metrics['welfare_improvement_rate'] = -metrics['User Welfare Change']  # Negative ΔCS indicates improvement

        # Unified fields for visualization
        metrics['peak_valley'] = metrics['Peak-Valley Difference']
        metrics['util_std'] = metrics['Utilization Standard Deviation']
        metrics['welfare_change'] = metrics['User Welfare Change']
        metrics['revenue'] = metrics['Total Revenue']
        return metrics

    def get_state(self, entity_type, hour):
        """Get entity state (compound entities split by |)"""
        period = self._get_period(hour)
        period_encoded = self.period_map[period]

        # Compound entity state acquisition
        if self.differentiation_type == 'site_user':
            try:
                site, user = entity_type.split('|')  # Safe split: only split by |, ensuring 2 values
                mask = (self.df['site_type'] == site) & (self.df['user_type'] == user) & self.hour_masks[hour]
            except ValueError:
                print(f"Invalid compound entity format: {entity_type}, should be 'site|user'")
                mask = self.hour_masks[hour]  # Degrade to global hourly mask
        else:
            mask = (self.df[f'{self.differentiation_type}_type'] == entity_type) & self.hour_masks[hour]

        # State feature normalization
        avg_load = self.df[mask]['Transaction power/kwh'].mean() or 0
        temp = self.df[self.hour_masks[hour]]['Temperature'].mean() or 25
        hour_norm = hour / 24.0
        avg_load_norm = avg_load / (self.df['Transaction power/kwh'].max() + 1e-8)
        temp_norm = (temp - 10) / 30.0  # Assuming temperature range 10-40℃

        return torch.FloatTensor([hour_norm, period_encoded, avg_load_norm, temp_norm]).to(DEVICE)

    def step(self, entity_type, period, price_factor):
        """Execute action (price adjustment) and return reward"""
        price_factor = np.clip(price_factor, 0.5, 1.5)  # Limit price factor range

        # Compound entity mask (split by |)
        if self.differentiation_type == 'site_user':
            try:
                site, user = entity_type.split('|')
                mask = (self.df['site_type'] == site) & (self.df['user_type'] == user) & (
                        self.df['price_period'] == period)
            except ValueError:
                print(f"Invalid compound entity format: {entity_type}")
                mask = self.df['price_period'] == period
        else:
            mask = (self.df[f'{self.differentiation_type}_type'] == entity_type) & (self.df['price_period'] == period)

        if not mask.any():
            return 0.0, {'price_change': 0, 'load_change': 0, 'revenue_change': 0, 'peak_impact': 0,
                         'welfare_change': 0}

        # Price and load calculation
        original_price = self.base_prices[period]
        new_price = original_price * price_factor
        elasticity = self.elasticity_results[self.differentiation_type].get(entity_type,
                                                                            self.elasticity_results['overall'])
        elasticity = np.clip(elasticity, -5.0, 0.0)  # Ensure elasticity is negative (law of demand)

        price_change = (new_price - original_price) / original_price
        load_change = elasticity * price_change
        load_change = np.clip(load_change, -0.8, 0.8)  # Limit load change magnitude

        original_power = self.df.loc[mask, 'Transaction power/kwh'].sum()
        new_power = max(0.1, original_power * (1 + load_change))  # Avoid negative power

        # Reward calculation (four objectives weighted)
        hour = self.df.loc[mask, 'hour'].mode().values[0] if len(self.df[mask]) > 0 else 0
        baseline_hourly = self.baseline_metrics['hourly_load']
        current_hour_load = baseline_hourly.get(hour, 0)

        # 1. Peak-valley reward (positive for peak shaving and valley filling)
        peak_impact = -load_change if current_hour_load == baseline_hourly.max() else (
            load_change if current_hour_load == baseline_hourly.min() else 0)
        peak_reward = np.clip(peak_impact / (self.baseline_peak_diff / 100), -1.0, 1.0)

        # 2. Utilization reward (positive for reduced standard deviation)
        period_idx = ['Valley', 'Flat', 'Peak'].index(period)
        baseline_utils = [self.baseline_metrics[f'{p} Utilization'] for p in ['Valley', 'Flat', 'Peak']]
        new_utils = baseline_utils.copy()
        total_power_change = new_power - original_power
        total_baseline = self.baseline_metrics['Total Charging Power']
        if total_baseline > 0:
            new_utils[period_idx] = (baseline_utils[period_idx] * total_baseline + total_power_change) / (
                    total_baseline + total_power_change + 1e-8)
            new_utils = [u / sum(new_utils) for u in new_utils]
            util_reward = np.clip((self.baseline_util_std - np.std(new_utils)) / (self.baseline_util_std + 1e-8), -1.0,
                                  1.0)
        else:
            util_reward = 0.0

        # 3. Welfare reward (positive for negative ΔCS)
        P_org, Q_org = np.mean(list(self.base_prices.values())), original_power
        P_new, Q_new = new_price, new_power
        delta_cs = 0.5 * (P_org - P_new) * (Q_org - Q_new)
        welfare_reward = np.clip(delta_cs * 0.01, -1.0, 1.0)  # Scaling factor

        # 4. Revenue reward (positive for revenue increase)
        original_rev = original_price * original_power
        new_rev = new_price * new_power
        revenue_reward = np.clip((new_rev - original_rev) / (self.baseline_revenue + 1e-8) * 10, -1.0, 1.0)

        # Total reward (equal weights)
        total_reward = (peak_reward + util_reward + welfare_reward + revenue_reward) / 4
        total_reward = 0.0 if np.isnan(total_reward) else total_reward

        return total_reward.item(), {
            'price_change': price_change, 'load_change': load_change,
            'revenue_change': new_rev - original_rev, 'peak_impact': peak_impact,
            'welfare_change': delta_cs
        }

    def _get_period(self, hour):
        """Get price period based on hour"""
        total_min = hour * 60
        if (0 <= total_min < 480) or (660 <= total_min < 780) or (1320 <= total_min < 1440):
            return 'Valley'
        elif (480 <= total_min < 660) or (780 <= total_min < 1140) or (1260 <= total_min < 1320):
            return 'Flat'
        else:
            return 'Peak'

    def _period_to_hour(self, period):
        """Map period to representative hour"""
        return 2 if period == 'Valley' else 9 if period == 'Flat' else 20

    def simulate_strategy(self, strategy):
        """Simulate strategy effect (compound entities split by |)"""
        df_sim = self.df.copy()
        df_sim['simulated_power'] = df_sim['Transaction power/kwh']
        df_sim['simulated_price'] = df_sim['period_price']

        for entity_type, period_factors in strategy.items():
            for period, factor in period_factors.items():
                factor = np.clip(factor, 0.5, 1.5)

                # Compound entity mask
                if self.differentiation_type == 'site_user':
                    try:
                        site, user = entity_type.split('|')
                        mask = (df_sim['site_type'] == site) & (df_sim['user_type'] == user) & (
                                df_sim['price_period'] == period)
                    except ValueError:
                        print(f"Invalid compound entity format: {entity_type}")
                        mask = df_sim['price_period'] == period
                else:
                    mask = (df_sim[f'{self.differentiation_type}_type'] == entity_type) & (
                            df_sim['price_period'] == period)

                if not mask.any():
                    continue

                # Price and load adjustment
                original_price = self.base_prices[period]
                new_price = original_price * factor
                df_sim.loc[mask, 'simulated_price'] = new_price

                elasticity = self.elasticity_results[self.differentiation_type].get(entity_type,
                                                                                    self.elasticity_results['overall'])
                elasticity = np.clip(elasticity, -5.0, 0.0)
                price_change = (new_price - original_price) / original_price
                load_change = elasticity * price_change
                load_change = np.clip(load_change, -0.8, 0.8)

                df_sim.loc[mask, 'simulated_power'] *= (1 + load_change)
                df_sim.loc[mask, 'simulated_power'] = df_sim.loc[mask, 'simulated_power'].clip(lower=0.1)

        return self.calculate_metrics(is_simulated=True, simulated_df=df_sim)


# ----------------------
# 5. Training Process
# ----------------------
def stable_ppo_train(env, entity_types, model_type, epochs=200):
    """PPO training (adapted to hyperparameters for three strategies)"""
    # Strategy-specific hyperparameters
    hyper_params = {
        'user': {'lr': 0.0003, 'clip': 0.25, 'batch': 128, 'gae_lambda': 0.92},
        'site': {'lr': 0.00025, 'clip': 0.3, 'batch': 128, 'gae_lambda': 0.95},
        'site_user': {'lr': 0.0002, 'clip': 0.28, 'batch': 256, 'gae_lambda': 0.93}
        # More conservative for dual dimension
    }
    params = hyper_params[model_type]

    # Initialize agent and optimizer
    agent = FixedPPOAgent(state_dim=4, action_dim=1, action_bound=0.3).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=params['lr'], eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - 50 if epochs > 50 else epochs, eta_min=1e-5
    )

    # Training tracking variables
    best_metrics = None
    rewards_history = []
    best_peak = float('inf')
    best_util = float('inf')
    best_welfare = float('inf')
    best_revenue = -float('inf')

    for epoch in range(epochs):
        # Learning rate scheduling (no decay for first 50 epochs)
        if epoch < 50:
            current_lr = params['lr']
        else:
            if epoch == 50:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 50, eta_min=1e-5)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # Collect trajectories
        states, actions, log_probs, rewards = [], [], [], []
        for entity in tqdm(entity_types, desc=f"Epoch {epoch + 1}/{epochs} | {model_type}", leave=False):
            for hour in range(24):
                period = env._get_period(hour)
                state = env.get_state(entity, hour)
                action, log_prob = agent.get_action(state)

                # Outlier handling
                action = torch.tensor([1.0], device=DEVICE) if torch.isnan(action).any() else action
                log_prob = torch.tensor([0.0], device=DEVICE) if torch.isnan(log_prob).any() else log_prob

                reward, _ = env.step(entity, period, action.item())
                reward = 0.0 if np.isnan(reward) else reward

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)

        # Data format conversion and outlier handling
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

        states = torch.nan_to_num(states, nan=0.0)
        actions = torch.nan_to_num(actions, nan=1.0)
        log_probs = torch.nan_to_num(log_probs, nan=0.0)
        rewards = torch.nan_to_num(rewards, nan=0.0)

        # GAE advantage calculation
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_return, last_adv = 0.0, 0.0
        gamma = 0.99

        for t in reversed(range(len(rewards))):
            last_return = rewards[t] + gamma * last_return
            td_error = rewards[t] + gamma * last_return - agent.critic_head(agent.shared(states[t])).item()
            last_adv = td_error + gamma * params['gae_lambda'] * last_adv

            returns[t] = last_return
            advantages[t] = last_adv

        # Advantage normalization
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Batch training
        dataset = TensorDataset(states, actions, log_probs, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=params['batch'], shuffle=True)

        for _ in range(4):  # Multiple updates
            for batch in dataloader:
                b_states, b_actions, b_log_probs, b_returns, b_advantages = batch
                b_new_log_probs, b_values = agent.evaluate_action(b_states, b_actions)

                # Outlier handling
                b_new_log_probs = torch.nan_to_num(b_new_log_probs, nan=0.0)
                b_values = torch.nan_to_num(b_values, nan=0.0)

                # Actor loss (PPO clipping)
                ratio = torch.exp(b_new_log_probs - b_log_probs.detach())
                ratio = torch.clamp(ratio, 0.1, 10.0)  # Limit extreme ratios
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - params['clip'], 1 + params['clip']) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (value clipping)
                values_clipped = b_log_probs.detach() + torch.clamp(b_values - b_log_probs.detach(), -params['clip'],
                                                                    params['clip'])
                critic_loss1 = F.mse_loss(b_values.squeeze(), b_returns)
                critic_loss2 = F.mse_loss(values_clipped.squeeze(), b_returns)
                critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()

                # Total loss and optimization
                total_loss = actor_loss + 0.5 * critic_loss
                if torch.isnan(total_loss):
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.3)  # Gradient clipping
                optimizer.step()

        # Training tracking and best model saving
        avg_reward = np.mean(rewards.cpu().numpy())
        rewards_history.append(avg_reward)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | {model_type} | Average reward: {avg_reward:.4f} | Learning rate: {current_lr:.6f}")

        # Evaluate current strategy
        current_strategy = generate_strategy(agent, env, entity_types)
        current_metrics = env.simulate_strategy(current_strategy)

        # Multi-objective best model judgment
        improvements = 0
        improvements += 1 if current_metrics['Peak-Valley Difference'] < best_peak else 0
        improvements += 1 if current_metrics['Utilization Standard Deviation'] < best_util else 0
        improvements += 1 if current_metrics['User Welfare Change'] < best_welfare else 0
        improvements += 1 if current_metrics['Total Revenue'] > best_revenue else 0

        if improvements >= 3:
            best_peak = min(best_peak, current_metrics['Peak-Valley Difference'])
            best_util = min(best_util, current_metrics['Utilization Standard Deviation'])
            best_welfare = min(best_welfare, current_metrics['User Welfare Change'])
            best_revenue = max(best_revenue, current_metrics['Total Revenue'])

            # Save best model
            save_path = os.path.join(output_dir, f"{model_type}_differentiation_model/best_agent.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.state_dict(), save_path)
            best_metrics = current_metrics

    return agent, rewards_history, best_metrics, current_strategy


def generate_strategy(agent, env, entity_types):
    """Generate strategy (price factors)"""
    agent.eval()
    strategy = {}
    with torch.no_grad():
        for entity in entity_types:
            strategy[entity] = {}
            for period in ['Valley', 'Flat', 'Peak']:
                hour = env._period_to_hour(period)
                state = env.get_state(entity, hour)
                action, _ = agent.get_action(state)
                action_val = np.clip(action.item(), 1.0 - 0.3, 1.0 + 0.3)  # Limit action range
                strategy[entity][period] = round(action_val, 4)
    return strategy


# ----------------------
# 6. Visualization Functions (Combined 4 charts in one canvas)
# ----------------------
def visualize_results(env, strategy, metrics, rewards_history, model_type):
    """Generate and save all four charts in a single 2x2 canvas with larger, bold text."""
    model_name_map = {
        'user': 'User Differentiation',
        'site': 'Site Differentiation',
        'site_user': 'Site-User Dual-Dimensional Differentiation'
    }
    model_name = model_name_map[model_type]
    save_dir = f"{output_dir}/{model_type}_differentiation_model"  # Ensure output_dir is defined externally

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create 2x2 subplot canvas
    plt.figure(figsize=(20, 16))

    # ----------------------
    # 1. Hourly load comparison (top left)
    # ----------------------
    plt.subplot(2, 2, 1)

    # Extend baseline and simulated load to 24 hours
    baseline_hourly_extended = env.baseline_metrics['hourly_load'].reindex(range(25), fill_value=0)
    baseline_hourly_extended.loc[24] = baseline_hourly_extended.loc[0]

    sim_hourly_extended = metrics['hourly_load'].reindex(range(25), fill_value=0)
    sim_hourly_extended.loc[24] = sim_hourly_extended.loc[0]

    plt.plot(baseline_hourly_extended.index, baseline_hourly_extended.values, 'b-', label='Baseline', linewidth=3)
    plt.plot(sim_hourly_extended.index, sim_hourly_extended.values, 'r--', label=f'RL Strategy', linewidth=3)

    # Mark periods
    plt.axvspan(0, 8, color='green', alpha=0.2, label='Valley')
    plt.axvspan(11, 13, color='green', alpha=0.2)
    plt.axvspan(22, 24, color='green', alpha=0.2)
    plt.axvspan(8, 11, color='orange', alpha=0.2, label='Flat')
    plt.axvspan(13, 19, color='orange', alpha=0.2)
    plt.axvspan(21, 22, color='orange', alpha=0.2)
    plt.axvspan(19, 21, color='red', alpha=0.2, label='Peak')

    plt.title('Hourly Load Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Hour', fontsize=14, fontweight='bold')
    plt.ylabel('Load (kWh)', fontsize=14, fontweight='bold')
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)

    # ----------------------
    # 2. Training reward curve (top right)
    # ----------------------
    plt.subplot(2, 2, 2)
    plt.plot(rewards_history, color='green', linewidth=2)
    plt.fill_between(
        range(len(rewards_history)),
        rewards_history,
        0,
        color='green',
        alpha=0.2
    )
    plt.title('Training Reward Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Average Reward', fontsize=14, fontweight='bold')
    plt.xlim(0, 200)
    plt.grid(alpha=0.3)
    plt.tick_params(axis='both', labelsize=14)

    # ----------------------
    # 3. Pricing strategy heatmap (bottom left)
    # ----------------------
    plt.subplot(2, 2, 3)
    periods = ['Valley', 'Flat', 'Peak']
    groups = list(strategy.keys())

    # Generate price matrix for heatmap
    price_matrix = []
    for group in groups:
        row = [env.base_prices[p] * strategy[group][p] for p in periods]
        price_matrix.append(row)

    # Green gradient heatmap
    sns.heatmap(
        price_matrix,
        annot=True,
        annot_kws={'size': 12},
        fmt=".4f",
        cmap="Greens",
        xticklabels=periods,
        yticklabels=groups,
        cbar_kws={'label': 'Price (CNY/kWh)'}
    )

    plt.title(f'Pricing Strategy by {model_name.split()[0]} Group', fontsize=16, fontweight='bold')
    plt.xlabel('Time Period', fontsize=14, fontweight='bold')
    plt.ylabel(f'{model_name.split()[0]} Group', fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.set_ylabel('Price (CNY/kWh)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # ----------------------
    # 4. Price adjustment factors (bottom right)
    # ----------------------
    plt.subplot(2, 2, 4)
    for group in strategy.keys():
        factors = [strategy[group][p] for p in periods]
        plt.plot(periods, factors, 'o-', label=group, linewidth=1.5)
    plt.axhline(y=1.0, color='k', linestyle='--', label='Baseline Factor')

    plt.title('Price Adjustment Factors', fontsize=16, fontweight='bold')
    plt.xlabel('Time Period', fontsize=14, fontweight='bold')
    plt.ylabel('Price Factor (x Baseline)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,
               fontsize=14)
    plt.grid(alpha=0.3)
    plt.tick_params(axis='y', labelsize=14)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    # Save high-resolution image
    plt.savefig(f"{save_dir}/Combined_Visualization_Charts.png", dpi=500, bbox_inches='tight')
    plt.close()


# ----------------------
# 7. Strategy Details Saving
# ----------------------
def save_strategy_details(strategy, env, metrics, strategy_type, save_dir):
    """Save strategy details and metrics comparison"""
    # Strategy details
    strategy_path = os.path.join(save_dir, f"{strategy_type}_pricing_strategies.txt")
    with open(strategy_path, 'w', encoding='utf-8') as f:
        f.write(f"{strategy_type.upper()} DIFFERENTIATED PRICING STRATEGY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Base Prices (CNY/kWh): Valley={0.3784}, Flat={0.9014}, Peak={1.2064}\n")
        f.write(f"Total Entities: {len(strategy)}\n")
        f.write("=" * 80 + "\n\n")

        for i, (entity, factors) in enumerate(strategy.items(), 1):
            f.write(f"Entity {i}: {entity}\n")
            for period in ['Valley', 'Flat', 'Peak']:
                base = env.base_prices[period]
                opt_price = base * factors[period]
                f.write(f"  {period}: {opt_price:.4f} CNY/kWh (Factor: {factors[period]:.4f})\n")
            f.write("-" * 60 + "\n")

    # Metrics comparison
    metrics_path = os.path.join(save_dir, f"{strategy_type}_metrics_comparison.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"{strategy_type.upper()} STRATEGY VS BASELINE\n")
        f.write("=" * 80 + "\n")
        f.write("Optimization Direction: ↓=Lower Better | ↑=Higher Better | ↓↓=More Negative Better\n")
        f.write("=" * 80 + "\n\n")

        # Peak-valley metrics
        f.write("1. Peak Shaving (↓):\n")
        base_peak = env.baseline_metrics['Peak-Valley Difference']
        new_peak = metrics['Peak-Valley Difference']
        imp = (1 - new_peak / base_peak) * 100
        f.write(f"   Peak-Valley Difference: {base_peak:.2f} → {new_peak:.2f} kWh ({imp:.1f}% Improvement)\n\n")

        # Utilization metrics
        f.write("2. Utilization (↓):\n")
        base_util = env.baseline_metrics['Utilization Standard Deviation']
        new_util = metrics['Utilization Standard Deviation']
        imp = (1 - new_util / base_util) * 100
        f.write(f"   Std Dev: {base_util:.4f} → {new_util:.4f} ({imp:.1f}% Improvement)\n\n")

        # Welfare metrics
        f.write("3. User Welfare (↓↓):\n")
        welfare = metrics['User Welfare Change']
        if welfare < 0:
            f.write(f"   ΔCS: {welfare:.2f} CNY (Improvement: {abs(welfare):.2f} CNY)\n")
        else:
            f.write(f"   ΔCS: {welfare:.2f} CNY (Loss: {welfare:.2f} CNY)\n")
        f.write(f"   Improvement Rate: {metrics['welfare_improvement_rate']:.2%}\n\n")

        # Revenue metrics
        f.write("4. Revenue (↑):\n")
        base_rev = env.baseline_metrics['Total Revenue']
        new_rev = metrics['Total Revenue']
        imp = (new_rev / base_rev - 1) * 100
        f.write(f"   Total Revenue: {base_rev:.0f} → {new_rev:.0f} CNY ({imp:.1f}% Increase)\n")


# ----------------------
# 8. Main Program
# ----------------------
def main():
    print("===== V2G Multi-dimensional Differentiated Pricing Optimization Program Started =====")
    print(
        "Supported strategies: Site differentiation | User differentiation | Site-user dual-dimensional differentiation")
    print(
        "Optimization objectives: Peak shaving and valley filling | Improve utilization | Enhance user welfare | Increase revenue\n")

    # 1. Data loading and preprocessing
    print("1. Loading and preprocessing data...")
    charging_df = load_and_preprocess_data()
    print(f"Data preprocessing completed, valid records: {len(charging_df)}")

    # 2. Price elasticity analysis
    print("\n2. Performing price elasticity analysis...")
    elasticity_results = analyze_price_elasticity(charging_df)

    # 3. Training three strategies
    strategies_data = {}  # Store (strategy, metrics, rewards) for all strategies

    # 3.1 Training site differentiation strategy
    print("\n3.1 Training site differentiation strategy:")
    site_env = TargetedV2GEnv(charging_df, elasticity_results, differentiation_type='site')
    site_model_path = os.path.join(output_dir, "site_differentiation_model/best_agent.pth")

    if os.path.exists(site_model_path):
        print("  Existing model found, loading directly...")
        site_agent = FixedPPOAgent(4, 1, 0.3).to(DEVICE)
        site_agent.load_state_dict(torch.load(site_model_path))
        site_agent.eval()
        site_strategy = generate_strategy(site_agent, site_env, site_env.site_types)
        site_metrics = site_env.simulate_strategy(site_strategy)
        site_rewards = []
    else:
        print("  No existing model, starting training (200 epochs)...")
        site_agent, site_rewards, site_metrics, site_strategy = stable_ppo_train(
            site_env, site_env.site_types, model_type='site', epochs=200
        )

    # Generate site strategy visualization results (including heatmap)
    visualize_results(site_env, site_strategy, site_metrics, site_rewards, 'site')
    strategies_data['site'] = (site_strategy, site_metrics, site_rewards)
    save_strategy_details(site_strategy, site_env, site_metrics, 'site',
                          os.path.join(output_dir, 'site_differentiation_model'))
    print("Site strategy training/loading completed")

    # 3.2 Training user differentiation strategy
    print("\n3.2 Training user differentiation strategy:")
    user_env = TargetedV2GEnv(charging_df, elasticity_results, differentiation_type='user')
    user_model_path = os.path.join(output_dir, "user_differentiation_model/best_agent.pth")

    if os.path.exists(user_model_path):
        print("  Existing model found, loading directly...")
        user_agent = FixedPPOAgent(4, 1, 0.3).to(DEVICE)
        user_agent.load_state_dict(torch.load(user_model_path))
        user_agent.eval()
        user_strategy = generate_strategy(user_agent, user_env, user_env.user_types)
        user_metrics = user_env.simulate_strategy(user_strategy)
        user_rewards = []
    else:
        print("No existing model, starting training (200 epochs)...")
        user_agent, user_rewards, user_metrics, user_strategy = stable_ppo_train(
            user_env, user_env.user_types, model_type='user', epochs=200
        )

    # Generate user strategy visualization results (including heatmap)
    visualize_results(user_env, user_strategy, user_metrics, user_rewards, 'user')
    strategies_data['user'] = (user_strategy, user_metrics, user_rewards)
    save_strategy_details(user_strategy, user_env, user_metrics, 'user',
                          os.path.join(output_dir, 'user_differentiation_model'))
    print("User strategy training/loading completed")

    # 3.3 Training dual-dimensional strategy
    print("\n3.3 Training site-user dual-dimensional strategy:")
    su_env = TargetedV2GEnv(charging_df, elasticity_results, differentiation_type='site_user')
    su_model_path = os.path.join(output_dir, "site_user_differentiation_model/best_agent.pth")

    if os.path.exists(su_model_path):
        print("  Existing model found, loading directly...")
        su_agent = FixedPPOAgent(4, 1, 0.3).to(DEVICE)
        su_agent.load_state_dict(torch.load(su_model_path))
        su_agent.eval()
        su_strategy = generate_strategy(su_agent, su_env, su_env.entity_types)
        su_metrics = su_env.simulate_strategy(su_strategy)
        su_rewards = []
    else:
        print("  No existing model, starting training (200 epochs)...")
        su_agent, su_rewards, su_metrics, su_strategy = stable_ppo_train(
            su_env, su_env.entity_types, model_type='site_user', epochs=200
        )

    # Generate dual-dimensional strategy visualization results (including heatmap)
    visualize_results(su_env, su_strategy, su_metrics, su_rewards, 'site_user')
    strategies_data['site_user'] = (su_strategy, su_metrics, su_rewards)
    save_strategy_details(su_strategy, su_env, su_metrics, 'site_user',
                          os.path.join(output_dir, 'site_user_differentiation_model'))
    print("Dual-dimensional strategy training/loading completed")

    # 4. Save all training records
    print("\n4. Saving training records...")
    with open(os.path.join(output_dir, 'All_Strategies_Training_Records.pkl'), 'wb') as f:
        pickle.dump({
            'baseline': site_env.baseline_metrics,
            'site': strategies_data['site'],
            'user': strategies_data['user'],
            'site_user': strategies_data['site_user'],
            'elasticity': elasticity_results
        }, f)

    print("\n===== Program execution completed =====")
    print(f"All results saved to directory: {output_dir}")
    print("Key files:")
    print("  - Visualization charts (including heatmaps) in each strategy directory")
    print("  - All_Strategies_Training_Records.pkl: Training records")
    print("  - In each strategy directory: Model files, strategy details, metrics comparison")


if __name__ == "__main__":
    main()