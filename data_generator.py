import pandas as pd
from datetime import datetime, timedelta
import random

class CampaignDataGenerator:
    def __init__(self):
        self.platforms = ['Google Ads', 'Facebook', 'Instagram', 'TikTok', 'LinkedIn', 'Twitter', 'YouTube', 'Snapchat']
        self.target_groups = ['18-24', '25-34', '35-44', '45-54', '55+', 'All Ages']
        self.campaign_types = ['Brand Awareness', 'Lead Generation', 'Sales', 'App Installs', 'Video Views', 'Traffic']
    
    def generate_campaigns(self, num_campaigns=12):
        """Generate diverse campaign data with realistic metrics"""
        campaigns = []
        
        campaign_names = [
            "Holiday Sale Blast", "Spring Collection Launch", "Summer Mega Sale",
            "Back to School Drive", "Black Friday Bonanza", "New Year Kickoff",
            "Valentine's Special", "Easter Promotion", "Mother's Day Campaign",
            "Father's Day Push", "Independence Day Sale", "Winter Collection Drop"
        ]
        
        for i in range(num_campaigns):
            # Base metrics with realistic correlations
            daily_budget = random.uniform(5000, 50000)
            impressions = random.uniform(10000, 100000)
            clicks = impressions * random.uniform(0.01, 0.05)  # 1-5% CTR
            conversions = clicks * random.uniform(0.02, 0.15)  # 2-15% conversion rate
            spend_per_day = daily_budget * random.uniform(0.8, 1.2)
            
            campaign = {
                'campaign_id': f"CMP_{i+1:03d}",
                'campaign_name': campaign_names[i] if i < len(campaign_names) else f"Campaign {i+1}",
                'platform': random.choice(self.platforms),
                'target_group': random.choice(self.target_groups),
                'campaign_type': random.choice(self.campaign_types),
                'daily_budget': daily_budget,
                'total_spend': spend_per_day * 30,  # 30 days
                'impressions': impressions * 30,
                'clicks': clicks * 30,
                'conversions': conversions * 30,
                'ctr': (clicks / impressions) * 100,
                'cpc': spend_per_day / clicks if clicks > 0 else 0,
                'conversion_rate': (conversions / clicks) * 100 if clicks > 0 else 0,
                'roi': ((conversions * 200) / spend_per_day - 1) * 100,  # Assume ₹200 per conversion
                'status': random.choice(['Active', 'Active', 'Active', 'Paused']),  # Mostly active
                'start_date': datetime.now() - timedelta(days=random.randint(30, 90)),
                'end_date': datetime.now() + timedelta(days=random.randint(7, 60))
            }
            campaigns.append(campaign)
        
        return pd.DataFrame(campaigns)
    
    def generate_time_series_data(self, campaigns_df, days_back=30, hourly_data=True):
        """Generate time series data with intentional anomalies"""
        all_data = []
        
        for _, campaign in campaigns_df.iterrows():
            # Generate base time series
            if hourly_data:
                time_points = pd.date_range(
                    start=datetime.now() - timedelta(days=days_back),
                    end=datetime.now(),
                    freq='h'
                )
            else:
                time_points = pd.date_range(
                    start=datetime.now() - timedelta(days=days_back),
                    end=datetime.now(),
                    freq='D'
                )
            
            for timestamp in time_points:
                # Base values with realistic patterns
                base_ctr = campaign['ctr']
                base_cpc = campaign['cpc']
                base_spend = campaign['daily_budget'] / (24 if hourly_data else 1)
                base_conversions = campaign['conversions'] / (days_back * (24 if hourly_data else 1))
                base_roi = campaign['roi']
                
                # Add time-based patterns (day of week, hour of day)
                day_of_week_factor = self._get_day_of_week_factor(timestamp.weekday())
                hour_factor = self._get_hour_factor(timestamp.hour) if hourly_data else 1.0
                
                # Apply patterns and add noise
                ctr = base_ctr * day_of_week_factor * hour_factor * random.uniform(0.8, 1.2)
                cpc = base_cpc * day_of_week_factor * hour_factor * random.uniform(0.9, 1.1)
                spend = base_spend * day_of_week_factor * hour_factor * random.uniform(0.85, 1.15)
                conversions = base_conversions * day_of_week_factor * hour_factor * random.uniform(0.7, 1.3)
                roi = base_roi * random.uniform(0.9, 1.1)
                
                # Inject strategic anomalies (5% chance per data point)
                if random.random() < 0.05:
                    anomaly_type = random.choice(['ctr_drop', 'spend_spike', 'cpc_spike', 'conversion_drop'])
                    
                    if anomaly_type == 'ctr_drop':
                        ctr *= random.uniform(0.4, 0.7)  # 30-60% drop
                    elif anomaly_type == 'spend_spike':
                        spend *= random.uniform(1.5, 2.5)  # 50-150% increase
                    elif anomaly_type == 'cpc_spike':
                        cpc *= random.uniform(1.4, 2.0)  # 40-100% increase
                    elif anomaly_type == 'conversion_drop':
                        conversions *= random.uniform(0.3, 0.6)  # 40-70% drop
                
                data_point = {
                    'campaign_id': campaign['campaign_id'],
                    'timestamp': timestamp,
                    'ctr': max(0, ctr),
                    'cpc': max(0, cpc),
                    'spend': max(0, spend),
                    'conversions': max(0, conversions),
                    'roi': roi,
                    'impressions': spend / (cpc / 1000) if cpc > 0 else 0,  # Derived metric
                    'clicks': (spend / cpc) if cpc > 0 else 0  # Derived metric
                }
                all_data.append(data_point)
        
        return pd.DataFrame(all_data)
    
    def _get_day_of_week_factor(self, weekday):
        """Get performance factor based on day of week"""
        # Monday=0, Sunday=6
        factors = [0.9, 1.0, 1.1, 1.0, 0.95, 0.8, 0.75]  # Lower on weekends
        return factors[weekday]
    
    def _get_hour_factor(self, hour):
        """Get performance factor based on hour of day"""
        # Peak hours: 9-11 AM and 2-4 PM and 7-9 PM
        if 9 <= hour <= 11 or 14 <= hour <= 16 or 19 <= hour <= 21:
            return random.uniform(1.2, 1.5)
        elif 6 <= hour <= 8 or 12 <= hour <= 13 or 17 <= hour <= 18:
            return random.uniform(1.0, 1.2)
        elif 22 <= hour <= 23 or 0 <= hour <= 5:
            return random.uniform(0.5, 0.8)
        else:
            return random.uniform(0.8, 1.0)
