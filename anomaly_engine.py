import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from scipy import stats

class AnomalyEngine:
    def __init__(self):
        self.rolling_window = 7  # 7-day rolling average
        self.min_periods = 3     # Minimum periods for rolling calculation
    
    def detect_anomalies(self, data, metric, z_threshold=2.0, deviation_threshold=30.0):
        """
        Detect anomalies in time series data using rolling averages and z-scores
        
        Args:
            data: DataFrame with 'timestamp' and metric columns
            metric: Column name to analyze
            z_threshold: Z-score threshold for anomaly detection
            deviation_threshold: Percentage deviation threshold
        
        Returns:
            List of anomaly dictionaries
        """
        if len(data) < self.min_periods:
            return []
        
        # Sort by timestamp
        data = data.sort_values('timestamp').copy()
        
        # Calculate rolling statistics
        data['rolling_mean'] = data[metric].rolling(
            window=self.rolling_window, 
            min_periods=self.min_periods,
            center=False
        ).mean()
        
        data['rolling_std'] = data[metric].rolling(
            window=self.rolling_window, 
            min_periods=self.min_periods,
            center=False
        ).std()
        
        # Calculate z-scores
        data['z_score'] = np.where(
            data['rolling_std'] > 0,
            (data[metric] - data['rolling_mean']) / data['rolling_std'],
            0
        )
        
        # Calculate percentage deviation
        data['deviation_percent'] = np.where(
            data['rolling_mean'] > 0,
            ((data[metric] - data['rolling_mean']) / data['rolling_mean']) * 100,
            0
        )
        
        # Identify anomalies
        anomalies = []
        
        for idx, row in data.iterrows():
            # Skip if we don't have enough data for rolling calculation
            if pd.isna(row['rolling_mean']) or pd.isna(row['z_score']):
                continue
            
            abs_z_score = abs(row['z_score'])
            abs_deviation = abs(row['deviation_percent'])
            
            # Check if point is anomalous
            is_anomaly = (abs_z_score >= z_threshold) and (abs_deviation >= deviation_threshold)
            
            if is_anomaly:
                # Classify severity
                severity = self._classify_severity(abs_z_score)
                
                # Determine anomaly direction
                direction = 'increase' if row['deviation_percent'] > 0 else 'decrease'
                
                # Calculate confidence
                confidence = min(95, 50 + (abs_z_score * 15))
                
                anomaly = {
                    'timestamp': row['timestamp'],
                    'metric': metric,
                    'observed_value': row[metric],
                    'expected_value': row['rolling_mean'],
                    'deviation_percent': row['deviation_percent'],
                    'z_score': row['z_score'],
                    'severity': severity,
                    'direction': direction,
                    'confidence': confidence,
                    'anomaly_type': self._get_anomaly_type(metric, direction, abs_deviation)
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _classify_severity(self, abs_z_score):
        """Classify anomaly severity based on z-score"""
        if abs_z_score >= 3.0:
            return 'L3'  # Critical
        elif abs_z_score >= 2.0:
            return 'L2'  # Moderate
        else:
            return 'L1'  # Mild
    
    def _get_anomaly_type(self, metric, direction, deviation):
        """Get human-readable anomaly type"""
        type_map = {
            'ctr': {
                'increase': 'CTR Spike',
                'decrease': 'CTR Drop'
            },
            'cpc': {
                'increase': 'CPC Spike', 
                'decrease': 'CPC Drop'
            },
            'spend': {
                'increase': 'Spend Surge',
                'decrease': 'Spend Drop'
            },
            'conversions': {
                'increase': 'Conversion Spike',
                'decrease': 'Conversion Drop'
            },
            'roi': {
                'increase': 'ROI Spike',
                'decrease': 'ROI Drop'
            }
        }
        
        return type_map.get(metric, {}).get(direction, f"{metric.upper()} {direction.title()}")
    
    def calculate_impact_score(self, anomaly, campaign_budget=10000, campaign_info=None):
        """Calculate enhanced business impact score with spend-at-risk calculations"""
        base_impact = abs(anomaly['deviation_percent']) / 100
        
        # Define negative impact directions (what constitutes a risk vs opportunity)
        negative_impact_directions = {
            'spend': ['increase'],        # Spend going up is bad
            'cpc': ['increase'],          # CPC going up is bad  
            'ctr': ['decrease'],          # CTR going down is bad
            'conversions': ['decrease'],  # Conversions going down is bad
            'roi': ['decrease']           # ROI going down is bad
        }
        
        # Determine if this is a negative impact (risk) or positive (opportunity)
        metric = anomaly['metric']
        direction = anomaly['direction']
        is_negative_impact = direction in negative_impact_directions.get(metric, [])
        
        # Weight by metric importance
        metric_weights = {
            'spend': 1.0,    # Direct cost impact
            'ctr': 0.8,      # Performance impact
            'cpc': 0.9,      # Cost efficiency impact
            'conversions': 1.0,  # Revenue impact
            'roi': 1.0       # Overall business impact
        }
        
        weight = metric_weights.get(metric, 0.7)
        base_impact_score = base_impact * weight * 100
        
        # Enhanced spend-at-risk calculations (only for negative impacts)
        spend_at_risk = 0
        revenue_at_risk = 0
        opportunity_value = 0
        
        if campaign_info:
            daily_budget = campaign_info.get('daily_budget', campaign_budget)
            current_roi = campaign_info.get('roi', 150) / 100  # Convert percentage to ratio
            avg_conversion_value = campaign_info.get('avg_conversion_value', 200)
        else:
            daily_budget = campaign_budget
            current_roi = 1.5
            avg_conversion_value = 200
        
        if is_negative_impact:
            # Calculate risk-based financial impact
            if metric == 'spend' and direction == 'increase':
                # Direct overspend risk
                daily_overspend = (anomaly['observed_value'] - anomaly['expected_value'])
                spend_at_risk = daily_overspend * 30  # Project over 30 days
                
            elif metric == 'cpc' and direction == 'increase':
                # Increased cost efficiency risk
                cpc_increase = abs(anomaly['deviation_percent']) / 100
                spend_at_risk = daily_budget * cpc_increase * 7  # Weekly projection
                
            elif metric == 'ctr' and direction == 'decrease':
                # Reduced efficiency leads to higher costs for same conversions
                efficiency_loss = abs(anomaly['deviation_percent']) / 100
                spend_at_risk = daily_budget * efficiency_loss * 7
                
            elif metric == 'conversions' and direction == 'decrease':
                # Lost revenue opportunity
                conversion_loss = abs(anomaly['observed_value'] - anomaly['expected_value'])
                revenue_at_risk = conversion_loss * avg_conversion_value * 7  # Weekly projection
                
            elif metric == 'roi' and direction == 'decrease':
                # Overall campaign efficiency degradation
                roi_loss = abs(anomaly['deviation_percent']) / 100
                spend_at_risk = daily_budget * roi_loss * 14  # Bi-weekly projection
        else:
            # Calculate opportunity value for positive impacts
            if metric == 'ctr' and direction == 'increase':
                # Better click-through rate = more conversions for same spend
                efficiency_gain = abs(anomaly['deviation_percent']) / 100
                opportunity_value = daily_budget * efficiency_gain * current_roi * 7
                
            elif metric == 'cpc' and direction == 'decrease':
                # Lower cost per click = savings
                cpc_savings = abs(anomaly['deviation_percent']) / 100
                opportunity_value = daily_budget * cpc_savings * 7
                
            elif metric == 'conversions' and direction == 'increase':
                # More conversions = more revenue
                conversion_gain = abs(anomaly['observed_value'] - anomaly['expected_value'])
                opportunity_value = conversion_gain * avg_conversion_value * 7
                
            elif metric == 'roi' and direction == 'increase':
                # Better ROI = more profit
                roi_gain = abs(anomaly['deviation_percent']) / 100
                opportunity_value = daily_budget * roi_gain * 14
                
            elif metric == 'spend' and direction == 'decrease':
                # Lower spend (could be good if maintaining performance)
                spend_reduction = abs(anomaly['observed_value'] - anomaly['expected_value'])
                opportunity_value = spend_reduction * 30  # Monthly savings
        
        # Calculate financial impact based on type
        if is_negative_impact:
            if metric == 'spend':
                financial_impact = spend_at_risk
            elif metric == 'conversions':
                financial_impact = revenue_at_risk
            else:
                financial_impact = spend_at_risk + (revenue_at_risk * 0.3)  # Weighted combination
        else:
            financial_impact = opportunity_value
        
        # Apply severity multiplier and determine category
        severity_multiplier = {
            'L1': 1.0,
            'L2': 1.5, 
            'L3': 2.5
        }.get(anomaly['severity'], 1.0)
        
        adjusted_impact_score = min(100, base_impact_score * severity_multiplier)
        
        # Categorize based on impact type
        if is_negative_impact:
            category = self._categorize_risk_level(adjusted_impact_score)
            category_type = 'risk'
        else:
            category = self._categorize_opportunity_level(adjusted_impact_score)
            category_type = 'opportunity'
        
        return {
            'impact_score': adjusted_impact_score,
            'financial_impact': financial_impact,
            'spend_at_risk': spend_at_risk if is_negative_impact else 0,
            'revenue_at_risk': revenue_at_risk if is_negative_impact else 0,
            'opportunity_value': opportunity_value if not is_negative_impact else 0,
            'risk_category': category if is_negative_impact else None,
            'opportunity_category': category if not is_negative_impact else None,
            'category_type': category_type,
            'is_negative_impact': is_negative_impact,
            'projected_days': self._get_projection_period(anomaly['metric']),
            'confidence_level': min(95, 50 + (abs(anomaly['z_score']) * 15))
        }
    
    def _categorize_risk_level(self, impact_score):
        """Categorize risk level based on impact score"""
        if impact_score >= 80:
            return 'Critical Risk'
        elif impact_score >= 50:
            return 'High Risk'
        elif impact_score >= 25:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    def _categorize_opportunity_level(self, impact_score):
        """Categorize opportunity level based on impact score"""
        if impact_score >= 80:
            return 'Major Opportunity'
        elif impact_score >= 50:
            return 'High Opportunity'
        elif impact_score >= 25:
            return 'Medium Opportunity'
        else:
            return 'Minor Opportunity'
    
    def _get_projection_period(self, metric):
        """Get projection period in days for different metrics"""
        projection_periods = {
            'spend': 30,      # Monthly projection for spend anomalies
            'cpc': 7,         # Weekly projection for cost changes
            'ctr': 7,         # Weekly projection for performance changes
            'conversions': 7, # Weekly projection for conversion changes
            'roi': 14         # Bi-weekly projection for overall performance
        }
        return projection_periods.get(metric, 7)
    
    def analyze_historical_patterns(self, all_anomalies_list, lookback_days=30):
        """Analyze historical anomaly patterns and trends"""
        if not all_anomalies_list:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_anomalies_list)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Pattern analysis results
        patterns = {
            'recurring_patterns': [],
            'seasonal_trends': {},
            'metric_correlations': {},
            'frequency_analysis': {},
            'severity_trends': {},
            'time_based_patterns': {}
        }
        
        # 1. Recurring Pattern Detection
        for metric in df['metric'].unique():
            metric_anomalies = df[df['metric'] == metric].copy()
            if len(metric_anomalies) >= 3:
                # Check for recurring time patterns (daily, weekly)
                metric_anomalies['hour'] = metric_anomalies['timestamp'].dt.hour
                metric_anomalies['day_of_week'] = metric_anomalies['timestamp'].dt.dayofweek
                
                # Find most common hours/days for anomalies
                common_hours = metric_anomalies['hour'].value_counts().head(3)
                common_days = metric_anomalies['day_of_week'].value_counts().head(2)
                
                if len(common_hours) > 0 and common_hours.iloc[0] >= 2:  # At least 2 anomalies at same hour
                    patterns['recurring_patterns'].append({
                        'metric': metric,
                        'pattern_type': 'hourly',
                        'common_times': common_hours.index.tolist(),
                        'frequency': common_hours.values.tolist()
                    })
                
                if len(common_days) > 0 and common_days.iloc[0] >= 2:  # At least 2 anomalies on same day
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    patterns['recurring_patterns'].append({
                        'metric': metric,
                        'pattern_type': 'daily',
                        'common_days': [day_names[i] for i in common_days.index.tolist()],
                        'frequency': common_days.values.tolist()
                    })
        
        # 2. Frequency Analysis
        for metric in df['metric'].unique():
            metric_count = len(df[df['metric'] == metric])
            metric_severity = df[df['metric'] == metric]['severity'].value_counts()
            patterns['frequency_analysis'][metric] = {
                'total_anomalies': metric_count,
                'anomaly_rate': metric_count / lookback_days if lookback_days > 0 else 0,
                'avg_severity': metric_severity.to_dict() if len(metric_severity) > 0 else {}
            }
        
        # 3. Severity Trend Analysis
        df_recent = df.sort_values('timestamp').tail(14)  # Last 2 weeks
        recent_severity = df_recent['severity'].value_counts()
        all_severity = df['severity'].value_counts()
        
        # Safe division check
        escalation_check = False
        if len(df) > 0 and len(df_recent) > 0:
            recent_l3_rate = recent_severity.get('L3', 0) / len(df_recent)
            historical_l3_rate = all_severity.get('L3', 0) / len(df)
            escalation_check = recent_l3_rate > historical_l3_rate
        
        patterns['severity_trends'] = {
            'recent_trend': recent_severity.to_dict(),
            'historical_avg': all_severity.to_dict(),
            'is_escalating': escalation_check
        }
        
        # 4. Time-based Pattern Analysis
        if len(df) >= 7:
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date', as_index=False).size()
            count_values = daily_counts['size'] if 'size' in daily_counts.columns else daily_counts[0]
            
            patterns['time_based_patterns'] = {
                'daily_average': count_values.mean(),
                'peak_days': dict(zip(daily_counts['date'].astype(str).tail(3), count_values.tail(3))),
                'quiet_periods': 0,  # Simplified for now
                'trend_direction': 'increasing' if count_values.tail(3).mean() > count_values.head(3).mean() else 'decreasing'
            }
        
        # 5. Cross-metric Correlations (simplified)
        if len(df['metric'].unique()) > 1:
            metric_pairs = {}
            for metric1 in df['metric'].unique():
                for metric2 in df['metric'].unique():
                    if metric1 != metric2:
                        # Count co-occurring anomalies within same hour
                        df1_hours = set(df[df['metric'] == metric1]['timestamp'].dt.floor('h'))
                        df2_hours = set(df[df['metric'] == metric2]['timestamp'].dt.floor('h'))
                        overlap = len(df1_hours & df2_hours)
                        
                        if overlap >= 2:  # At least 2 co-occurrences
                            pair_key = f"{metric1}_{metric2}"
                            metric_pairs[pair_key] = {
                                'co_occurrences': overlap,
                                'correlation_strength': overlap / min(len(df1_hours), len(df2_hours))
                            }
            
            patterns['metric_correlations'] = metric_pairs
        
        return patterns
    
    def detect_correlated_anomalies(self, anomalies_data: List[Dict], time_window_hours: int = 2) -> Dict:
        """Detect correlated anomalies across campaigns and metrics within time windows."""
        if len(anomalies_data) < 2:
            return {'correlated_groups': [], 'campaign_failures': [], 'metric_cascades': []}
        
        df = pd.DataFrame(anomalies_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        correlated_groups = []
        campaign_failures = []
        metric_cascades = []
        
        # 1. Find time-based correlation groups (anomalies within time windows)
        df_sorted = df.sort_values('timestamp')
        for i, anomaly in df_sorted.iterrows():
            time_window_start = anomaly['timestamp'] - pd.Timedelta(hours=time_window_hours)
            time_window_end = anomaly['timestamp'] + pd.Timedelta(hours=time_window_hours)
            
            # Find all anomalies within the time window
            window_anomalies = df_sorted[
                (df_sorted['timestamp'] >= time_window_start) & 
                (df_sorted['timestamp'] <= time_window_end) &
                (df_sorted.index != i)  # Exclude the current anomaly
            ]
            
            if len(window_anomalies) >= 2:  # At least 2 other anomalies in window
                group = {
                    'primary_anomaly': anomaly.to_dict(),
                    'correlated_anomalies': window_anomalies.to_dict('records'),
                    'time_window_hours': time_window_hours,
                    'correlation_strength': len(window_anomalies) / len(df),
                    'affected_campaigns': list(window_anomalies['campaign_name'].unique()),
                    'affected_metrics': list(window_anomalies['metric'].unique())
                }
                
                # Avoid duplicates by checking if similar group exists
                is_duplicate = any(
                    abs((pd.to_datetime(g['primary_anomaly']['timestamp']) - anomaly['timestamp']).total_seconds()) < 3600
                    for g in correlated_groups
                )
                
                if not is_duplicate:
                    correlated_groups.append(group)
        
        # 2. Detect campaign-wide failures (multiple metrics failing in same campaign)
        campaign_metric_counts = df.groupby('campaign_name')['metric'].nunique()
        for campaign, metric_count in campaign_metric_counts.items():
            if metric_count >= 3:  # At least 3 different metrics with anomalies
                campaign_anomalies = df[df['campaign_name'] == campaign]
                
                # Check if anomalies are clustered in time (within 6 hours)
                time_span = (campaign_anomalies['timestamp'].max() - campaign_anomalies['timestamp'].min()).total_seconds() / 3600
                
                if time_span <= 6:  # Campaign failure if all issues within 6 hours
                    failure_data = {
                        'campaign_name': campaign,
                        'affected_metrics': list(campaign_anomalies['metric'].unique()),
                        'anomaly_count': len(campaign_anomalies),
                        'time_span_hours': time_span,
                        'severity_distribution': campaign_anomalies['severity'].value_counts().to_dict(),
                        'total_spend_at_risk': campaign_anomalies['spend_at_risk'].sum(),
                        'failure_period': {
                            'start': campaign_anomalies['timestamp'].min().isoformat(),
                            'end': campaign_anomalies['timestamp'].max().isoformat()
                        }
                    }
                    campaign_failures.append(failure_data)
        
        # 3. Detect metric cascades (one metric anomaly leading to others)
        for metric1 in df['metric'].unique():
            for metric2 in df['metric'].unique():
                if metric1 != metric2:
                    # Check for temporal ordering (metric1 anomalies before metric2)
                    m1_anomalies = df[df['metric'] == metric1].sort_values('timestamp')
                    m2_anomalies = df[df['metric'] == metric2].sort_values('timestamp')
                    
                    cascade_count = 0
                    for _, m1_anom in m1_anomalies.iterrows():
                        # Find m2 anomalies within 1-4 hours after m1 anomaly
                        subsequent_m2 = m2_anomalies[
                            (m2_anomalies['timestamp'] > m1_anom['timestamp']) &
                            (m2_anomalies['timestamp'] <= m1_anom['timestamp'] + pd.Timedelta(hours=4))
                        ]
                        cascade_count += len(subsequent_m2)
                    
                    if cascade_count >= 2:
                        cascade_data = {
                            'trigger_metric': metric1,
                            'affected_metric': metric2,
                            'cascade_occurrences': cascade_count,
                            'cascade_strength': cascade_count / max(len(m1_anomalies), 1),
                            'typical_delay_hours': '1-4 hours'
                        }
                        metric_cascades.append(cascade_data)
        
        return {
            'correlated_groups': correlated_groups,
            'campaign_failures': campaign_failures,
            'metric_cascades': metric_cascades
        }
    
    def get_pattern_insights(self, patterns):
        """Generate actionable insights from pattern analysis"""
        insights = []
        
        # Recurring pattern insights
        for pattern in patterns.get('recurring_patterns', []):
            if pattern['pattern_type'] == 'hourly':
                hours_str = ', '.join(map(str, pattern['common_times'][:2]))
                insights.append(f"⏰ {pattern['metric'].upper()} anomalies frequently occur at hours {hours_str}")
            elif pattern['pattern_type'] == 'daily':
                days_str = ', '.join(pattern['common_days'][:2])
                insights.append(f"📅 {pattern['metric'].upper()} issues commonly happen on {days_str}")
        
        # Frequency insights
        freq_analysis = patterns.get('frequency_analysis', {})
        high_freq_metrics = [(m, data['anomaly_rate']) for m, data in freq_analysis.items() if data['anomaly_rate'] > 0.5]
        if high_freq_metrics:
            top_metric = max(high_freq_metrics, key=lambda x: x[1])
            insights.append(f"🚨 {top_metric[0].upper()} shows high instability ({top_metric[1]:.1f} anomalies/day)")
        
        # Severity trend insights
        severity_trends = patterns.get('severity_trends', {})
        if severity_trends.get('is_escalating', False):
            insights.append("📈 Critical anomalies (L3) are increasing in recent weeks")
        
        # Correlation insights
        correlations = patterns.get('metric_correlations', {})
        strong_correlations = [(pair, data['correlation_strength']) for pair, data in correlations.items() if data['correlation_strength'] > 0.6]
        if strong_correlations:
            top_correlation = max(strong_correlations, key=lambda x: x[1])
            metrics = top_correlation[0].split('_')
            insights.append(f"🔗 {metrics[0].upper()} and {metrics[1].upper()} anomalies often occur together")
        
        # Time-based insights
        time_patterns = patterns.get('time_based_patterns', {})
        if time_patterns.get('trend_direction') == 'increasing':
            insights.append("📊 Overall anomaly frequency is increasing over time")
        elif time_patterns.get('daily_average', 0) > 2:
            insights.append(f"⚡ High anomaly activity: {time_patterns['daily_average']:.1f} anomalies/day on average")
        
        return insights[:5]  # Return top 5 insights
    
    def get_recommendations(self, anomaly):
        """Generate actionable recommendations based on anomaly type"""
        recommendations = []
        
        metric = anomaly['metric']
        direction = anomaly['direction']
        severity = anomaly['severity']
        
        if metric == 'ctr' and direction == 'decrease':
            recommendations.extend([
                "Review ad creative performance and refresh underperforming ads",
                "Check audience targeting relevance",
                "Analyze competitor activity for increased competition",
                "Consider A/B testing new ad formats"
            ])
        
        elif metric == 'cpc' and direction == 'increase':
            recommendations.extend([
                "Review keyword bid strategy",
                "Analyze Quality Score factors",
                "Check for increased competition in auction",
                "Consider expanding to less competitive keywords"
            ])
        
        elif metric == 'spend' and direction == 'increase':
            recommendations.extend([
                "Check automated bidding settings",
                "Review daily budget caps",
                "Analyze traffic quality and conversion rates",
                "Consider pausing low-performing ad groups"
            ])
        
        elif metric == 'conversions' and direction == 'decrease':
            recommendations.extend([
                "Check landing page performance and load times",
                "Review conversion tracking implementation",
                "Analyze user experience and checkout process",
                "Consider seasonal or external factors"
            ])
        
        # Add severity-specific recommendations
        if severity == 'L3':
            recommendations.insert(0, "URGENT: Consider pausing campaign immediately")
            recommendations.append("Schedule emergency stakeholder meeting")
        
        return recommendations[:4]  # Return top 4 recommendations
