import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from data_generator import CampaignDataGenerator
from anomaly_engine import AnomalyEngine
from ui_components import UIComponents


# Page config
st.set_page_config(
    page_title="Campaign Guardian - Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_data
def load_data():
    generator = CampaignDataGenerator()
    campaigns_data = generator.generate_campaigns()
    time_series_data = generator.generate_time_series_data(campaigns_data)
    return campaigns_data, time_series_data

@st.cache_data
def initialize_campaign_settings(campaigns_data):
    """Initialize default alert settings for all campaigns"""
    settings = {}
    for _, campaign in campaigns_data.iterrows():
        campaign_id = campaign['campaign_id']
        settings[campaign_id] = {
            'ctr_threshold': 30,
            'cpc_threshold': 30, 
            'spend_threshold': 30,
            'conversions_threshold': 30,
            'roi_threshold': 30,
            'sensitivity_level': 'L2 - Moderate (z > 2.0)',
            'z_threshold': 2.0
        }
    return settings

def main():
    # Header
    st.title("🛡️ Campaign Guardian")
    st.markdown("**Advanced Ad Campaign Anomaly Detection System**")
    st.markdown("---")
    
    # Load data
    campaigns_data, time_series_data = load_data()
    
    # Initialize campaign settings (stored in session state for persistence)
    if 'campaign_settings' not in st.session_state:
        st.session_state.campaign_settings = initialize_campaign_settings(campaigns_data)
    
    # Initialize engines
    anomaly_engine = AnomalyEngine()
    ui_components = UIComponents()
    
    # Sidebar
    st.sidebar.header("🎛️ Control Panel")
    
    # Alert Configuration Mode
    config_mode = st.sidebar.radio(
        "Configuration Mode:",
        ["Quick Settings", "Advanced Campaign Config"],
        index=0
    )
    
    # Campaign selector
    campaign_names = campaigns_data['campaign_name'].tolist()
    selected_campaign = st.sidebar.selectbox(
        "Select Campaign:",
        options=campaign_names,
        index=0
    )
    
    # Get selected campaign data
    campaign_info = campaigns_data[campaigns_data['campaign_name'] == selected_campaign].iloc[0]
    campaign_ts_data = pd.DataFrame(time_series_data[time_series_data['campaign_id'] == campaign_info['campaign_id']])
    
    # Validate data structure
    if campaign_ts_data.empty:
        st.error("No data available for the selected campaign.")
        return
    
    # Metric filter
    available_metrics = ['ctr', 'cpc', 'spend', 'conversions', 'roi']
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Monitor:",
        options=available_metrics,
        default=['ctr', 'cpc', 'spend']
    )
    
    # Time range selector
    st.sidebar.markdown("---")
    time_range_options = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
        "All data": None
    }
    
    selected_time_range = st.sidebar.selectbox(
        "📅 Time Range:",
        options=list(time_range_options.keys()),
        index=1  # Default to 14 days
    )
    
    time_range_days = time_range_options[selected_time_range]
    
    # Standardize timestamp format
    if 'timestamp' in campaign_ts_data.columns:
        campaign_ts_data['timestamp'] = pd.to_datetime(campaign_ts_data['timestamp'])
    
    # Apply time range filtering if specified
    original_count = len(campaign_ts_data)
    if time_range_days is not None:
        # Calculate cutoff date
        max_date = campaign_ts_data['timestamp'].max()
        cutoff_date = max_date - pd.Timedelta(days=time_range_days)
        
        # Filter data
        campaign_ts_data = campaign_ts_data[campaign_ts_data['timestamp'] >= cutoff_date]
        filtered_count = len(campaign_ts_data)
        
        # Show filtering info in sidebar
        st.sidebar.info(f"📊 Showing {filtered_count} of {original_count} data points")
    else:
        # Show info for all data too
        st.sidebar.info(f"📊 Showing all {original_count} data points")
    
    # Final validation after filtering
    if campaign_ts_data.empty:
        st.error(f"No data available for the selected campaign in the {selected_time_range.lower()}.")
        return
    
    # Sensitivity selector
    sensitivity_levels = {
        'L1 - Conservative (z > 1.5)': {'threshold': 1.5, 'deviation': 20},
        'L2 - Moderate (z > 2.0)': {'threshold': 2.0, 'deviation': 30},
        'L3 - Aggressive (z > 3.0)': {'threshold': 3.0, 'deviation': 40}
    }
    
    selected_sensitivity = st.sidebar.selectbox(
        "Anomaly Sensitivity:",
        options=list(sensitivity_levels.keys()),
        index=1
    )
    
    sensitivity_config = sensitivity_levels[selected_sensitivity]
    
    # Get current campaign settings
    campaign_id = campaign_info['campaign_id']
    current_settings = st.session_state.campaign_settings.get(campaign_id, {})
    
    if config_mode == "Quick Settings":
        # Quick global thresholds
        st.sidebar.subheader("📊 Quick Thresholds")
        custom_thresholds = {}
        for metric in selected_metrics:
            custom_thresholds[metric] = st.sidebar.slider(
                f"{metric.upper()} Deviation %:",
                min_value=10,
                max_value=100,
                value=sensitivity_config['deviation'],
                step=5
            )
    else:
        # Advanced campaign-specific configuration
        st.sidebar.subheader(f"⚙️ {selected_campaign} Settings")
        
        # Update campaign-specific settings
        updated_settings = {}
        updated_settings['sensitivity_level'] = st.sidebar.selectbox(
            "Campaign Sensitivity:",
            options=list(sensitivity_levels.keys()),
            index=list(sensitivity_levels.keys()).index(current_settings.get('sensitivity_level', 'L2 - Moderate (z > 2.0)')),
            key=f"sensitivity_{campaign_id}"
        )
        
        sensitivity_config = sensitivity_levels[updated_settings['sensitivity_level']]
        updated_settings['z_threshold'] = sensitivity_config['threshold']
        
        # Metric-specific thresholds for this campaign
        custom_thresholds = {}
        for metric in selected_metrics:
            threshold_key = f"{metric}_threshold"
            updated_settings[threshold_key] = st.sidebar.slider(
                f"{metric.upper()} Threshold %:",
                min_value=10,
                max_value=100,
                value=current_settings.get(threshold_key, 30),
                step=5,
                key=f"{metric}_threshold_{campaign_id}"
            )
            custom_thresholds[metric] = updated_settings[threshold_key]
        
        # Merge updated settings with existing ones to preserve other thresholds
        existing_settings = st.session_state.campaign_settings[campaign_id].copy()
        existing_settings.update(updated_settings)
        st.session_state.campaign_settings[campaign_id] = existing_settings
        
        # Alert setup confirmation
        with st.sidebar.expander("🔧 Alert Setup Summary", expanded=False):
            st.write(f"**Campaign:** {selected_campaign}")
            st.write(f"**Sensitivity:** {updated_settings['sensitivity_level'].split(' - ')[0]}")
            st.write(f"**Z-Score:** {updated_settings['z_threshold']}")
            st.write("**Thresholds:**")
            for metric, threshold in custom_thresholds.items():
                st.write(f"• {metric.upper()}: {threshold}%")
            
            if st.button("💾 Save Campaign Settings", key=f"save_{campaign_id}"):
                st.success(f"✅ Settings saved for {selected_campaign}!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 Campaign: {selected_campaign}")
        
        # Time range indicator
        if time_range_days is not None:
            date_range_str = f"Showing data from {selected_time_range.lower()}"
        else:
            date_range_str = "Showing all available data"
        
        st.caption(f"📅 {date_range_str} • {len(campaign_ts_data)} data points")
        
        # Campaign info cards
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Platform", campaign_info['platform'])
        with info_cols[1]:
            st.metric("Target Group", campaign_info['target_group'])
        with info_cols[2]:
            st.metric("Total Spend", f"₹{campaign_info['total_spend']:,.0f}")
        with info_cols[3]:
            st.metric("Current ROI", f"{campaign_info['roi']:.1f}%")
    
    with col2:
        st.subheader("🎯 Alert Configuration")
        if config_mode == "Quick Settings":
            st.info(f"**Mode:** Quick Settings")
            st.write(f"**Sensitivity:** {selected_sensitivity.split(' - ')[0]}")
            st.write(f"**Z-Score Threshold:** {sensitivity_config['threshold']}")
            st.write(f"**Deviation Thresholds:** Global")
        else:
            st.info(f"**Mode:** Campaign-Specific")
            st.write(f"**Campaign:** {selected_campaign}")
            campaign_settings = st.session_state.campaign_settings.get(campaign_id, {})
            st.write(f"**Sensitivity:** {campaign_settings.get('sensitivity_level', 'L2').split(' - ')[0]}")
            st.write(f"**Z-Score:** {campaign_settings.get('z_threshold', 2.0)}")
            
            # Show active thresholds
            st.write("**Active Thresholds:**")
            for metric in selected_metrics:
                threshold = campaign_settings.get(f"{metric}_threshold", 30)
                st.write(f"• {metric.upper()}: {threshold}%")
        
        # Configuration recommendations
        with st.expander("💡 Configuration Tips"):
            st.write("**Sensitivity Levels:**")
            st.write("• L1: Conservative - fewer false alarms")
            st.write("• L2: Balanced - recommended for most campaigns") 
            st.write("• L3: Aggressive - catches subtle changes")
            st.write("\n**Threshold Guidelines:**")
            st.write("• CTR: 20-40% for established campaigns")
            st.write("• CPC: 25-50% based on market volatility")
            st.write("• Spend: 30-60% for budget protection")
    
    # Process anomalies
    anomalies_data = []
    for metric in selected_metrics:
        metric_data = campaign_ts_data[['timestamp', metric]].copy()
        metric_data = metric_data.sort_values('timestamp')
        
        # Get the appropriate z_threshold
        if config_mode == "Advanced Campaign Config":
            campaign_settings = st.session_state.campaign_settings.get(campaign_id, {})
            z_threshold = campaign_settings.get('z_threshold', 2.0)
        else:
            z_threshold = sensitivity_config['threshold']
        
        anomalies = anomaly_engine.detect_anomalies(
            metric_data, 
            metric, 
            z_threshold=z_threshold,
            deviation_threshold=custom_thresholds[metric]
        )
        
        for anomaly in anomalies:
            anomaly['campaign_name'] = selected_campaign
            anomaly['campaign_id'] = campaign_info['campaign_id']
            
            # Calculate enhanced impact score with spend-at-risk
            impact_data = anomaly_engine.calculate_impact_score(
                anomaly, 
                campaign_budget=campaign_info.get('daily_budget', 10000),
                campaign_info=campaign_info.to_dict()
            )
            
            # Add impact data to anomaly
            anomaly.update(impact_data)
            
            anomalies_data.append(anomaly)
    
    # Display anomalies table
    st.subheader("🚨 Detected Anomalies")
    
    if anomalies_data:
        anomalies_df = pd.DataFrame(anomalies_data)
        
        # Add severity badges (plain text for dataframe compatibility)
        def get_severity_badge(severity):
            badges = {
                'L1': '⚠️ L1 - Mild',
                'L2': '🔥 L2 - Moderate', 
                'L3': '🚨 L3 - Critical'
            }
            return badges.get(severity, '⚪ Unknown')
        
        def get_severity_html_badge(severity):
            """HTML version for legend display"""
            colors = {
                'L1': ('⚠️', '#FFA500', 'Mild'),
                'L2': ('🔥', '#FF6B35', 'Moderate'), 
                'L3': ('🚨', '#DC143C', 'Critical')
            }
            icon, color, label = colors.get(severity, ('⚪', '#808080', 'Unknown'))
            return f'<span style="background: {color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-right: 5px;">{icon} {severity} - {label}</span>'
        
        anomalies_df['severity_badge'] = anomalies_df['severity'].apply(get_severity_badge)
        
        # Display table with enhanced impact information
        display_columns = ['timestamp', 'metric', 'expected_value', 'observed_value', 
                          'deviation_percent', 'severity_badge', 'z_score', 'category_type']
        
        # Add appropriate category and financial columns based on type
        formatted_df = anomalies_df[display_columns].copy()
        
        # Create a unified category column
        def get_unified_category(row):
            if row.get('risk_category'):
                return row['risk_category']
            elif row.get('opportunity_category'):
                return row['opportunity_category']
            else:
                return 'Unknown'
        
        # Create a unified financial impact column  
        def get_financial_impact_display(row):
            if row.get('spend_at_risk', 0) > 0:
                return f"₹{row['spend_at_risk']:,.0f} spend risk"
            elif row.get('revenue_at_risk', 0) > 0:
                return f"₹{row['revenue_at_risk']:,.0f} revenue risk"
            elif row.get('opportunity_value', 0) > 0:
                return f"₹{row['opportunity_value']:,.0f} opportunity"
            else:
                return "-"
        
        formatted_df['Impact Category'] = anomalies_df.apply(get_unified_category, axis=1)
        formatted_df['Financial Impact'] = anomalies_df.apply(get_financial_impact_display, axis=1)
        
        formatted_df.columns = ['Timestamp', 'Metric', 'Expected', 'Observed', 
                               'Deviation %', 'Severity', 'Z-Score', 'Type', 'Impact Category', 'Financial Impact']
        
        # Format numeric columns
        for col in ['Expected', 'Observed']:
            formatted_df[col] = formatted_df[col].round(2)
        formatted_df['Deviation %'] = formatted_df['Deviation %'].round(1)
        formatted_df['Z-Score'] = formatted_df['Z-Score'].round(2)
        
        # Display enhanced anomalies table with color coding
        st.markdown("**🚨 Anomaly Detection Results**")
        
        # Create styled dataframe with conditional formatting
        def highlight_severity(row):
            if 'L3' in str(row['Severity']):
                return ['background-color: #ffebee'] * len(row)
            elif 'L2' in str(row['Severity']):
                return ['background-color: #fff3e0'] * len(row)
            elif 'L1' in str(row['Severity']):
                return ['background-color: #f9fbe7'] * len(row)
            return [''] * len(row)
        
        styled_df = formatted_df.style.apply(highlight_severity, axis=1)
        st.dataframe(styled_df, height=300)
        
        # Show severity badge legend with HTML badges
        st.markdown("**Severity Levels:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(get_severity_html_badge('L1'), unsafe_allow_html=True)
        with col2:
            st.markdown(get_severity_html_badge('L2'), unsafe_allow_html=True) 
        with col3:
            st.markdown(get_severity_html_badge('L3'), unsafe_allow_html=True)
        
        # Enhanced Risk Analysis Dashboard
        st.markdown("---")
        st.subheader("💰 Risk Analysis & Opportunity Insights")
        
        # Separate risks and opportunities
        risks = [a for a in anomalies_data if a.get('is_negative_impact', True)]
        opportunities = [a for a in anomalies_data if not a.get('is_negative_impact', True)]
        
        # Calculate aggregated metrics (only for actual risks)
        total_spend_at_risk = sum(a.get('spend_at_risk', 0) for a in risks)
        total_revenue_at_risk = sum(a.get('revenue_at_risk', 0) for a in risks)
        total_opportunity_value = sum(a.get('opportunity_value', 0) for a in opportunities)
        
        # Risk level distribution (only for risks)
        risk_distribution = {}
        for risk in risks:
            risk_cat = risk.get('risk_category', 'Unknown')
            if risk_cat != 'Unknown':  # Only count valid risk categories
                risk_distribution[risk_cat] = risk_distribution.get(risk_cat, 0) + 1
        
        # Opportunity distribution
        opportunity_distribution = {}
        for opp in opportunities:
            opp_cat = opp.get('opportunity_category', 'Unknown') 
            if opp_cat != 'Unknown':  # Only count valid opportunity categories
                opportunity_distribution[opp_cat] = opportunity_distribution.get(opp_cat, 0) + 1
        
        # Financial impact summary
        risk_cols = st.columns(4)
        
        with risk_cols[0]:
            st.metric(
                "💸 Total Spend at Risk",
                f"₹{total_spend_at_risk:,.0f}",
                help="Projected additional spend if risk anomalies persist"
            )
        
        with risk_cols[1]:
            st.metric(
                "📉 Revenue at Risk", 
                f"₹{total_revenue_at_risk:,.0f}",
                help="Projected revenue loss from performance degradation"
            )
        
        with risk_cols[2]:
            st.metric(
                "💰 Total Opportunity Value",
                f"₹{total_opportunity_value:,.0f}",
                delta_color="normal",
                help="Projected value from positive performance anomalies"
            )
        
        with risk_cols[3]:
            high_risk_count = len([r for r in risks if r.get('risk_category', '').startswith(('High', 'Critical'))])
            high_opp_count = len([o for o in opportunities if o.get('opportunity_category', '').startswith(('High', 'Major'))])
            st.metric(
                "🔥 Priority Items",
                f"{high_risk_count + high_opp_count} alerts",
                help="High/Critical risks plus High/Major opportunities requiring attention"
            )
        
        # Create tabs for risks vs opportunities
        if risks or opportunities:
            risk_tab, opp_tab = st.tabs(["🚨 Risk Analysis", "🚀 Opportunities"])
            
            with risk_tab:
                if risks:
                    # Risk category breakdown
                    if risk_distribution:
                        st.markdown("**Risk Distribution:**")
                        risk_breakdown_cols = st.columns(len(risk_distribution))
                        
                        for i, (risk_cat, count) in enumerate(risk_distribution.items()):
                            with risk_breakdown_cols[i]:
                                # Color code by risk level
                                risk_color = {
                                    'Critical Risk': '🔴',
                                    'High Risk': '🟠', 
                                    'Medium Risk': '🟡',
                                    'Low Risk': '🟢'
                                }.get(risk_cat, '⚪')
                                
                                st.metric(f"{risk_color} {risk_cat}", f"{count} risks")
                    
                    # Top risks
                    st.markdown("**Top Risk Anomalies:**")
                    top_risks = sorted(risks, key=lambda x: x.get('impact_score', 0), reverse=True)[:3]
                    
                    for i, risk in enumerate(top_risks, 1):
                        with st.expander(f"#{i} - {risk['metric'].upper()} {risk['direction'].title()} ({risk['severity']})"):
                            risk_detail_cols = st.columns(3)
                            
                            with risk_detail_cols[0]:
                                st.write(f"**Impact Score:** {risk.get('impact_score', 0):.1f}/100")
                                st.write(f"**Risk Category:** {risk.get('risk_category', 'Unknown')}")
                                st.write(f"**Confidence:** {risk.get('confidence_level', 85):.0f}%")
                            
                            with risk_detail_cols[1]:
                                spend_risk = risk.get('spend_at_risk', 0)
                                revenue_risk = risk.get('revenue_at_risk', 0)
                                if spend_risk > 0:
                                    st.write(f"**Spend at Risk:** ₹{spend_risk:,.0f}")
                                if revenue_risk > 0:
                                    st.write(f"**Revenue at Risk:** ₹{revenue_risk:,.0f}")
                                
                                projection_days = risk.get('projected_days', 7)
                                st.write(f"**Projection Period:** {projection_days} days")
                            
                            with risk_detail_cols[2]:
                                st.write(f"**Observed:** {risk['observed_value']:.2f}")
                                st.write(f"**Expected:** {risk['expected_value']:.2f}")
                                st.write(f"**Deviation:** {risk['deviation_percent']:.1f}%")
                else:
                    st.info("🎉 No risks detected - all anomalies are positive!")
            
            with opp_tab:
                if opportunities:
                    # Opportunity category breakdown
                    if opportunity_distribution:
                        st.markdown("**Opportunity Distribution:**")
                        opp_breakdown_cols = st.columns(len(opportunity_distribution))
                        
                        for i, (opp_cat, count) in enumerate(opportunity_distribution.items()):
                            with opp_breakdown_cols[i]:
                                # Color code by opportunity level
                                opp_color = {
                                    'Major Opportunity': '💎',
                                    'High Opportunity': '⭐', 
                                    'Medium Opportunity': '🌟',
                                    'Minor Opportunity': '✨'
                                }.get(opp_cat, '⚪')
                                
                                st.metric(f"{opp_color} {opp_cat}", f"{count} opportunities")
                    
                    # Top opportunities
                    st.markdown("**Top Opportunities:**")
                    top_opportunities = sorted(opportunities, key=lambda x: x.get('impact_score', 0), reverse=True)[:3]
                    
                    for i, opp in enumerate(top_opportunities, 1):
                        with st.expander(f"#{i} - {opp['metric'].upper()} {opp['direction'].title()} ({opp['severity']})"):
                            opp_detail_cols = st.columns(3)
                            
                            with opp_detail_cols[0]:
                                st.write(f"**Impact Score:** {opp.get('impact_score', 0):.1f}/100")
                                st.write(f"**Opportunity:** {opp.get('opportunity_category', 'Unknown')}")
                                st.write(f"**Confidence:** {opp.get('confidence_level', 85):.0f}%")
                            
                            with opp_detail_cols[1]:
                                opportunity_value = opp.get('opportunity_value', 0)
                                if opportunity_value > 0:
                                    st.write(f"**Opportunity Value:** ₹{opportunity_value:,.0f}")
                                
                                projection_days = opp.get('projected_days', 7)
                                st.write(f"**Projection Period:** {projection_days} days")
                            
                            with opp_detail_cols[2]:
                                st.write(f"**Observed:** {opp['observed_value']:.2f}")
                                st.write(f"**Expected:** {opp['expected_value']:.2f}")
                                st.write(f"**Deviation:** +{opp['deviation_percent']:.1f}%")
                                
                                # Opportunity-specific recommendations would go here
                                st.write("**Optimization Ideas:**")
                                if opp['metric'] == 'ctr' and opp['direction'] == 'increase':
                                    st.write("• Scale successful ad creative")
                                    st.write("• Increase budget allocation")
                                elif opp['metric'] == 'cpc' and opp['direction'] == 'decrease':
                                    st.write("• Expand to similar keywords")
                                    st.write("• Increase bid competitiveness")
                else:
                    st.info("💼 No significant opportunities identified in current data.")
        
        # Historical Pattern Analysis
        if len(anomalies_data) >= 5:  # Need sufficient data for pattern analysis
            st.markdown("---")
            st.subheader("📊 Historical Pattern Analysis")
            
            # Analyze patterns in current anomalies
            historical_patterns = anomaly_engine.analyze_historical_patterns(
                anomalies_data, 
                lookback_days=time_range_days if time_range_days else 30
            )
            
            # Generate insights from patterns
            pattern_insights = anomaly_engine.get_pattern_insights(historical_patterns)
            
            # Display pattern analysis in tabs
            pattern_tab1, pattern_tab2 = st.tabs(["🔍 Key Insights", "📈 Pattern Details"])
            
            with pattern_tab1:
                if pattern_insights:
                    st.markdown("**🧠 AI-Generated Insights:**")
                    for i, insight in enumerate(pattern_insights, 1):
                        st.write(f"{i}. {insight}")
                    
                    # Pattern summary metrics
                    st.markdown("**Pattern Summary:**")
                    insight_cols = st.columns(3)
                    
                    with insight_cols[0]:
                        recurring_count = len(historical_patterns.get('recurring_patterns', []))
                        st.metric("⏰ Time Patterns", f"{recurring_count} detected")
                    
                    with insight_cols[1]:
                        correlation_count = len(historical_patterns.get('metric_correlations', {}))
                        st.metric("🔗 Correlations", f"{correlation_count} found")
                    
                    with insight_cols[2]:
                        freq_analysis = historical_patterns.get('frequency_analysis', {})
                        unstable_metrics = len([m for m, data in freq_analysis.items() if data.get('anomaly_rate', 0) > 0.3])
                        st.metric("⚡ Unstable Metrics", f"{unstable_metrics} metrics")
                else:
                    st.info("📋 Insufficient data for meaningful pattern analysis. More anomalies needed.")
            
            with pattern_tab2:
                # Detailed pattern breakdown
                patterns_found = False
                
                # Recurring patterns
                recurring_patterns = historical_patterns.get('recurring_patterns', [])
                if recurring_patterns:
                    patterns_found = True
                    st.markdown("**⏰ Recurring Time Patterns:**")
                    for pattern in recurring_patterns:
                        if pattern['pattern_type'] == 'hourly':
                            times_str = ', '.join(map(str, pattern['common_times']))
                            freq_str = ', '.join(map(str, pattern['frequency']))
                            st.write(f"• **{pattern['metric'].upper()}**: Most active at hours {times_str} (occurrences: {freq_str})")
                        elif pattern['pattern_type'] == 'daily':
                            days_str = ', '.join(pattern['common_days'])
                            freq_str = ', '.join(map(str, pattern['frequency']))
                            st.write(f"• **{pattern['metric'].upper()}**: Problematic on {days_str} (occurrences: {freq_str})")
                
                # Frequency analysis
                freq_analysis = historical_patterns.get('frequency_analysis', {})
                if freq_analysis:
                    patterns_found = True
                    st.markdown("**📊 Metric Stability Analysis:**")
                    
                    # Create frequency table
                    freq_data = []
                    for metric, data in freq_analysis.items():
                        stability_level = "High" if data['anomaly_rate'] < 0.2 else "Medium" if data['anomaly_rate'] < 0.5 else "Low"
                        freq_data.append({
                            'Metric': metric.upper(),
                            'Total Anomalies': data['total_anomalies'],
                            'Rate (per day)': f"{data['anomaly_rate']:.2f}",
                            'Stability': stability_level
                        })
                    
                    if freq_data:
                        freq_df = pd.DataFrame(freq_data)
                        st.dataframe(freq_df, hide_index=True)
                
                # Correlation analysis
                correlations = historical_patterns.get('metric_correlations', {})
                if correlations:
                    patterns_found = True
                    st.markdown("**🔗 Cross-Metric Correlations:**")
                    for pair, data in correlations.items():
                        metrics = pair.split('_')
                        strength_text = "Strong" if data['correlation_strength'] > 0.7 else "Moderate" if data['correlation_strength'] > 0.4 else "Weak"
                        st.write(f"• **{metrics[0].upper()} ↔ {metrics[1].upper()}**: {strength_text} correlation ({data['co_occurrences']} co-occurrences)")
                
                # Time-based patterns
                time_patterns = historical_patterns.get('time_based_patterns', {})
                if time_patterns:
                    patterns_found = True
                    st.markdown("**📈 Time-based Trends:**")
                    
                    trend_cols = st.columns(2)
                    with trend_cols[0]:
                        st.write(f"**Daily Average:** {time_patterns.get('daily_average', 0):.1f} anomalies")
                        st.write(f"**Trend Direction:** {time_patterns.get('trend_direction', 'stable').title()}")
                    
                    with trend_cols[1]:
                        st.write(f"**Quiet Days:** {time_patterns.get('quiet_periods', 0)} days")
                        if time_patterns.get('peak_days'):
                            peak_day = list(time_patterns['peak_days'].keys())[0]
                            peak_count = list(time_patterns['peak_days'].values())[0]
                            st.write(f"**Peak Day:** {peak_day} ({peak_count} anomalies)")
                
                if not patterns_found:
                    st.info("🔄 Continue monitoring for more comprehensive pattern analysis.")
        
        else:
            if len(anomalies_data) > 0:
                st.markdown("---")
                st.info("📈 Historical pattern analysis will be available when more anomalies are detected (minimum 5 required).")
        
        # Correlated Anomaly Detection
        if len(anomalies_data) >= 3:  # Need at least 3 anomalies for correlation analysis
            st.markdown("---")
            st.subheader("🔗 Correlated Anomaly Analysis")
            
            # Detect correlated anomalies
            correlation_results = anomaly_engine.detect_correlated_anomalies(
                anomalies_data, 
                time_window_hours=2
            )
            
            # Display results in tabs
            corr_tab1, corr_tab2, corr_tab3 = st.tabs(["🎯 Correlation Groups", "⚠️ Campaign Failures", "📊 Metric Cascades"])
            
            with corr_tab1:
                correlated_groups = correlation_results.get('correlated_groups', [])
                if correlated_groups:
                    st.markdown("**Time-clustered anomaly groups detected:**")
                    
                    for i, group in enumerate(correlated_groups[:3], 1):  # Show top 3 groups
                        with st.expander(f"🔍 Correlation Group {i} - {len(group['affected_campaigns'])} campaigns affected"):
                            primary = group['primary_anomaly']
                            corr_anomalies = group['correlated_anomalies']
                            
                            st.write(f"**Primary Anomaly:** {primary['campaign_name']} - {primary['metric'].upper()}")
                            st.write(f"**Time Window:** {str(primary['timestamp'])[:16]} (±{group['time_window_hours']}h)")
                            st.write(f"**Correlation Strength:** {group['correlation_strength']:.2%}")
                            
                            # Show correlated anomalies
                            if corr_anomalies:
                                corr_data = []
                                for anom in corr_anomalies:
                                    time_diff = (pd.to_datetime(anom['timestamp']) - pd.to_datetime(primary['timestamp'])).total_seconds() / 3600
                                    corr_data.append({
                                        'Campaign': anom['campaign_name'],
                                        'Metric': anom['metric'].upper(),
                                        'Severity': anom['severity'],
                                        'Time Offset': f"{time_diff:+.1f}h"
                                    })
                                
                                st.write("**Correlated Anomalies:**")
                                st.dataframe(pd.DataFrame(corr_data), hide_index=True)
                else:
                    st.info("🔍 No significant correlation groups detected in current time window.")
            
            with corr_tab2:
                campaign_failures = correlation_results.get('campaign_failures', [])
                if campaign_failures:
                    st.markdown("**Campaign-wide failures detected:**")
                    
                    for failure in campaign_failures:
                        with st.expander(f"🚨 {failure['campaign_name']} - {failure['anomaly_count']} anomalies"):
                            st.write(f"**Failure Period:** {failure['failure_period']['start'][:16]} → {failure['failure_period']['end'][:16]}")
                            st.write(f"**Duration:** {failure['time_span_hours']:.1f} hours")
                            st.write(f"**Spend at Risk:** ${failure['total_spend_at_risk']:,.2f}")
                            
                            # Show affected metrics
                            metrics_cols = st.columns(len(failure['affected_metrics']))
                            for i, metric in enumerate(failure['affected_metrics']):
                                with metrics_cols[i]:
                                    st.metric(metric.upper(), "Failed", delta="⚠️")
                            
                            # Show severity distribution
                            if failure['severity_distribution']:
                                st.write("**Severity Breakdown:**")
                                sev_cols = st.columns(len(failure['severity_distribution']))
                                for i, (sev, count) in enumerate(failure['severity_distribution'].items()):
                                    with sev_cols[i]:
                                        st.metric(f"Level {sev}", f"{count} issues")
                else:
                    st.info("✅ No campaign-wide failures detected in the current dataset.")
            
            with corr_tab3:
                metric_cascades = correlation_results.get('metric_cascades', [])
                if metric_cascades:
                    st.markdown("**Metric cascade patterns detected:**")
                    
                    # Sort by cascade strength
                    cascades_sorted = sorted(metric_cascades, key=lambda x: x['cascade_strength'], reverse=True)
                    
                    for cascade in cascades_sorted[:5]:  # Show top 5 cascades
                        strength_text = "Strong" if cascade['cascade_strength'] > 0.7 else "Moderate" if cascade['cascade_strength'] > 0.4 else "Weak"
                        
                        cascade_col1, cascade_col2 = st.columns([3, 1])
                        with cascade_col1:
                            st.write(f"**{cascade['trigger_metric'].upper()}** → **{cascade['affected_metric'].upper()}**")
                            st.write(f"*{cascade['cascade_occurrences']} occurrences, {strength_text} correlation*")
                        
                        with cascade_col2:
                            st.metric("Delay", cascade['typical_delay_hours'])
                    
                    # Cascade insights
                    st.markdown("**💡 Cascade Insights:**")
                    if any(c['cascade_strength'] > 0.5 for c in cascades_sorted):
                        st.write("• Strong cascade patterns suggest systematic issues")
                        st.write("• Focus on addressing trigger metrics to prevent downstream effects")
                    else:
                        st.write("• Cascade patterns are weak, issues may be independent")
                else:
                    st.info("📊 No significant metric cascade patterns detected.")
        
        else:
            if len(anomalies_data) > 0:
                st.markdown("---")
                st.info("🔗 Correlated anomaly analysis will be available when more anomalies are detected (minimum 3 required).")
        
        # Email alert buttons
        st.subheader("📧 Alert Actions")
        alert_cols = st.columns(3)
        
        with alert_cols[0]:
            if st.button("📨 Send Email Alert"):
                ui_components.show_email_modal(selected_campaign, anomalies_df.iloc[0] if not anomalies_df.empty else None)
        
        with alert_cols[1]:
            if st.button("🔔 Toast Notification"):
                st.success("🔔 Anomaly alert sent to dashboard!")
        
        with alert_cols[2]:
            if st.button("📋 Generate Report"):
                st.info("📋 Detailed anomaly report generated!")
        
    
    else:
        st.success("✅ No anomalies detected for the selected metrics and sensitivity level.")
    
    # Enhanced Export Section - Available for all scenarios
    st.subheader("📤 Export & Downloads")
    download_cols = st.columns(4)
    
    with download_cols[0]:
        st.write("**Campaign Summary**")
        # Create campaign summary report
        summary_data = {
            'Campaign': [selected_campaign],
            'Platform': [campaign_info['platform']],
            'Target Group': [campaign_info['target_group']],
            'Daily Budget': [f"₹{campaign_info['daily_budget']:,.0f}"],
            'Total Anomalies': [len(anomalies_data) if anomalies_data else 0],
            'Critical (L3)': [len([a for a in anomalies_data if a['severity'] == 'L3']) if anomalies_data else 0],
            'Moderate (L2)': [len([a for a in anomalies_data if a['severity'] == 'L2']) if anomalies_data else 0],
            'Mild (L1)': [len([a for a in anomalies_data if a['severity'] == 'L1']) if anomalies_data else 0],
            'Report Generated': [datetime.now().strftime('%Y-%m-%d %H:%M')]
        }
        summary_df = pd.DataFrame(summary_data)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        summary_filename = f"campaign_summary_{selected_campaign.replace(' ', '_')}_{timestamp}.csv"
        ui_components.create_download_button(
            summary_df,
            summary_filename,
            "csv",
            "📄 Download Summary"
        )
    
    with download_cols[1]:
        if anomalies_data:
            st.write("**Anomaly Reports**")
            # Generate export data
            export_data = ui_components.export_anomaly_report(
                anomalies_data, 
                selected_campaign,
                campaign_info.to_dict()
            )
            
            if export_data is not None:
                filename = f"anomaly_report_{selected_campaign.replace(' ', '_')}_{timestamp}.csv"
                ui_components.create_download_button(
                    export_data, 
                    filename, 
                    "csv", 
                    "📊 Download CSV Report"
                )
                
                # PDF Report
                pdf_data = ui_components.create_anomaly_pdf_report(
                    anomalies_data,
                    selected_campaign,
                    campaign_info.to_dict()
                )
                if pdf_data:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_data,
                        file_name=f"anomaly_report_{selected_campaign.replace(' ', '_')}_{timestamp}.pdf",
                        mime="application/pdf",
                        key=f"pdf_report_{timestamp}"
                    )
        else:
            st.write("**No Anomalies**")
            st.info("No anomaly reports available")
    
    with download_cols[2]:
        st.write("**Chart Exports**")
        # We'll add chart download buttons after creating the charts
        st.info("Charts available below")
    
    with download_cols[3]:
        st.write("**Bulk Export**")
        if st.button("📦 Export All Campaigns", key="bulk_export_main"):
            # Generate bulk export for all campaigns with anomalies
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            all_anomalies = []
            for _, camp in campaigns_data.iterrows():
                camp_ts_data = pd.DataFrame(time_series_data[time_series_data['campaign_id'] == camp['campaign_id']])
                if not camp_ts_data.empty:
                    for metric in ['ctr', 'cpc', 'spend', 'conversions', 'roi']:
                        if metric in camp_ts_data.columns:
                            metric_data = camp_ts_data[['timestamp', metric]].copy()
                            metric_data = metric_data.sort_values('timestamp')
                            
                            anomalies = anomaly_engine.detect_anomalies(
                                metric_data, metric, z_threshold=2.0, deviation_threshold=30
                            )
                            
                            for anomaly in anomalies:
                                anomaly['campaign_name'] = camp['campaign_name']
                                anomaly['platform'] = camp['platform']
                                anomaly['campaign_id'] = camp['campaign_id']
                                all_anomalies.append(anomaly)
            
            if all_anomalies:
                bulk_df = pd.DataFrame(all_anomalies)
                if 'timestamp' in bulk_df.columns:
                    bulk_df['timestamp'] = bulk_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                bulk_filename = f"all_campaigns_anomalies_{timestamp}.csv"
                ui_components.create_download_button(
                    bulk_df,
                    bulk_filename,
                    "csv",
                    "📦 Download Bulk Report"
                )
                st.success(f"✅ Found {len(all_anomalies)} anomalies across {len(campaigns_data)} campaigns!")
            else:
                st.info("ℹ️ No anomalies found across all campaigns.")
    
    # KPI Trend Dashboard
    st.subheader("📈 KPI Trend Dashboard")
    
    # Create KPI trend charts for key metrics
    if 'ctr' in campaign_ts_data.columns and 'cpc' in campaign_ts_data.columns:
        kpi_cols = st.columns(2)
        
        # CTR Trend Chart
        with kpi_cols[0]:
            st.write("**CTR Performance Over Time**")
            
            ctr_data = campaign_ts_data[['timestamp', 'ctr']].copy()
            ctr_data = ctr_data.sort_values('timestamp')
            ctr_data['ctr_ma'] = ctr_data['ctr'].rolling(window=7, min_periods=1).mean()
            
            # Calculate CTR benchmark (industry average ~2%)
            ctr_benchmark = 2.0
            
            fig_ctr = go.Figure()
            
            # Actual CTR line
            fig_ctr.add_trace(go.Scatter(
                x=ctr_data['timestamp'],
                y=ctr_data['ctr'],
                mode='lines+markers',
                name='Actual CTR',
                line=dict(color='#3498db', width=3),
                marker=dict(size=5),
                hovertemplate='<b>CTR</b><br>Rate: %{y:.2f}%<br>Date: %{x}<extra></extra>'
            ))
            
            # Moving average
            fig_ctr.add_trace(go.Scatter(
                x=ctr_data['timestamp'],
                y=ctr_data['ctr_ma'],
                mode='lines',
                name='7-day Moving Average',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                hovertemplate='<b>7-day Average</b><br>Rate: %{y:.2f}%<br>Date: %{x}<extra></extra>'
            ))
            
            # Industry benchmark line
            fig_ctr.add_hline(
                y=ctr_benchmark,
                line_dash="dot",
                line_color="#95a5a6",
                annotation_text=f"Industry Avg: {ctr_benchmark}%",
                annotation_position="bottom right"
            )
            
            # Performance zones (Good/Warning/Poor)
            current_ctr = ctr_data['ctr'].iloc[-1] if len(ctr_data) > 0 else 0
            if current_ctr >= ctr_benchmark * 1.5:
                zone_color = "#27ae60"  # Green - Excellent
                zone_text = "🚀 Excellent Performance"
            elif current_ctr >= ctr_benchmark:
                zone_color = "#f39c12"  # Orange - Good
                zone_text = "✅ Above Average"
            else:
                zone_color = "#e74c3c"  # Red - Needs Attention
                zone_text = "⚠️ Below Benchmark"
            
            fig_ctr.update_layout(
                title=dict(
                    text=f"CTR Trend - {zone_text}",
                    x=0.5,
                    font=dict(size=14, color=zone_color)
                ),
                xaxis_title="Date",
                yaxis_title="CTR (%)",
                height=320,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_ctr)
            
            # CTR metrics
            ctr_change = ((current_ctr - ctr_data['ctr_ma'].iloc[-7]) / ctr_data['ctr_ma'].iloc[-7] * 100) if len(ctr_data) >= 7 else 0
            st.metric(
                "Current CTR",
                f"{current_ctr:.2f}%",
                delta=f"{ctr_change:+.1f}%" if ctr_change != 0 else None,
                delta_color="normal" if ctr_change >= 0 else "inverse"
            )
        
        # CPC Trend Chart
        with kpi_cols[1]:
            st.write("**CPC Performance Over Time**")
            
            cpc_data = campaign_ts_data[['timestamp', 'cpc']].copy()
            cpc_data = cpc_data.sort_values('timestamp')
            cpc_data['cpc_ma'] = cpc_data['cpc'].rolling(window=7, min_periods=1).mean()
            
            # Calculate CPC target (lower is better)
            cpc_target = cpc_data['cpc'].quantile(0.25)  # Best 25% performance
            
            fig_cpc = go.Figure()
            
            # Actual CPC line
            fig_cpc.add_trace(go.Scatter(
                x=cpc_data['timestamp'],
                y=cpc_data['cpc'],
                mode='lines+markers',
                name='Actual CPC',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=5),
                hovertemplate='<b>CPC</b><br>Cost: ₹%{y:.2f}<br>Date: %{x}<extra></extra>'
            ))
            
            # Moving average
            fig_cpc.add_trace(go.Scatter(
                x=cpc_data['timestamp'],
                y=cpc_data['cpc_ma'],
                mode='lines',
                name='7-day Moving Average',
                line=dict(color='#e67e22', width=2, dash='dash'),
                hovertemplate='<b>7-day Average</b><br>Cost: ₹%{y:.2f}<br>Date: %{x}<extra></extra>'
            ))
            
            # Target CPC line
            fig_cpc.add_hline(
                y=cpc_target,
                line_dash="dot",
                line_color="#95a5a6",
                annotation_text=f"Target: ₹{cpc_target:.2f}",
                annotation_position="top right"
            )
            
            # Performance zones (Good/Warning/Poor)
            current_cpc = cpc_data['cpc'].iloc[-1] if len(cpc_data) > 0 else 0
            if current_cpc <= cpc_target:
                zone_color = "#27ae60"  # Green - Excellent
                zone_text = "🎯 Cost Efficient"
            elif current_cpc <= cpc_target * 1.5:
                zone_color = "#f39c12"  # Orange - Acceptable
                zone_text = "📊 Acceptable Range"
            else:
                zone_color = "#e74c3c"  # Red - High Cost
                zone_text = "💸 High CPC Alert"
            
            fig_cpc.update_layout(
                title=dict(
                    text=f"CPC Trend - {zone_text}",
                    x=0.5,
                    font=dict(size=14, color=zone_color)
                ),
                xaxis_title="Date",
                yaxis_title="CPC (₹)",
                height=320,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_cpc)
            
            # CPC metrics
            cpc_change = ((current_cpc - cpc_data['cpc_ma'].iloc[-7]) / cpc_data['cpc_ma'].iloc[-7] * -100) if len(cpc_data) >= 7 else 0  # Negative change is good for CPC
            st.metric(
                "Current CPC",
                f"₹{current_cpc:.2f}",
                delta=f"{cpc_change:+.1f}%" if cpc_change != 0 else None,
                delta_color="normal" if cpc_change >= 0 else "inverse"
            )
        
        # KPI Summary Row
        st.markdown("---")
        kpi_summary_cols = st.columns(4)
        
        with kpi_summary_cols[0]:
            avg_ctr = ctr_data['ctr'].mean() if len(ctr_data) > 0 else 0
            st.metric("Average CTR", f"{avg_ctr:.2f}%")
        
        with kpi_summary_cols[1]:
            avg_cpc = cpc_data['cpc'].mean() if len(cpc_data) > 0 else 0
            st.metric("Average CPC", f"₹{avg_cpc:.2f}")
        
        with kpi_summary_cols[2]:
            ctr_volatility = ctr_data['ctr'].std() if len(ctr_data) > 1 else 0
            st.metric("CTR Volatility", f"{ctr_volatility:.2f}%")
        
        with kpi_summary_cols[3]:
            cpc_volatility = cpc_data['cpc'].std() if len(cpc_data) > 1 else 0
            st.metric("CPC Volatility", f"₹{cpc_volatility:.2f}")
    
    else:
        st.info("🔍 CTR and CPC data not available for this campaign. KPI dashboard requires these metrics.")
    
    # Visualizations
    st.subheader("📊 Metric Visualizations")
    
    if selected_metrics:
        chart_cols = st.columns(min(len(selected_metrics), 2))
        
        for i, metric in enumerate(selected_metrics):
            with chart_cols[i % 2]:
                st.write(f"**{metric.upper()} Trend**")
                
                # Create chart data
                metric_data = campaign_ts_data[['timestamp', metric]].copy()
                metric_data = metric_data.sort_values('timestamp')
                
                # Calculate rolling average
                metric_data['rolling_avg'] = metric_data[metric].rolling(window=7, min_periods=1).mean()
                
                # Create plotly chart
                fig = go.Figure()
                
                # Add observed values with enhanced tooltips
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data[metric],
                    mode='lines+markers',
                    name='Observed',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{metric.upper()}</b><br>' +
                                 'Value: %{y:.2f}<br>' +
                                 'Time: %{x}<br>' +
                                 '<extra></extra>'
                ))
                
                # Add rolling average with baseline shading
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data['rolling_avg'],
                    mode='lines',
                    name='7-day Average (Baseline)',
                    line=dict(color='#2ca02c', width=2, dash='dash'),
                    hovertemplate='<b>Baseline</b><br>' +
                                 'Average: %{y:.2f}<br>' +
                                 'Time: %{x}<br>' +
                                 '<extra></extra>'
                ))
                
                # Add confidence band around baseline
                if len(metric_data) > 7:
                    rolling_std = metric_data[metric].rolling(window=7, min_periods=3).std()
                    upper_band = metric_data['rolling_avg'] + (rolling_std * 1.5)
                    lower_band = metric_data['rolling_avg'] - (rolling_std * 1.5)
                    
                    fig.add_trace(go.Scatter(
                        x=metric_data['timestamp'],
                        y=upper_band,
                        mode='lines',
                        name='Normal Range',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=metric_data['timestamp'],
                        y=lower_band,
                        mode='lines',
                        name='Normal Range',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(46, 160, 44, 0.1)',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add severity-based anomaly points with enhanced markers
                metric_anomalies = [a for a in anomalies_data if a['metric'] == metric]
                if metric_anomalies:
                    severity_colors = {'L1': '#FFA500', 'L2': '#FF6B35', 'L3': '#DC143C'}
                    severity_sizes = {'L1': 8, 'L2': 10, 'L3': 12}
                    
                    for severity in ['L1', 'L2', 'L3']:
                        severity_anomalies = [a for a in metric_anomalies if a['severity'] == severity]
                        if severity_anomalies:
                            timestamps = [a['timestamp'] for a in severity_anomalies]
                            values = [a['observed_value'] for a in severity_anomalies]
                            deviations = [a['deviation_percent'] for a in severity_anomalies]
                            z_scores = [a['z_score'] for a in severity_anomalies]
                            
                            # Prepare custom data for tooltips
                            customdata = [[d, z] for d, z in zip(deviations, z_scores)]
                            
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=values,
                                mode='markers',
                                name=f'{severity} Anomalies',
                                marker=dict(
                                    color=severity_colors[severity],
                                    size=severity_sizes[severity],
                                    symbol='diamond',
                                    line=dict(width=2, color='white')
                                ),
                                customdata=customdata,
                                hovertemplate=f'<b>🚨 {severity} Anomaly</b><br>' +
                                             f'{metric.upper()}: %{{y:.2f}}<br>' +
                                             'Deviation: %{customdata[0]:.1f}%<br>' +
                                             'Z-Score: %{customdata[1]:.2f}<br>' +
                                             'Time: %{x}<br>' +
                                             '<extra></extra>'
                            ))
                            
                            # Add severity-based background shading for anomaly time periods
                            for i, timestamp in enumerate(timestamps):
                                fig.add_vrect(
                                    x0=timestamp - pd.Timedelta(hours=2),
                                    x1=timestamp + pd.Timedelta(hours=2),
                                    fillcolor=severity_colors[severity],
                                    opacity=0.1,
                                    layer='below',
                                    line_width=0,
                                    annotation_text="",
                                    showlegend=False
                                )
                
                # Enhanced chart styling
                fig.update_layout(
                    height=350,
                    showlegend=True,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="Time",
                    yaxis_title=metric.upper(),
                    title=dict(
                        text=f"{metric.upper()} Anomaly Detection",
                        x=0.5,
                        font=dict(size=14)
                    ),
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=11),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                # Add grid
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                
                st.plotly_chart(fig)
                
                # Chart export options
                chart_export_cols = st.columns(2)
                with chart_export_cols[0]:
                    # PNG Export
                    png_bytes, png_mime = ui_components.export_chart_as_image(fig, f"{metric}_chart", "png")
                    if png_bytes:
                        st.download_button(
                            label=f"📷 Download {metric.upper()} PNG",
                            data=png_bytes,
                            file_name=f"{selected_campaign}_{metric}_chart_{timestamp}.png",
                            mime=png_mime,
                            key=f"png_{metric}_{timestamp}"
                        )
                
                with chart_export_cols[1]:
                    # PDF Export
                    pdf_bytes, pdf_mime = ui_components.export_chart_as_image(fig, f"{metric}_chart", "pdf")
                    if pdf_bytes:
                        st.download_button(
                            label=f"📄 Download {metric.upper()} PDF",
                            data=pdf_bytes,
                            file_name=f"{selected_campaign}_{metric}_chart_{timestamp}.pdf", 
                            mime=pdf_mime,
                            key=f"pdf_{metric}_{timestamp}"
                        )
    
    # Summary statistics
    st.subheader("📈 Campaign Summary")
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        total_anomalies = len(anomalies_data)
        st.metric("Total Anomalies", total_anomalies)
    
    with summary_cols[1]:
        critical_anomalies = len([a for a in anomalies_data if a['severity'] == 'L3'])
        st.metric("Critical (L3)", critical_anomalies)
    
    with summary_cols[2]:
        if anomalies_data:
            avg_deviation = np.mean([abs(a['deviation_percent']) for a in anomalies_data])
            st.metric("Avg Deviation", f"{avg_deviation:.1f}%")
        else:
            st.metric("Avg Deviation", "0.0%")
    
    with summary_cols[3]:
        data_points = len(campaign_ts_data)
        st.metric("Data Points", data_points)

if __name__ == "__main__":
    main()
