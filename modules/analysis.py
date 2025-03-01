import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import json
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_trends(df: pd.DataFrame) -> dict:
    """Analyze wound care data with simplified data handling"""
    analysis = {}

    # Convert WEEK to categorical with proper ordering
    df['WEEK'] = pd.Categorical(df['WEEK'],
                              categories=sorted(df['WEEK'].unique(), key=lambda x: int(x.split()[-1])),
                              ordered=True)

    # 1. Weekly Trends Analysis
    weekly = df.groupby('WEEK', observed=True)['TOTAL_WOUND_AREA'].agg(['mean', 'std']).reset_index()
    weekly.columns = ['WEEK', 'Mean_Wound_Area', 'Std_Wound_Area']
    weekly['Total_Wounds'] = df.groupby('WEEK', observed=True)['WOUND_COUNT'].sum().reindex(weekly['WEEK']).values

    # Ensure numeric types
    weekly['Mean_Wound_Area'] = pd.to_numeric(weekly['Mean_Wound_Area'], errors='coerce')
    weekly['Std_Wound_Area'] = pd.to_numeric(weekly['Std_Wound_Area'], errors='coerce')
    weekly['Total_Wounds'] = pd.to_numeric(weekly['Total_Wounds'], errors='coerce')

    # Store raw data separately
    analysis['weekly_trends_data'] = weekly.to_dict(orient='records')

    # Generate Plotly figure
    fig = px.line(
        weekly,
        x='WEEK',
        y='Mean_Wound_Area',
        error_y='Std_Wound_Area',
        title='Weekly Wound Area Trends',
        labels={'Mean_Wound_Area': 'Mean Wound Area (cm²)', 'WEEK': 'Week'}
    )
    analysis['weekly_trends'] = fig.to_json()

    # 2. Product Performance Analysis
    product_stats = df.groupby('NAME', observed=True).agg({
        'TOTAL_WOUND_AREA': 'mean',
        'WOUND_COUNT': 'count'
    }).reset_index()
    product_stats.columns = ['NAME', 'Mean_Wound_Area', 'Usage_Count']
    product_stats = product_stats.sort_values('Mean_Wound_Area', ascending=False).head(10)

    analysis['product_performance'] = px.bar(
        product_stats,
        x='NAME',
        y='Mean_Wound_Area',
        title='Top 10 Products by Mean Wound Area',
        labels={'NAME': 'Product Name', 'Mean_Wound_Area': 'Average Wound Area (cm²)'}
    ).to_json()

    # 3. Correlation Matrix
    numeric_cols = ['TOTAL_WOUND_AREA', 'WOUND_COUNT', 'AVG_WOUND_AREA']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    analysis['correlation_matrix'] = df[numeric_cols].corr().round(2).to_dict()

    # 4. Time Series Decomposition
    if len(weekly) > 12:
        weekly_ts = weekly.set_index('WEEK')['Mean_Wound_Area']
        decomposition = seasonal_decompose(weekly_ts, model='additive', period=4)
        analysis['seasonality'] = {
            'trend': decomposition.trend.reset_index().rename(columns={'WEEK': 'Week', 'trend': 'Value'}).to_dict(orient='records'),
            'seasonal': decomposition.seasonal.reset_index().rename(columns={'WEEK': 'Week', 'seasonal': 'Value'}).to_dict(orient='records'),
            'residual': decomposition.resid.reset_index().rename(columns={'WEEK': 'Week', 'resid': 'Value'}).to_dict(orient='records')
        }

    # 5. Treatment Efficacy Analysis
    treatment_efficacy = df.groupby('NAME', observed=True).agg({
        'TOTAL_WOUND_AREA': 'sum',
        'WEEK': 'count'
    }).reset_index()
    treatment_efficacy.columns = ['NAME', 'Total_Healed_Area', 'Average_Treatment_Duration']
    treatment_efficacy['Healing_Rate'] = treatment_efficacy['Total_Healed_Area'] / treatment_efficacy['Average_Treatment_Duration']

    analysis['treatment_efficacy'] = px.scatter(
        treatment_efficacy,
        x='Average_Treatment_Duration',
        y='Healing_Rate',
        size='Total_Healed_Area',
        title='Treatment Efficacy Analysis',
        labels={'Average_Treatment_Duration': 'Average Treatment Duration (Weeks)',
                'Healing_Rate': 'Healing Rate (cm²/week)'}
    ).to_json()

    # 6. T-test for week-over-week comparison
    ttest_results = []
    weeks = sorted(df['WEEK'].unique())
    
    for i in range(len(weeks) - 1):
        week1 = df[df['WEEK'] == weeks[i]]['TOTAL_WOUND_AREA']
        week2 = df[df['WEEK'] == weeks[i+1]]['TOTAL_WOUND_AREA']
        t_stat, p_value = stats.ttest_ind(week1, week2, equal_var=False)
        
        ttest_results.append({
            'week1': str(weeks[i]),
            'week2': str(weeks[i+1]),
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        })
    
    analysis['ttest_results'] = {'week_over_week': ttest_results}
    
    return analysis

def summarize_data(df: pd.DataFrame) -> str:
    """Enhanced summary statistics"""
    stats = {
        "Total Records": len(df),
        "Unique Products": df['NAME'].nunique(),
        "Treatment Weeks": df['WEEK'].nunique(),
        "Total Wound Area (cm²)": f"{df['TOTAL_WOUND_AREA'].sum():,.2f}",
        "Average Wound Size (cm²)": f"{df['AVG_WOUND_AREA'].mean():.2f} ± {df['AVG_WOUND_AREA'].std():.2f}"
    }
    return json.dumps(stats, indent=2)

def summarize_analysis(analysis: dict) -> str:
    """Enhanced analysis summary"""
    insights = {
        "Weekly Trends": "Identified weekly patterns in wound area changes",
        "Top Products": "Ranked products by average wound area treated",
        "Key Correlations": "Discovered relationships between clinical variables",
        "Seasonal Patterns": "Decomposed trend/seasonality components" if 'seasonality' in analysis else "Insufficient data for decomposition",
        "Week over week t-test": "Performed t-test to compare week-over-week wound area"
    }
    return json.dumps(insights, indent=2)