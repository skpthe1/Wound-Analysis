import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import json
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.io import from_json
import base64  # Added for decoding bdata

def analyze_trends(df: pd.DataFrame) -> dict:
    """Analyze wound care data with simplified data handling and robust error handling"""
    analysis = {}

    # Check for empty DataFrame or missing columns
    required_cols = ['WEEK', 'TOTAL_WOUND_AREA', 'NAME', 'WOUND_COUNT', 'AVG_WOUND_AREA']
    if df.empty or not all(col in df.columns for col in required_cols):
        analysis['error'] = f"Input DataFrame is empty or missing required columns: {', '.join(required_cols)}"
        return analysis

    # Ensure WEEK is categorical with proper ordering
    try:
        df['WEEK'] = pd.Categorical(df['WEEK'],
                                  categories=sorted(df['WEEK'].unique(), key=lambda x: int(x.split()[-1])),
                                  ordered=True)
    except (ValueError, TypeError) as e:
        analysis['error'] = f"Failed to process WEEK column: {str(e)}"
        return analysis

    # 1. Weekly Trends Analysis
    try:
        weekly = df.groupby('WEEK', observed=True)['TOTAL_WOUND_AREA'].agg(['mean', 'std']).reset_index()
        weekly.columns = ['WEEK', 'Mean_Wound_Area', 'Std_Wound_Area']
        weekly['Total_Wounds'] = df.groupby('WEEK', observed=True)['WOUND_COUNT'].sum().reindex(weekly['WEEK']).fillna(0).values
        weekly['Mean_Wound_Area'] = pd.to_numeric(weekly['Mean_Wound_Area'], errors='coerce').fillna(0)
        weekly['Std_Wound_Area'] = pd.to_numeric(weekly['Std_Wound_Area'], errors='coerce').fillna(0)
        weekly['Total_Wounds'] = pd.to_numeric(weekly['Total_Wounds'], errors='coerce').fillna(0)

        analysis['weekly_trends_data'] = weekly.to_dict(orient='records')
        fig = px.line(weekly, x='WEEK', y='Mean_Wound_Area', error_y='Std_Wound_Area',
                      title='Weekly Wound Area Trends',
                      labels={'Mean_Wound_Area': 'Mean Wound Area (cm²)', 'WEEK': 'Week'})
        analysis['weekly_trends'] = fig.to_json()
    except Exception as e:
        analysis['weekly_trends'] = f'{{"error": "Weekly trends analysis failed: {str(e)}"}}'

    # 2. Product Performance Analysis
    try:
        product_stats = df.groupby('NAME', observed=True).agg({
            'TOTAL_WOUND_AREA': 'mean',
            'WOUND_COUNT': 'count'
        }).reset_index()
        product_stats.columns = ['NAME', 'Mean_Wound_Area', 'Usage_Count']
        product_stats['Mean_Wound_Area'] = pd.to_numeric(product_stats['Mean_Wound_Area'], errors='coerce').fillna(0)
        product_stats = product_stats[product_stats['Mean_Wound_Area'] > 0]
        product_stats = product_stats.sort_values('Mean_Wound_Area', ascending=False).head(10)

        if not product_stats.empty:
            fig = px.bar(product_stats, x='NAME', y='Mean_Wound_Area',
                        title='Top 10 Products by Mean Wound Area',
                        labels={'NAME': 'Product Name', 'Mean_Wound_Area': 'Average Wound Area (cm²)'})
            analysis['product_performance'] = fig.to_json()
        else:
            fig = px.bar(pd.DataFrame({'NAME': ['No Valid Data'], 'Mean_Wound_Area': [0]}),
                        x='NAME', y='Mean_Wound_Area',
                        title='Top 10 Products by Mean Wound Area (No Valid Data)')
            analysis['product_performance'] = fig.to_json()
    except Exception as e:
        analysis['product_performance'] = f'{{"error": "Product performance analysis failed: {str(e)}"}}'

    # 3. Correlation Matrix
    try:
        numeric_cols = ['TOTAL_WOUND_AREA', 'WOUND_COUNT', 'AVG_WOUND_AREA']
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        analysis['correlation_matrix'] = numeric_df.corr().round(2).to_dict()
    except Exception as e:
        analysis['correlation_matrix'] = f'{{"error": "Correlation matrix failed: {str(e)}"}}'

    # 4. Time Series Decomposition
    try:
        if len(weekly) > 12:
            weekly_ts = weekly.set_index('WEEK')['Mean_Wound_Area']
            decomposition = seasonal_decompose(weekly_ts, model='additive', period=4)
            analysis['seasonality'] = {
                'trend': decomposition.trend.reset_index().rename(columns={'WEEK': 'Week', 'trend': 'Value'}).to_dict(orient='records'),
                'seasonal': decomposition.seasonal.reset_index().rename(columns={'WEEK': 'Week', 'seasonal': 'Value'}).to_dict(orient='records'),
                'residual': decomposition.resid.reset_index().rename(columns={'WEEK': 'Week', 'resid': 'Value'}).to_dict(orient='records')
            }
    except Exception as e:
        analysis['seasonality'] = f'{{"error": "Seasonality analysis failed: {str(e)}"}}'

    # 5. Treatment Efficacy Analysis
    try:
        treatment_efficacy = df.groupby('NAME', observed=True).agg({
            'TOTAL_WOUND_AREA': 'sum',
            'WEEK': 'count'
        }).reset_index()
        treatment_efficacy.columns = ['NAME', 'Total_Healed_Area', 'Average_Treatment_Duration']
        treatment_efficacy['Healing_Rate'] = treatment_efficacy['Total_Healed_Area'] / treatment_efficacy['Average_Treatment_Duration'].replace(0, 1)
        fig = px.scatter(treatment_efficacy, x='Average_Treatment_Duration', y='Healing_Rate',
                         size='Total_Healed_Area', title='Treatment Efficacy Analysis',
                         labels={'Average_Treatment_Duration': 'Average Treatment Duration (Weeks)',
                                 'Healing_Rate': 'Healing Rate (cm²/week)'})
        analysis['treatment_efficacy'] = fig.to_json()
    except Exception as e:
        analysis['treatment_efficacy'] = f'{{"error": "Treatment efficacy analysis failed: {str(e)}"}}'

    # 6. T-test for week-over-week comparison
    try:
        ttest_results = []
        weeks = sorted(df['WEEK'].unique())
        for i in range(len(weeks) - 1):
            week1 = df[df['WEEK'] == weeks[i]]['TOTAL_WOUND_AREA'].dropna()
            week2 = df[df['WEEK'] == weeks[i+1]]['TOTAL_WOUND_AREA'].dropna()
            if (len(week1) > 1 and len(week2) > 1 and
                not week1.empty and not week2.empty and
                week1.var() > 1e-10 and week2.var() > 1e-10):
                t_stat, p_value = stats.ttest_ind(week1, week2, equal_var=False)
                ttest_results.append({
                    'week1': str(weeks[i]),
                    'week2': str(weeks[i+1]),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value)
                })
            else:
                ttest_results.append({
                    'week1': str(weeks[i]),
                    'week2': str(weeks[i+1]),
                    't_statistic': None,
                    'p_value': None,
                    'note': 'Skipped due to insufficient variance or sample size'
                })
        analysis['ttest_results'] = {'week_over_week': ttest_results}
    except Exception as e:
        analysis['ttest_results'] = f'{{"error": "T-test analysis failed: {str(e)}"}}'

    return analysis

def summarize_data(df: pd.DataFrame) -> str:
    """Enhanced summary statistics"""
    stats = {
        "Total Records": len(df),
        "Unique Products": df['NAME'].nunique() if 'NAME' in df.columns else 0,
        "Treatment Weeks": df['WEEK'].nunique(),
        "Total Wound Area (cm²)": f"{df['TOTAL_WOUND_AREA'].sum():,.2f}",
        "Average Wound Size (cm²)": f"{df['AVG_WOUND_AREA'].mean():.2f} ± {df['AVG_WOUND_AREA'].std():.2f}" if 'AVG_WOUND_AREA' in df.columns else "N/A"
    }
    return json.dumps(stats, indent=2)

def summarize_analysis(analysis: dict) -> str:
    """Generate a dynamic summary based on actual analysis results"""
    summary = {}

    if 'error' in analysis:
        summary['Error'] = analysis['error']
        return json.dumps(summary, indent=2)

    # 1. Weekly Trends
    if 'weekly_trends_data' in analysis and 'error' not in analysis.get('weekly_trends_data', ''):
        weekly_data = pd.DataFrame(analysis['weekly_trends_data'])
        if not weekly_data.empty:
            max_week = weekly_data.loc[weekly_data['Mean_Wound_Area'].idxmax()]['WEEK']
            min_week = weekly_data.loc[weekly_data['Mean_Wound_Area'].idxmin()]['WEEK']
            summary["Weekly Trends"] = f"Mean wound area peaked in {max_week} and was lowest in {min_week}"
        else:
            summary["Weekly Trends"] = "No valid weekly trend data available"

    # 2. Top Products
    if 'product_performance' in analysis:
        try:
            product_fig = from_json(analysis['product_performance'])
            if hasattr(product_fig, 'data') and product_fig.data and len(product_fig.data) > 0:
                trace = product_fig.data[0]
                if hasattr(trace, 'x') and trace.x and hasattr(trace, 'y') and trace.y:
                    # Handle case where y is a dict with dtype and bdata
                    if isinstance(trace.y, dict) and 'bdata' in trace.y and 'dtype' in trace.y:
                        y_array = np.frombuffer(base64.b64decode(trace.y['bdata']), dtype=trace.y['dtype'])
                        x_data = list(trace.x)
                        y_data = list(y_array)
                    else:
                        x_data = list(trace.x)
                        y_data = list(trace.y)
                    
                    if x_data and y_data and len(x_data) > 0 and len(y_data) > 0:
                        top_product = x_data[0]
                        top_product_area = float(y_data[0])
                        summary["Top Products"] = f"{top_product} treated the largest average wound area ({top_product_area:.2f} cm²)"
                    else:
                        summary["Top Products"] = "Product data arrays are empty"
                else:
                    summary["Top Products"] = "No valid x or y data in product trace"
            else:
                summary["Top Products"] = "No product data available in figure"
        except Exception as e:
            summary["Top Products"] = f"Error processing product data: {str(e)}"
    else:
        summary["Top Products"] = "Product performance data not found"

    # 3. Key Correlations
    if 'correlation_matrix' in analysis and 'error' not in analysis['correlation_matrix']:
        corr_matrix = analysis['correlation_matrix']
        max_corr = 0
        corr_pair = ""
        for col1 in corr_matrix:
            for col2 in corr_matrix[col1]:
                if col1 != col2 and abs(corr_matrix[col1][col2]) > abs(max_corr) and abs(corr_matrix[col1][col2]) < 1:
                    max_corr = corr_matrix[col1][col2]
                    corr_pair = f"{col1} vs {col2}"
        summary["Key Correlations"] = f"Strongest correlation ({max_corr:.2f}) between {corr_pair}" if max_corr != 0 else "No significant correlations found"

    # 4. Seasonal Patterns
    if 'seasonality' in analysis and 'error' not in analysis['seasonality']:
        summary["Seasonal Patterns"] = "Seasonal patterns detected with 4-week periodicity"
    else:
        summary["Seasonal Patterns"] = "Insufficient data for seasonal analysis"

    # 5. Week-over-week t-test
    if 'ttest_results' in analysis and 'error' not in analysis['ttest_results']:
        ttest_data = analysis['ttest_results']['week_over_week']
        significant_changes = sum(1 for t in ttest_data if t['p_value'] is not None and t['p_value'] < 0.05)
        skipped_tests = sum(1 for t in ttest_data if 'note' in t)
        summary["Week over week t-test"] = (f"Found {significant_changes} significant week-over-week changes (p<0.05)" +
                                            (f", {skipped_tests} tests skipped due to low variance" if skipped_tests > 0 else ""))

    return json.dumps(summary, indent=2)