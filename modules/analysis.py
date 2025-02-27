import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def analyze_trends(df: pd.DataFrame) -> dict:
    analysis = {}
    
    # Weekly trends
    weekly = df.groupby('WEEK').agg({
        'TOTAL_WOUND_AREA': 'mean',
        'WOUND_COUNT': 'sum',
        'AVG_WOUND_AREA': 'mean'
    })
    
    # Product performance
    products = df.groupby('NAME').agg({
        'TOTAL_WOUND_AREA': 'mean',
        'WOUND_COUNT': 'sum',
        'AVG_WOUND_AREA': 'mean'
    }).sort_values('TOTAL_WOUND_AREA', ascending=False)
    
    # Matplotlib visualization
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    weekly['TOTAL_WOUND_AREA'].plot(ax=ax[0], title='Weekly Total Wound Area Trend')
    products['TOTAL_WOUND_AREA'].head(10).plot(kind='bar', ax=ax[1], title='Top 10 Products by Total Wound Area')
    
    # Plotly visualizations
    weekly_plotly = px.line(weekly, y='TOTAL_WOUND_AREA', title='Weekly Total Wound Area Trend')
    products_plotly = px.bar(products.head(10), y='TOTAL_WOUND_AREA', title='Top 10 Products by Total Wound Area')
    
    analysis['weekly'] = weekly
    analysis['products'] = products
    analysis['matplotlib_fig'] = fig
    analysis['weekly_plotly'] = weekly_plotly
    analysis['products_plotly'] = products_plotly
    
    return analysis
