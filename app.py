"""
Tips Music: YouTube Performance vs Stock Price Correlation Dashboard
Author: Financial Analytics Team
Last Updated: January 2026
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import create_engine
import os
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

# Supabase connection (set as environment variables in Render)
SUPABASE_URL = os.getenv('SUPABASE_URL', 'postgresql://postgres:XQ4AJxtEbr41vAzs@db.bfafqccvzboyfjewzvhk.supabase.co:5432/postgres')

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="Tips Music Analytics",
    update_title="Loading...",
    suppress_callback_exceptions=True
)
server = app.server  # For Render deployment

# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_db_connection():
    """Create database connection"""
    return create_engine(SUPABASE_URL)

def fetch_monthly_data(label):
    """Fetch monthly aggregated data for Tips or Saregama"""
    engine = get_db_connection()
    
    query = f"""
    SELECT 
        month::date,
        label,
        total_subscribers,
        total_views,
        total_videos,
        prev_month_subs,
        prev_month_views,
        mom_subs_growth,
        mom_views_growth,
        mom_subs_growth_pct,
        mom_views_growth_pct
    FROM monthly_label_totals
    WHERE label = '{label}'
    ORDER BY month DESC;
    """
    
    df = pd.read_sql(query, engine)
    return df

def fetch_unified_data(days=365):
    """Fetch unified YouTube + Stock data"""
    engine = get_db_connection()
    
    query = f"""
    SELECT 
        date,
        tips_subscribers,
        tips_views,
        tips_videos,
        daily_subs_change,
        daily_views_change,
        weekly_subs_growth_pct,
        monthly_subs_growth_pct,
        stock_price,
        stock_volume,
        daily_return_pct,
        sma_7d,
        sma_30d,
        volatility_30d
    FROM unified_analysis
    WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
        AND stock_price IS NOT NULL
    ORDER BY date;
    """
    
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    return df

def fetch_daily_views():
    """Fetch daily views for Tips channels"""
    engine = get_db_connection()
    
    query = """
    SELECT 
        ds.date,
        SUM(ds.views) AS total_views,
        SUM(ds.subscribers) AS total_subscribers
    FROM daily_stats ds
    JOIN channels c ON ds.channel_id = c.channel_id
    WHERE c.label = 'Tips'
        AND ds.date >= '2023-01-01'
    GROUP BY ds.date
    ORDER BY ds.date;
    """
    
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate daily view change
    df['daily_view_change'] = df['total_views'].diff()
    
    return df

def calculate_rolling_correlation(df, window=45):
    """Calculate rolling correlation between views and stock price"""
    # Merge daily views with stock prices
    engine = get_db_connection()
    
    stock_query = """
    SELECT 
        date,
        close AS stock_price
    FROM stock_prices
    WHERE symbol = 'TIPSMUSIC'
        AND date >= '2023-01-01'
    ORDER BY date;
    """
    
    stock_df = pd.read_sql(stock_query, engine)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    # Merge
    merged = pd.merge(df, stock_df, on='date', how='inner')
    
    # Calculate rolling average
    merged['views_rolling_avg'] = merged['daily_view_change'].rolling(window=window, min_periods=1).mean()
    merged['stock_rolling_avg'] = merged['stock_price'].rolling(window=window, min_periods=1).mean()
    
    return merged

def find_best_correlation(df, max_lag=60):
    """Find best correlation between YouTube metrics and stock price"""
    results = []
    
    # Test different metrics and lags
    metrics = [
        ('daily_views_change', 'Daily Views Change'),
        ('weekly_subs_growth_pct', 'Weekly Subscriber Growth %'),
        ('monthly_subs_growth_pct', 'Monthly Subscriber Growth %'),
    ]
    
    for metric, metric_name in metrics:
        if metric not in df.columns:
            continue
            
        for lag in range(0, max_lag + 1):
            try:
                # Create lagged data
                df_clean = df[[metric, 'daily_return_pct']].copy()
                df_clean[f'{metric}_lagged'] = df_clean[metric].shift(lag)
                df_clean = df_clean.dropna()
                
                if len(df_clean) < 30:
                    continue
                
                # Calculate Pearson correlation
                corr_pearson, p_pearson = pearsonr(
                    df_clean[f'{metric}_lagged'], 
                    df_clean['daily_return_pct']
                )
                
                # Calculate Spearman correlation
                corr_spearman, p_spearman = spearmanr(
                    df_clean[f'{metric}_lagged'], 
                    df_clean['daily_return_pct']
                )
                
                results.append({
                    'metric': metric_name,
                    'lag_days': lag,
                    'pearson_corr': corr_pearson,
                    'pearson_p_value': p_pearson,
                    'spearman_corr': corr_spearman,
                    'spearman_p_value': p_spearman,
                    'is_significant': p_pearson < 0.05
                })
            except:
                continue
    
    results_df = pd.DataFrame(results)
    
    # Sort by absolute correlation (Pearson)
    results_df['abs_pearson'] = results_df['pearson_corr'].abs()
    results_df = results_df.sort_values('abs_pearson', ascending=False)
    
    return results_df

# ============================================================
# DASHBOARD LAYOUT
# ============================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📊 Tips Music: YouTube vs Stock Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#1f77b4', 'marginBottom': 10}),
        html.P("Correlation Analysis: YouTube Performance → Stock Price Prediction",
               style={'textAlign': 'center', 'color': '#666', 'fontSize': 16}),
        html.P(f"Last Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p IST')}",
               style={'textAlign': 'center', 'color': '#999', 'fontSize': 12})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Tabs
    dcc.Tabs(id='main-tabs', value='tab-monthly', children=[
        
        # TAB 1: Monthly Data Tables
        dcc.Tab(label='📅 Monthly Data', value='tab-monthly', children=[
            html.Div([
                html.H2("Monthly Performance Data", style={'marginTop': 20}),
                
                # Label selector
                html.Div([
                    html.Label("Select Label:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.RadioItems(
                        id='label-selector',
                        options=[
                            {'label': ' Tips Music', 'value': 'Tips'},
                            {'label': ' Saregama', 'value': 'Saregama'}
                        ],
                        value='Tips',
                        inline=True,
                        style={'marginBottom': 20}
                    )
                ]),
                
                # Metric selector
                html.Div([
                    html.Label("Show Columns:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Checklist(
                        id='metric-selector',
                        options=[
                            {'label': ' Subscribers', 'value': 'subs'},
                            {'label': ' Views', 'value': 'views'},
                            {'label': ' Videos', 'value': 'videos'},
                            {'label': ' Growth %', 'value': 'growth'}
                        ],
                        value=['subs', 'views', 'growth'],
                        inline=True,
                        style={'marginBottom': 20}
                    )
                ]),
                
                # Monthly data table
                html.Div(id='monthly-table-container')
                
            ], style={'padding': 20})
        ]),
        
        # TAB 2: Time Series Charts
        dcc.Tab(label='📈 Time Series', value='tab-timeseries', children=[
            html.Div([
                html.H2("Daily & Monthly Trends", style={'marginTop': 20}),
                
                # Date range selector
                html.Div([
                    html.Label("Select Time Range:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='time-range',
                        options=[
                            {'label': 'Last 30 Days', 'value': 30},
                            {'label': 'Last 90 Days', 'value': 90},
                            {'label': 'Last 180 Days', 'value': 180},
                            {'label': 'Last 365 Days', 'value': 365},
                            {'label': 'All Time (3 Years)', 'value': 1095}
                        ],
                        value=365,
                        style={'width': 300, 'marginBottom': 20}
                    )
                ]),
                
                # Chart type selector
                html.Div([
                    html.Label("Chart Scale:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.RadioItems(
                        id='scale-type',
                        options=[
                            {'label': ' Linear', 'value': 'linear'},
                            {'label': ' Logarithmic', 'value': 'log'}
                        ],
                        value='linear',
                        inline=True,
                        style={'marginBottom': 20}
                    )
                ]),
                
                # Daily views chart
                dcc.Graph(id='daily-views-chart'),
                
                # Monthly views chart
                dcc.Graph(id='monthly-views-chart'),
                
                # Monthly subscribers chart
                dcc.Graph(id='monthly-subs-chart')
                
            ], style={'padding': 20})
        ]),
        
        # TAB 3: Rolling Average vs Stock Price
        dcc.Tab(label='🔄 45-Day Rolling Average', value='tab-rolling', children=[
            html.Div([
                html.H2("45-Day Rolling Average: Daily Views vs Stock Price", style={'marginTop': 20}),
                html.P("This chart shows the 45-day moving average of daily view changes compared to Tips Music stock price.",
                       style={'color': '#666', 'marginBottom': 20}),
                
                # Rolling average chart
                dcc.Graph(id='rolling-average-chart'),
                
                # Correlation stats
                html.Div(id='rolling-stats', style={'marginTop': 20})
                
            ], style={'padding': 20})
        ]),
        
        # TAB 4: Correlation Analysis
        dcc.Tab(label='🔗 Correlation & Leading Indicators', value='tab-correlation', children=[
            html.Div([
                html.H2("Leading Indicators: YouTube Metrics → Stock Price", style={'marginTop': 20}),
                html.P("Finding the best predictive signals from YouTube data to stock price movements.",
                       style={'color': '#666', 'marginBottom': 20}),
                
                # Best correlation summary
                html.Div(id='best-correlation-summary', style={
                    'backgroundColor': '#e7f3ff',
                    'padding': 20,
                    'borderRadius': 5,
                    'marginBottom': 20
                }),
                
                # Lag correlation heatmap
                dcc.Graph(id='lag-correlation-heatmap'),
                
                # Detailed correlation table
                html.H3("Detailed Correlation Analysis", style={'marginTop': 30}),
                html.Div(id='correlation-table'),
                
                # Scatter plot: Best correlation
                html.H3("Regression Plot: Best Leading Indicator", style={'marginTop': 30}),
                dcc.Graph(id='best-correlation-scatter')
                
            ], style={'padding': 20})
        ])
    ]),
    
    # Footer
    html.Div([
        html.P("💡 Dashboard powered by Plotly Dash • Data: Supabase (YouTube) + Yahoo Finance (Stock)",
               style={'textAlign': 'center', 'color': '#999', 'fontSize': 12, 'marginTop': 40})
    ])
    
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto'})

# ============================================================
# CALLBACKS
# ============================================================

# Callback 1: Monthly Data Table
@app.callback(
    Output('monthly-table-container', 'children'),
    [Input('label-selector', 'value'),
     Input('metric-selector', 'value')]
)
def update_monthly_table(label, metrics):
    df = fetch_monthly_data(label)
    
    if df.empty:
        return html.P("No data available", style={'color': 'red'})
    
    # Select columns based on user choice
    base_cols = ['month', 'label']
    
    if 'subs' in metrics:
        base_cols.extend(['total_subscribers', 'prev_month_subs', 'mom_subs_growth'])
    if 'views' in metrics:
        base_cols.extend(['total_views', 'prev_month_views', 'mom_views_growth'])
    if 'videos' in metrics:
        base_cols.append('total_videos')
    if 'growth' in metrics:
        base_cols.extend(['mom_subs_growth_pct', 'mom_views_growth_pct'])
    
    # Filter columns that exist
    cols_to_show = [col for col in base_cols if col in df.columns]
    df_display = df[cols_to_show].copy()
    
    # Format numbers
    for col in df_display.columns:
        if 'pct' in col:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
        elif col in ['total_subscribers', 'total_views', 'prev_month_subs', 'prev_month_views', 
                     'mom_subs_growth', 'mom_views_growth', 'total_videos']:
            df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    
    # Create DataTable
    table = dash_table.DataTable(
        data=df_display.to_dict('records'),
        columns=[{'name': col.replace('_', ' ').title(), 'id': col} for col in df_display.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': 14
        },
        style_header={
            'backgroundColor': '#1f77b4',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            },
            {
                'if': {
                    'column_id': 'mom_subs_growth_pct',
                    'filter_query': '{mom_subs_growth_pct} > 0'
                },
                'color': 'green',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'column_id': 'mom_subs_growth_pct',
                    'filter_query': '{mom_subs_growth_pct} < 0'
                },
                'color': 'red',
                'fontWeight': 'bold'
            }
        ],
        page_size=20
    )
    
    return table

# Callback 2: Daily Views Chart
@app.callback(
    Output('daily-views-chart', 'figure'),
    [Input('time-range', 'value'),
     Input('scale-type', 'value')]
)
def update_daily_views_chart(days, scale):
    df = fetch_daily_views()
    
    # Filter by date range
    cutoff_date = df['date'].max() - timedelta(days=days)
    df = df[df['date'] >= cutoff_date]
    
    # Calculate daily view change
    df['daily_views'] = df['total_views'].diff()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['daily_views'],
        mode='lines',
        name='Daily View Change',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title="Daily View Changes (Tips Channels)",
        xaxis_title="Date",
        yaxis_title="Daily Views Added",
        yaxis_type='log' if scale == 'log' else 'linear',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Callback 3: Monthly Views Chart
@app.callback(
    Output('monthly-views-chart', 'figure'),
    [Input('scale-type', 'value')]
)
def update_monthly_views_chart(scale):
    tips_df = fetch_monthly_data('Tips')
    saregama_df = fetch_monthly_data('Saregama')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tips_df['month'],
        y=tips_df['total_views'],
        mode='lines+markers',
        name='Tips',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=saregama_df['month'],
        y=saregama_df['total_views'],
        mode='lines+markers',
        name='Saregama',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Monthly Total Views: Tips vs Saregama",
        xaxis_title="Month",
        yaxis_title="Total Views",
        yaxis_type='log' if scale == 'log' else 'linear',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Callback 4: Monthly Subscribers Chart
@app.callback(
    Output('monthly-subs-chart', 'figure'),
    [Input('scale-type', 'value')]
)
def update_monthly_subs_chart(scale):
    tips_df = fetch_monthly_data('Tips')
    saregama_df = fetch_monthly_data('Saregama')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tips_df['month'],
        y=tips_df['total_subscribers'],
        mode='lines+markers',
        name='Tips',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=saregama_df['month'],
        y=saregama_df['total_subscribers'],
        mode='lines+markers',
        name='Saregama',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Monthly Total Subscribers: Tips vs Saregama",
        xaxis_title="Month",
        yaxis_title="Total Subscribers",
        yaxis_type='log' if scale == 'log' else 'linear',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Callback 5: 45-Day Rolling Average Chart
@app.callback(
    [Output('rolling-average-chart', 'figure'),
     Output('rolling-stats', 'children')],
    [Input('main-tabs', 'value')]
)
def update_rolling_chart(tab):
    if tab != 'tab-rolling':
        return {}, ""
    
    df = fetch_daily_views()
    df = calculate_rolling_correlation(df, window=45)
    
    # Create dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Views rolling average
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['views_rolling_avg'],
            name='45-Day Avg Daily Views',
            line=dict(color='#1f77b4', width=3)
        ),
        secondary_y=False
    )
    
    # Stock price rolling average
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['stock_rolling_avg'],
            name='45-Day Avg Stock Price',
            line=dict(color='#ff7f0e', width=3)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Daily Views (45-Day Avg)", secondary_y=False)
    fig.update_yaxes(title_text="Stock Price ₹ (45-Day Avg)", secondary_y=True)
    
    fig.update_layout(
        title="45-Day Rolling Average: Daily Views vs Stock Price",
        hovermode='x unified',
        height=600
    )
    
    # Calculate correlation
    corr_data = df[['views_rolling_avg', 'stock_rolling_avg']].dropna()
    if len(corr_data) > 30:
        corr, p_value = pearsonr(corr_data['views_rolling_avg'], corr_data['stock_rolling_avg'])
        
        stats = html.Div([
            html.H3("Correlation Statistics"),
            html.P(f"📊 Pearson Correlation: {corr:.4f}", style={'fontSize': 18, 'fontWeight': 'bold'}),
            html.P(f"📈 P-Value: {p_value:.6f}", style={'fontSize': 16}),
            html.P(
                f"✅ {'Statistically Significant' if p_value < 0.05 else '⚠️ Not Statistically Significant'} (α = 0.05)",
                style={'fontSize': 16, 'color': 'green' if p_value < 0.05 else 'orange'}
            ),
            html.P(f"📅 Data points: {len(corr_data)}", style={'fontSize': 14, 'color': '#666'})
        ])
    else:
        stats = html.P("Insufficient data for correlation analysis")
    
    return fig, stats

# Callback 6: Correlation Analysis
@app.callback(
    [Output('best-correlation-summary', 'children'),
     Output('lag-correlation-heatmap', 'figure'),
     Output('correlation-table', 'children'),
     Output('best-correlation-scatter', 'figure')],
    [Input('main-tabs', 'value')]
)
def update_correlation_analysis(tab):
    if tab != 'tab-correlation':
        return "", {}, "", {}
    
    # Fetch data
    df = fetch_unified_data(days=1095)  # 3 years
    
    # Find best correlations
    corr_results = find_best_correlation(df, max_lag=60)
    
    if corr_results.empty:
        return html.P("No correlation data available"), {}, "", {}
    
    # Best correlation summary
    best = corr_results.iloc[0]
    
    summary = html.Div([
        html.H3("🎯 Best Leading Indicator Found!", style={'color': '#2ca02c'}),
        html.P(f"Metric: {best['metric']}", style={'fontSize': 20, 'fontWeight': 'bold'}),
        html.P(f"Optimal Lag: {int(best['lag_days'])} days", style={'fontSize': 18}),
        html.P(f"Pearson Correlation: {best['pearson_corr']:.4f}", style={'fontSize': 18}),
        html.P(f"P-Value: {best['pearson_p_value']:.6f}", style={'fontSize': 16}),
        html.P(
            f"{'✅ Statistically Significant' if best['is_significant'] else '⚠️ Not Statistically Significant'}",
            style={'fontSize': 16, 'color': 'green' if best['is_significant'] else 'orange', 'fontWeight': 'bold'}
        ),
        html.Hr(),
        html.P(f"💡 Interpretation: {best['metric']} from {int(best['lag_days'])} days ago has a "
               f"{'positive' if best['pearson_corr'] > 0 else 'negative'} correlation of "
               f"{abs(best['pearson_corr']):.4f} with today's stock price movement.",
               style={'fontSize': 14, 'fontStyle': 'italic'})
    ])
    
    # Heatmap: Correlation at different lags
    metrics = corr_results['metric'].unique()
    heatmap_data = []
    
    for metric in metrics[:3]:  # Top 3 metrics
        metric_data = corr_results[corr_results['metric'] == metric].sort_values('lag_days')
        heatmap_data.append(
            go.Scatter(
                x=metric_data['lag_days'],
                y=metric_data['pearson_corr'],
                mode='lines+markers',
                name=metric,
                line=dict(width=3),
                marker=dict(size=8)
            )
        )
    
    heatmap_fig = go.Figure(data=heatmap_data)
    heatmap_fig.update_layout(
        title="Correlation Strength at Different Time Lags",
        xaxis_title="Lag (Days)",
        yaxis_title="Pearson Correlation Coefficient",
        hovermode='x unified',
        height=500
    )
    heatmap_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    heatmap_fig.add_hline(y=0.3, line_dash="dot", line_color="green", annotation_text="Moderate")
    heatmap_fig.add_hline(y=-0.3, line_dash="dot", line_color="red")
    
    # Correlation table (top 20)
    corr_display = corr_results.head(20).copy()
    corr_display['pearson_corr'] = corr_display['pearson_corr'].apply(lambda x: f"{x:.4f}")
    corr_display['pearson_p_value'] = corr_display['pearson_p_value'].apply(lambda x: f"{x:.6f}")
    corr_display['spearman_corr'] = corr_display['spearman_corr'].apply(lambda x: f"{x:.4f}")
    corr_display['lag_days'] = corr_display['lag_days'].astype(int)
    corr_display['is_significant'] = corr_display['is_significant'].apply(lambda x: '✅ Yes' if x else '❌ No')
    
    corr_table = dash_table.DataTable(
        data=corr_display.to_dict('records'),
        columns=[
            {'name': 'YouTube Metric', 'id': 'metric'},
            {'name': 'Lag (Days)', 'id': 'lag_days'},
            {'name': 'Pearson r', 'id': 'pearson_corr'},
            {'name': 'P-Value', 'id': 'pearson_p_value'},
            {'name': 'Spearman ρ', 'id': 'spearman_corr'},
            {'name': 'Significant?', 'id': 'is_significant'}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': 14
        },
        style_header={
            'backgroundColor': '#1f77b4',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            },
            {
                'if': {
                    'column_id': 'is_significant',
                    'filter_query': '{is_significant} = "✅ Yes"'
                },
                'color': 'green',
                'fontWeight': 'bold'
            }
        ]
    )
    
    # Scatter plot: Best correlation
    best_metric_col = {
        'Daily Views Change': 'daily_views_change',
        'Weekly Subscriber Growth %': 'weekly_subs_growth_pct',
        'Monthly Subscriber Growth %': 'monthly_subs_growth_pct'
    }.get(best['metric'], 'weekly_subs_growth_pct')
    
    best_lag = int(best['lag_days'])
    
    scatter_df = df[[best_metric_col, 'daily_return_pct']].copy()
    scatter_df['metric_lagged'] = scatter_df[best_metric_col].shift(best_lag)
    scatter_df = scatter_df.dropna()
    
    scatter_fig = go.Figure()
    
    scatter_fig.add_trace(go.Scatter(
        x=scatter_df['metric_lagged'],
        y=scatter_df['daily_return_pct'],
        mode='markers',
        name='Data Points',
        marker=dict(color='#1f77b4', size=8, opacity=0.6)
    ))
    
    # Regression line
    if len(scatter_df) > 10:
        z = np.polyfit(scatter_df['metric_lagged'], scatter_df['daily_return_pct'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(scatter_df['metric_lagged'].min(), scatter_df['metric_lagged'].max(), 100)
        
        scatter_fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name='Regression Line',
            line=dict(color='#ff7f0e', width=3)
        ))
    
    scatter_fig.update_layout(
        title=f"{best['metric']} ({best_lag}-Day Lag) vs Stock Daily Return",
        xaxis_title=f"{best['metric']} (Lagged {best_lag} days)",
        yaxis_title="Stock Daily Return (%)",
        height=500
    )
    
    return summary, heatmap_fig, corr_table, scatter_fig

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    # For local testing
    # app.run_server(debug=True, host='0.0.0.0', port=8050)
    
    # For production (Render)
    app.run_server(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 10000)))
