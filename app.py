import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests 

# --- UI Configuration ---
st.set_page_config(page_title="PriceOptima AI", layout="wide", page_icon="🚀", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 20px; font-weight: 500;}
    .diag-card { padding: 25px; border-radius: 12px; margin-bottom: 15px; border: 1px solid; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .diag-title { font-size: 1.5rem; font-weight: 800; display: flex; align-items: center; gap: 10px; margin-bottom: 15px;}
    .diag-rec { font-weight: 800; font-size: 1.3rem; letter-spacing: 0.5px; margin-bottom: 10px;}
    .diag-reason { font-size: 1.05rem; opacity: 0.9; line-height: 1.5; }
    .card-star { background-color: #dcfce7; color: #14532d; border-color: #bbf7d0; } 
    .card-dog { background-color: #fee2e2; color: #7f1d1d; border-color: #fecaca; } 
    .card-bomb { background-color: #fce7f3; color: #831843; border-color: #fbcfe8; } 
    .card-gem { background-color: #dbeafe; color: #1e3a8a; border-color: #bfdbfe; } 
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e2e8f0; padding: 15px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
    st.markdown("## PriceOptima AI")
    st.markdown("Upload your market data to begin.")
    
    csv_template = "product_name,price,monthly_sales,comments\nPremium Headphones,249.99,450,Great sound but feels cheap. Battery life is amazing.\nBudget Earbuds,29.99,1200,Amazing value for money! Highly recommend.\nSmart Watch,199.99,800,Horrible battery. Screen cracked after a week.\nLuxury Case,89.99,40,Beautiful design but way too expensive. I love it though."
    st.download_button(label="📥 Download Template", data=csv_template, file_name="template.csv", mime="text/csv", use_container_width=True)
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

# --- Main App Area ---
if uploaded_file is None:
    # --- Landing Page ---
    st.markdown('<div class="main-header">PriceOptima AI 🚀</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enterprise Retail Intelligence Engine 📊 | Sentiment-Driven Price Optimization 💰</div>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown("""
    <div style='background-color: #ffffff; padding: 40px; border-radius: 15px; text-align: center; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 40px;'>
        <h2 style='color: #1e293b; margin-bottom: 15px;'>Turn Customer Reviews into Revenue 💸</h2>
        <p style='color: #64748b; font-size: 1.2rem; max-width: 700px; margin: 0 auto;'>
            Stop guessing your pricing strategy. Upload your product data on the sidebar to instantly unlock AI-driven, sentiment-backed pricing recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.markdown("### 📊 Analyze Data")
        st.write("Upload raw sales data alongside unstructured customer feedback.")
    with f2:
        st.image("https://images.unsplash.com/photo-1518186285589-2f7649de83e0?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.markdown("### 🧠 AI NLP Insights")
        st.write("Extract sentiment regarding price value and product quality instantly.")
    with f3:
        st.image("https://images.unsplash.com/photo-1579621970588-a35d0e7ab9b6?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.markdown("### 💰 Maximize Profits")
        st.write("Classify products into actionable quadrants to defend market share.")

else:
    API_URL = "https://priceoptima-api.onrender.com/analyze"
    
    # Changed the loading text to hide Render from the end-user
    with st.spinner("Processing intelligence data..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                api_data = response.json()
                results = api_data["data"]
                results_df = pd.DataFrame(results)
                
                st.sidebar.divider()
                st.sidebar.markdown("### 🔍 Filter Dashboard")
                filter_option = st.sidebar.radio(
                    "Select View:",
                    ["Portfolio Overview (All Products)", "Deep Dive (Single Product)"]
                )
                
                if filter_option == "Deep Dive (Single Product)":
                    selected_product = st.sidebar.selectbox("Select a Product to Analyze:", results_df['product_name'].unique())
                
                st.markdown('<div class="main-header">Intelligence Dashboard 🧠</div>', unsafe_allow_html=True)
                st.divider()

                if filter_option == "Portfolio Overview (All Products)":
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("📦 Total Products Analyzed", len(results_df))
                    k2.metric("⭐ Star Products (Increase Price)", len(results_df[results_df['diagnosis'] == 'Star']))
                    k3.metric("💣 Time Bombs (At Risk)", len(results_df[results_df['diagnosis'] == 'Ticking Time Bomb']))
                    k4.metric("📈 Avg Portfolio Sentiment", round(results_df['net_sentiment'].mean(), 2))
                    
                    st.markdown("### 💵 Portfolio Positioning")
                    chart_col1, chart_col2 = st.columns(2)
                    
                    color_map = {"Star": "#22c55e", "Dog": "#ef4444", "Ticking Time Bomb": "#ec4899", "Hidden Gem": "#3b82f6", "Unknown": "#cbd5e1"}
                    
                    with chart_col1:
                        fig_scatter = px.scatter(results_df, x='net_sentiment', y='monthly_sales', color='diagnosis',
                                                 hover_name='product_name',
                                                 title="Demand vs. Sentiment Matrix (Enterprise View)",
                                                 color_discrete_map=color_map,
                                                 labels={"net_sentiment": "Net Sentiment", "monthly_sales": "Volume"},
                                                 render_mode='webgl') 
                        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                    with chart_col2:
                        summary_df = results_df.groupby('diagnosis')['monthly_sales'].sum().reset_index()
                        fig_bar = px.bar(summary_df, x='diagnosis', y='monthly_sales', color='diagnosis', 
                                         title="Total Portfolio Volume by Strategy", 
                                         color_discrete_map=color_map)
                        fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Strategy Matrix", yaxis_title="Total Predicted Volume")
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    st.markdown("### 📑 Raw Intelligence Data")
                    st.dataframe(results_df[['product_name', 'current_price', 'monthly_sales', 'net_sentiment', 'recommendation']], use_container_width=True)

                else:
                    product_data = results_df[results_df['product_name'] == selected_product].iloc[0]
                    
                    st.markdown(f"### Deep Dive Analysis: **{product_data['product_name']}**")
                    
                    css_class, icon = "card-dog", "🐕"
                    if product_data['diagnosis'] == "Star": css_class, icon = "card-star", "⭐"
                    elif product_data['diagnosis'] == "Hidden Gem": css_class, icon = "card-gem", "💎"
                    elif product_data['diagnosis'] == "Ticking Time Bomb": css_class, icon = "card-bomb", "💣"
                    elif product_data['diagnosis'] == "Unknown": css_class, icon = "", "❓"
                    
                    card_html = f"""
                    <div class="diag-card {css_class}">
                        <div class="diag-title">{icon} {product_data['diagnosis']}</div>
                        <div class="diag-rec">ACTION: {product_data['recommendation']}</div>
                        <div class="diag-reason"><strong>AI Logic:</strong> {product_data['reasoning']}</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    col_metrics, col_gauge = st.columns([1, 1])
                    
                    '''with col_metrics:
                        st.markdown("#### 📊 Current Metrics")
                        m1, m2 = st.columns(2)
                        m1.metric("💰 Current Price", f"${product_data['current_price']}")
                        m2.metric("📦 Monthly Volume", product_data['monthly_sales'])
                        
                        st.markdown("#### 💡 Next Steps for Management")
                        if "INCREASE" in product_data['recommendation']:
                            st.info("Initiate a 5-10% price test. Monitor volume for 14 days to ensure elasticity holds.")
                        elif "DECREASE" in product_data['recommendation'] and "PHASE OUT" not in product_data['recommendation']:
                            st.info("Deploy promotional pricing. Highlight positive customer reviews in new marketing copy.")
                        elif "IMPROVE" in product_data['recommendation']:
                            st.warning("Halt price increases immediately. Route product feedback to QA/Manufacturing teams.")
                        else:
                            st.error("Begin inventory liquidation. Do not reorder this SKU.")'''
                    with col_metrics:
                        st.markdown("#### 📊 Current Metrics")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("💰 Current Price", f"${product_data['current_price']}")
                        m2.metric("🎯 A/B Test Target", f"${product_data['ab_test_price']}")
                        m3.metric("📦 Monthly Volume", product_data['monthly_sales'])
                        
                        st.markdown("#### 💡 Strategy & A/B Test Execution")
                        if "INCREASE" in product_data['recommendation']:
                            st.info(f"**A/B Test Warning:** Do not change prices globally. Test the new ${product_data['ab_test_price']} target on 20% of web traffic for 14 days to monitor elasticity.")
                        elif product_data['diagnosis'] == "Hidden Gem":
                            st.success(f"**Marketing Alert:** Drop price to ${product_data['ab_test_price']}. Clearly display **'10% OFF'** badges on the product page to entice conversion.")
                        elif "IMPROVE" in product_data['recommendation']:
                            st.warning(f"**Quality Alert:** Hold price at ${product_data['current_price']}. Routing feedback to QA to prevent demand collapse.")
                        else:
                            st.error(f"**Liquidation Alert:** Drop price to ${product_data['ab_test_price']}. Clearly display **'50% OFF - FINAL SALE'** to clear inventory.")
                            
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = product_data['net_sentiment'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Customer Sentiment Score", 'font': {'size': 20}},
                            gauge = {
                                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "rgba(0,0,0,0.3)"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [-1, -0.3], 'color': '#fee2e2'}, # Red
                                    {'range': [-0.3, 0.3], 'color': '#f1f5f9'}, # Neutral Grey
                                    {'range': [0.3, 1], 'color': '#dcfce7'}    # Green
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': product_data['net_sentiment']
                                }
                            }
                        ))
                        fig_gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to process request. Error details: {e}")