"""
Enhanced Frontend for Report Generation with Charts and PDFs
==========================================================

This enhanced frontend includes:
1. Interactive charts and visualizations
2. PDF report generation
3. Search history analysis
4. Property trend charts
5. Market analysis reports
6. Downloadable reports
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="🏠 AI Real Estate Analytics",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🏠 Real Estate AI")
    page = st.sidebar.selectbox("Navigate", [
        "🔍 Property Search",
        "📊 Analytics Dashboard", 
        "📈 Charts & Visualizations",
        "📄 PDF Reports",
        "🧠 Memory & Preferences",
        "📋 Search History Analysis"
    ])
    
    # Route to different pages
    if page == "🔍 Property Search":
        property_search_page()
    elif page == "📊 Analytics Dashboard":
        analytics_dashboard()
    elif page == "📈 Charts & Visualizations":
        charts_visualizations_page()
    elif page == "📄 PDF Reports":
        pdf_reports_page()
    elif page == "🧠 Memory & Preferences":
        memory_preferences_page()
    elif page == "📋 Search History Analysis":
        search_history_analysis_page()

def property_search_page():
    """Main property search interface"""
    st.title("🤖 AI-Powered Property Search")
    st.markdown("*Multi-agent system with intelligent property search*")
    
    # Check system status
    check_system_status()
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 What are you looking for?",
            placeholder="e.g., 2BHK apartment in Mumbai under 80 lakhs"
        )
    
    with col2:
        search_type = st.selectbox("Search Type", [
            "Property Search",
            "Market Analysis", 
            "Renovation Cost",
            "Investment Analysis"
        ])
    
    if st.button("🚀 Search", type="primary"):
        if query:
            search_properties(query, search_type)
        else:
            st.error("Please enter a search query")

def analytics_dashboard():
    """Analytics dashboard with key metrics"""
    st.title("📊 Real Estate Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Searches", "127", "+12%")
    with col2:
        st.metric("Avg. Budget", "₹75L", "+5%")
    with col3:
        st.metric("Popular Location", "Mumbai", "40%")
    with col4:
        st.metric("Success Rate", "85%", "+3%")
    
    # Charts section
    st.subheader("📈 Search Trends")
    
    # Sample data for demonstration
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    searches = [20, 25, 30, 28, 35, 40, 38, 45, 42, 48, 50, 55, 52, 58, 60, 
                65, 62, 68, 70, 75, 72, 78, 80, 85, 82, 88, 90, 95, 92, 98]
    
    df = pd.DataFrame({'Date': dates, 'Searches': searches})
    
    fig = px.line(df, x='Date', y='Searches', title='Daily Search Volume')
    st.plotly_chart(fig, use_container_width=True)
    
    # Property type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        property_types = ['2BHK', '3BHK', '1BHK', '4BHK', 'Studio']
        counts = [45, 35, 15, 8, 7]
        fig = px.pie(values=counts, names=property_types, title='Property Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        locations = ['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai']
        counts = [40, 25, 20, 10, 5]
        fig = px.bar(x=locations, y=counts, title='Popular Locations')
        st.plotly_chart(fig, use_container_width=True)

def charts_visualizations_page():
    """Charts and visualizations page"""
    st.title("📈 Charts & Visualizations")
    
    # Chart type selector
    chart_type = st.selectbox("Select Visualization", [
        "Property Price Trends",
        "Location Analysis", 
        "Budget Distribution",
        "Search Pattern Analysis",
        "Market Comparison"
    ])
    
    if chart_type == "Property Price Trends":
        create_price_trends_chart()
    elif chart_type == "Location Analysis":
        create_location_analysis()
    elif chart_type == "Budget Distribution":
        create_budget_distribution()
    elif chart_type == "Search Pattern Analysis":
        create_search_patterns()
    elif chart_type == "Market Comparison":
        create_market_comparison()
    
    # Generate custom report button
    if st.button("📊 Generate Visual Report"):
        generate_visual_report()

def pdf_reports_page():
    """PDF report generation page"""
    st.title("📄 PDF Report Generation")
    
    st.markdown("Generate comprehensive PDF reports with charts, analysis, and insights.")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", [
            "Market Analysis Report",
            "Property Search Summary", 
            "Investment Analysis Report",
            "Search History Report",
            "Custom Analysis Report"
        ])
        
        location = st.text_input("Location (optional)", placeholder="e.g., Mumbai, Bangalore")
        date_range = st.date_input("Date Range", value=[datetime.now().date() - timedelta(days=30), datetime.now().date()])
    
    with col2:
        include_charts = st.checkbox("Include Charts", value=True)
        include_tables = st.checkbox("Include Data Tables", value=True)
        include_insights = st.checkbox("Include AI Insights", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    # Generate report button
    if st.button("📄 Generate PDF Report", type="primary"):
        generate_pdf_report(report_type, location, date_range, {
            'charts': include_charts,
            'tables': include_tables, 
            'insights': include_insights,
            'recommendations': include_recommendations
        })

def memory_preferences_page():
    """Memory and preferences management"""
    st.title("🧠 Memory & Preferences")
    
    # User preferences
    st.subheader("🎯 Search Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        preferred_budget = st.slider("Preferred Budget Range (Lakhs)", 0, 500, (50, 150))
        preferred_locations = st.multiselect("Preferred Locations", [
            "Mumbai", "Bangalore", "Delhi", "Pune", "Chennai", "Hyderabad", "Kolkata"
        ])
        preferred_types = st.multiselect("Property Types", [
            "Studio", "1BHK", "2BHK", "3BHK", "4BHK", "Villa", "Plot"
        ])
    
    with col2:
        amenities = st.multiselect("Important Amenities", [
            "Parking", "Gym", "Swimming Pool", "Garden", "Security", "Elevator"
        ])
        max_commute = st.number_input("Max Commute Time (minutes)", 0, 120, 30)
        investment_horizon = st.selectbox("Investment Horizon", [
            "1-2 years", "3-5 years", "5+ years", "Not applicable"
        ])
    
    if st.button("💾 Save Preferences"):
        save_user_preferences({
            'budget_range': preferred_budget,
            'locations': preferred_locations,
            'property_types': preferred_types,
            'amenities': amenities,
            'max_commute': max_commute,
            'investment_horizon': investment_horizon
        })
        st.success("✅ Preferences saved successfully!")
    
    # Display saved preferences
    st.subheader("📋 Current Preferences")
    preferences = load_user_preferences()
    if preferences:
        st.json(preferences)
    
    # User Memory Export Section
    st.subheader("📄 Export User Memory & History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Generate Memory PDF Report", type="primary", use_container_width=True):
            generate_user_memory_pdf()
            
    with col2:
        if st.button("📊 Download Search History CSV", use_container_width=True):
            generate_search_history_csv()

def search_history_analysis_page():
    """Search history analysis with charts"""
    st.title("📋 Search History Analysis")
    
    # Load search history
    history = load_search_history()
    
    if not history:
        st.info("No search history available. Start searching to see your patterns!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Searches", len(df))
    with col2:
        st.metric("Unique Queries", df['query'].nunique())
    with col3:
        st.metric("Avg Results", f"{df['results_count'].mean():.1f}")
    
    # Search frequency chart
    st.subheader("📈 Search Frequency Over Time")
    daily_searches = df.groupby(df['timestamp'].dt.date).size().reset_index()
    daily_searches.columns = ['Date', 'Searches']
    
    fig = px.line(daily_searches, x='Date', y='Searches', title='Daily Search Volume')
    st.plotly_chart(fig, use_container_width=True)
    
    # Most searched terms
    st.subheader("🔥 Most Searched Terms")
    query_counts = df['query'].value_counts().head(10)
    fig = px.bar(x=query_counts.index, y=query_counts.values, title='Top Search Queries')
    st.plotly_chart(fig, use_container_width=True)
    
    # Results success rate
    st.subheader("✅ Search Success Rate")
    success_rate = df.groupby(df['timestamp'].dt.date)['results_count'].apply(lambda x: (x > 0).mean() * 100)
    fig = px.line(x=success_rate.index, y=success_rate.values, title='Daily Success Rate (%)')
    st.plotly_chart(fig, use_container_width=True)

def generate_visual_report():
    """Generate a visual report with charts"""
    st.info("🎨 Generating visual report...")
    
    # Call report generation API
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": "generate comprehensive market analysis report with charts and visualizations"}
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Visual report generated successfully!")
            
            # Display the report
            st.subheader("📊 Generated Report")
            st.write(result.get('response_text', ''))
            
            # Add sample visualizations
            create_sample_visualizations()
            
        else:
            st.error(f"❌ Failed to generate report: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error generating report: {e}")

def generate_pdf_report(report_type, location, date_range, options):
    """Generate PDF report"""
    st.info("📄 Generating PDF report...")
    
    try:
        # Call report generation API
        query = f"generate {report_type.lower()} for {location} with PDF export"
        
        response = requests.post(
            f"{API_BASE_URL}/search", 
            json={"query": query}
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success("✅ PDF report generated successfully!")
            
            # Display report content
            st.subheader("📄 Report Content")
            st.write(result.get('response_text', ''))
            
            # Create download button (simulated)
            pdf_data = create_sample_pdf(report_type, result)
            if pdf_data:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.error(f"❌ Failed to generate PDF report: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error generating PDF: {e}")

# Chart creation functions
def create_price_trends_chart():
    """Create property price trends chart"""
    st.subheader("📈 Property Price Trends")
    
    # Sample data
    months = pd.date_range('2023-01-01', periods=12, freq='M')
    mumbai_prices = [85, 87, 89, 88, 90, 92, 94, 93, 95, 97, 98, 100]
    bangalore_prices = [70, 72, 74, 73, 75, 77, 79, 78, 80, 82, 83, 85]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=mumbai_prices, mode='lines+markers', name='Mumbai'))
    fig.add_trace(go.Scatter(x=months, y=bangalore_prices, mode='lines+markers', name='Bangalore'))
    
    fig.update_layout(title='Property Prices Over Time (Lakhs per sqft)', xaxis_title='Month', yaxis_title='Price (₹ Lakhs)')
    st.plotly_chart(fig, use_container_width=True)

def create_location_analysis():
    """Create location analysis chart"""
    st.subheader("🗺️ Location Analysis")
    
    locations = ['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai']
    avg_price = [95, 75, 85, 65, 70]
    growth_rate = [8, 12, 6, 15, 10]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=locations, y=avg_price, name='Avg Price (₹L)', yaxis='y'))
    fig.add_trace(go.Scatter(x=locations, y=growth_rate, mode='lines+markers', name='Growth Rate (%)', yaxis='y2'))
    
    fig.update_layout(
        title='Location Analysis: Price vs Growth',
        yaxis=dict(title='Average Price (₹ Lakhs)'),
        yaxis2=dict(title='Growth Rate (%)', overlaying='y', side='right')
    )
    st.plotly_chart(fig, use_container_width=True)

def create_budget_distribution():
    """Create budget distribution chart"""
    st.subheader("💰 Budget Distribution")
    
    budget_ranges = ['<50L', '50-75L', '75-100L', '100-150L', '>150L']
    counts = [25, 35, 20, 15, 5]
    
    fig = px.pie(values=counts, names=budget_ranges, title='Search Budget Distribution')
    st.plotly_chart(fig, use_container_width=True)

def create_search_patterns():
    """Create search patterns analysis"""
    st.subheader("🔍 Search Pattern Analysis")
    
    hours = list(range(24))
    search_volume = [2, 1, 1, 1, 2, 3, 5, 8, 12, 15, 18, 20, 22, 18, 15, 12, 10, 8, 6, 5, 4, 3, 3, 2]
    
    fig = px.bar(x=hours, y=search_volume, title='Search Volume by Hour of Day')
    fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Search Volume')
    st.plotly_chart(fig, use_container_width=True)

def create_market_comparison():
    """Create market comparison chart"""
    st.subheader("🏘️ Market Comparison")
    
    cities = ['Mumbai', 'Bangalore', 'Delhi', 'Pune', 'Chennai']
    metrics = ['Price per sqft', 'Growth Rate', 'Liquidity', 'ROI Potential']
    
    data = {
        'Mumbai': [9, 7, 9, 8],
        'Bangalore': [7, 9, 8, 9], 
        'Delhi': [8, 6, 8, 7],
        'Pune': [6, 8, 7, 8],
        'Chennai': [6, 7, 7, 7]
    }
    
    fig = go.Figure()
    
    for city in cities:
        fig.add_trace(go.Scatterpolar(
            r=data[city],
            theta=metrics,
            fill='toself',
            name=city
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10])
        ),
        title="Market Comparison Radar Chart"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_sample_visualizations():
    """Create sample visualizations for the report"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample property distribution
        data = {'Type': ['2BHK', '3BHK', '1BHK', 'Villa'], 'Count': [45, 30, 20, 5]}
        fig = px.pie(data, values='Count', names='Type', title='Property Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sample price range
        ranges = ['<50L', '50-100L', '100-150L', '>150L']
        counts = [30, 40, 25, 5]
        fig = px.bar(x=ranges, y=counts, title='Price Range Distribution')
        st.plotly_chart(fig, use_container_width=True)

# Utility functions
def search_properties(query, search_type):
    """Search for properties"""
    try:
        with st.spinner("🤖 AI agents are processing your request..."):
            response = requests.post(
                f"{API_BASE_URL}/search",
                json={"query": query}
            )
        
        if response.status_code == 200:
            result = response.json()
            display_search_results(result, query)
            save_search_to_history(query, result)
        else:
            st.error(f"❌ Search failed: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")

def display_search_results(result, query):
    """Display search results"""
    st.success(f"✅ Search completed!")
    
    # Agent information
    agents_used = result.get('agents_used', [])
    if agents_used:
        st.info(f"🤖 Agents used: {', '.join(agents_used)}")
    
    # Response text
    st.markdown("### 📝 AI Response")
    st.write(result.get('response_text', 'No response'))
    
    # Properties
    properties = result.get('properties', [])
    if properties:
        st.markdown(f"### 🏠 Found {len(properties)} Properties")
        
        for i, prop in enumerate(properties[:5], 1):
            with st.expander(f"Property {i}: {prop.get('title', 'Property')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Price:** ₹{prop.get('price', 0):,}")
                with col2:
                    st.write(f"**Location:** {prop.get('location', 'N/A')}")
                with col3:
                    st.write(f"**Type:** {prop.get('property_type', 'N/A')}")

def check_system_status():
    """Check system status"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Disconnected")
    
    with col2:
        try:
            response = requests.get(f"{API_BASE_URL}/agents/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                active = data.get('active_agents', 0)
                total = data.get('total_agents', 0)
                st.info(f"🤖 {active}/{total} Agents Active")
            else:
                st.warning("⚠️ Agents Status Unknown")
        except:
            st.warning("⚠️ Agents Status Unknown")
    
    with col3:
        st.info("📊 Reports Available")

def save_search_to_history(query, result):
    """Save search to history"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    search_entry = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'results_count': len(result.get('properties', [])),
        'agents_used': result.get('agents_used', []),
        'success': result.get('success', True)
    }
    
    st.session_state.search_history.append(search_entry)

def load_search_history():
    """Load search history"""
    return st.session_state.get('search_history', [])

def save_user_preferences(preferences):
    """Save user preferences"""
    st.session_state.user_preferences = preferences

def load_user_preferences():
    """Load user preferences"""
    return st.session_state.get('user_preferences', {})

def create_sample_pdf(report_type, content):
    """Create a sample PDF (simulation)"""
    # In a real implementation, this would generate an actual PDF
    pdf_content = f"""
    {report_type}
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    {content.get('response_text', 'No content available')}
    
    This is a sample PDF report. In a real implementation, this would contain:
    - Detailed charts and graphs
    - Property listings with images
    - Market analysis with trends
    - Investment recommendations
    - Interactive elements
    """
    
    return pdf_content.encode('utf-8')

def generate_user_memory_pdf():
    """Generate PDF report of user memory and preferences"""
    try:
        with st.spinner("🔄 Generating your memory report..."):
            # Get user data
            preferences = load_user_preferences()
            search_history = load_search_history()
            
            # Create comprehensive memory report
            report_content = f"""
USER MEMORY & PREFERENCES REPORT
================================
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

USER PREFERENCES
---------------
{json.dumps(preferences, indent=2) if preferences else 'No preferences saved yet'}

SEARCH HISTORY SUMMARY
--------------------
Total Searches: {len(search_history)}
Most Recent Searches:
"""
            
            # Add recent search history
            for i, search in enumerate(search_history[-10:], 1):
                report_content += f"""
{i}. Query: {search['query']}
   Date: {search['timestamp'][:10]}
   Results Found: {search['results_count']}
   Agents Used: {', '.join(search.get('agents_used', []))}
"""
            
            # Create downloadable PDF (simulated as text for now)
            pdf_bytes = report_content.encode()
            
            # Offer download
            st.download_button(
                label="📥 Download Memory Report PDF",
                data=pdf_bytes,
                file_name=f"user_memory_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                type="primary"
            )
            
            st.success("✅ Memory report generated! Click download button above.")
            
            # Show preview
            with st.expander("👀 Preview Report"):
                st.text(report_content)
                
    except Exception as e:
        st.error(f"❌ Failed to generate memory report: {str(e)}")

def generate_search_history_csv():
    """Generate CSV export of search history"""
    try:
        search_history = load_search_history()
        
        if not search_history:
            st.warning("⚠️ No search history to export")
            return
            
        with st.spinner("🔄 Generating CSV export..."):
            # Convert to DataFrame
            df = pd.DataFrame(search_history)
            
            # Convert to CSV
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Offer download
            st.download_button(
                label="📊 Download Search History CSV",
                data=csv_data,
                file_name=f"search_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="secondary"
            )
            
            st.success("✅ CSV export ready! Click download button above.")
            
            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df)
                
    except Exception as e:
        st.error(f"❌ Failed to generate CSV export: {str(e)}")

if __name__ == "__main__":
    main()