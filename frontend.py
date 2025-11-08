import streamlit as st
import requests
import json
from datetime import datetime
import time
import pandas as pd
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64

# Configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="🏠 AI Real Estate Search",
        page_icon="🏠",
        layout="wide"
    )
    
    # Title and description
    st.title("🤖 ReAltoR Search Engine")
    st.markdown("*Multi-agent system with intelligent property search, memory, and reporting*")
    
    # Check system status
    if not check_system_status():
        return
    
    # Sidebar for additional features
    create_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "🧠 Memory & History", "📊 Reports", "⚙️ Settings"])
    
    with tab1:
        search_interface()
    
    with tab2:
        memory_interface()
    
    with tab3:
        report_interface()
    
    with tab4:
        settings_interface()

def check_system_status():
    """Check API and agent status"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if check_api_connection():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.code("python backend_fixed.py")
            st.stop()
    
    with col2:
        agents_status = get_agents_status()
        if agents_status:
            # Use the active_agents count from the API response
            active_count = agents_status.get("active_agents", 0)
            total_count = agents_status.get("total_agents", 0)
            st.info(f"🤖 {active_count}/{total_count} Agents Active")
        else:
            st.warning("⚠️ Agents Status Unknown")
    
    with col3:
        health = get_health_status()
        if health and health.get("orchestrator_available"):
            st.success("🧠 LangGraph Ready")
        else:
            st.warning("⚠️ Individual Agents")
    
    return True

def search_interface():
    """Main search interface"""
    st.subheader("🔍 Property Search")
    
    # Example queries
    st.markdown("**💡 Try these example queries:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏢 Studio ", use_container_width=True):
            st.session_state.search_query = "Studio"
    with col2:
        if st.button("🏡 Villa in Nagpur", use_container_width=True):
            st.session_state.search_query = "villa in Nagpur"
    with col3:
        if st.button("🔨 Renovation Cost", use_container_width=True):
            st.session_state.search_query = "renovation cost for 3BHK apartment in Mumbai"
    with col4:
        if st.button("🏠 Houses under 50L", use_container_width=True):
            st.session_state.search_query = "house under 50 lakhs"
    
    st.markdown("---")
    
    # Main search box
    search_query = st.text_area(
        "🔍 Enter your property search query:",
        value=st.session_state.get("search_query", ""),
        placeholder="Examples:\n• Find 2BHK apartments in Mumbai under 80 lakhs\n• What is the renovation cost for a 1200 sqft house?\n• Show me villas in Bangalore with parking\n• Properties in Delhi with 3 bedrooms",
        height=120
    )
    
    # Search button
    if st.button("🚀 Search with AI Agents", type="primary", use_container_width=True):
        if search_query.strip():
            search_with_agents(search_query)
        else:
            st.warning("Please enter a search query")
    
    # AI Visual Report Generator - Prominently placed on main page
    st.markdown("---")
    st.subheader("🤖 AI-Powered Visual Report Generator")
    st.markdown("*Generate personalized reports with charts and visualizations based on your preferences and search history*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**🎯 Generate intelligent reports with:**")
        st.markdown("• 📊 Visual charts based on your search patterns")
        st.markdown("• 🏠 Property recommendations matching your preferences") 
        st.markdown("• 📈 Market trends analysis for your preferred locations")
        st.markdown("• 💰 Budget analysis and investment insights")
        st.markdown("• 🧠 Personalized insights from your search memory")
    
    with col2:
        if st.button("🤖 Generate AI Visual Report", type="secondary", use_container_width=True, key="main_page_report"):
            generate_intelligent_visual_report()
        
        # Quick stats preview
        search_count = len(st.session_state.get("search_history", []))
        preferences_set = bool(st.session_state.get("user_preferences", {}))
        
        if search_count > 0:
            st.metric("🔍 Searches Done", search_count)
        if preferences_set:
            st.success("✅ Preferences Set")
        else:
            st.info("💡 Set preferences in Memory tab")

def search_with_agents(query: str):
    """Execute search using AI agents"""
    
    # Create status container
    status_container = st.container()
    with status_container:
        st.info(f"🤖 AI Agents processing: '{query}'")
        
        # Progress simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            "🧠 Analyzing query intent...",
            "🗃️ Searching property database...", 
            "📚 Checking knowledge base...",
            "🔨 Processing estimates...",
            "📊 Synthesizing results..."
        ]
        
        for i, stage in enumerate(stages):
            progress_bar.progress((i + 1) * 20)
            status_text.text(stage)
            time.sleep(0.4)
    
    # Make API call
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": query},
            timeout=30
        )
        
        # Clear progress
        status_container.empty()
        
        if response.status_code == 200:
            result = response.json()
            display_results(result, query)
        else:
            st.error(f"❌ Search failed: {response.status_code}")
            st.code(response.text)
            
    except requests.exceptions.RequestException as e:
        status_container.empty()
        st.error(f"❌ Connection error: {str(e)}")
        st.info("💡 Make sure the backend is running: `python backend_fixed.py`")

def display_results(result: dict, query: str):
    """Display search results"""
    
    if not result.get("success"):
        st.error("❌ Search failed")
        if result.get("response_text"):
            st.error(result["response_text"])
        return
    
    properties = result.get("properties", [])
    agents_used = result.get("agents_used", [])
    execution_time = result.get("execution_time", 0)
    agent_details = result.get("agent_details", {})
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏠 Properties", len(properties))
    with col2:
        st.metric("🤖 Agents", len(agents_used))
    with col3:
        st.metric("⏱️ Time", f"{execution_time:.2f}s")
    with col4:
        if properties and any(p.get("price", 0) > 0 for p in properties):
            prices = [p.get("price", 0) for p in properties if p.get("price", 0) > 0]
            avg_price = sum(prices) // len(prices) if prices else 0
            st.metric("💰 Avg Price", f"₹{avg_price:,}" if avg_price > 0 else "Estimates")
        else:
            st.metric("💰 Results", "Found")
    
    # AI Response
    st.subheader("🎯 AI Agent Response")
    response_text = result.get("response_text", "")
    st.success(response_text)
    
    # Agent execution details (expandable)
    with st.expander("🤖 Agent Execution Details"):
        st.markdown("**Agents Used:**")
        for agent in agents_used:
            st.markdown(f"• ✅ {agent.replace('_', ' ').title()} Agent")
        
        if agent_details:
            st.markdown("**Agent Output Details:**")
            for agent_name, details in agent_details.items():
                st.markdown(f"**{agent_name.title()} Agent:**")
                if isinstance(details, dict):
                    # Show simplified view
                    if "intent" in details:
                        st.markdown(f"- Intent: {details['intent']}")
                    if "entities" in details:
                        st.markdown(f"- Extracted entities: {details.get('extracted_entities', {})}")
                    if "properties" in details:
                        st.markdown(f"- Properties found: {len(details['properties'])}")
                st.markdown("---")
    
    # Display properties
    if properties:
        st.subheader(f"📋 Property Results ({len(properties)})")
        
        for i, prop in enumerate(properties, 1):
            with st.container():
                
                # Check if renovation estimate
                if prop.get("source") == "Renovation_Agent":
                    display_renovation_result(prop, i)
                else:
                    display_property_result(prop, i)
                
                if i < len(properties):
                    st.markdown("---")
    
    # Save to history
    save_search_history(query, result)

def display_property_result(prop: dict, index: int):
    """Display property result"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        title = prop.get("title", "Property")
        location = prop.get("location", "Location not specified")
        
        st.markdown(f"**{index}. 🏠 {title}**")
        st.markdown(f"📍 {location}")
        
        # Property details
        details = []
        if prop.get("num_rooms"):
            details.append(f"🏠 {prop['num_rooms']} rooms")
        if prop.get("property_size_sqft"):
            details.append(f"📐 {prop['property_size_sqft']} sqft")
        if prop.get("num_bathrooms"):
            details.append(f"🚿 {prop['num_bathrooms']} bathrooms")
        if prop.get("parking_spaces"):
            details.append(f"🚗 {prop['parking_spaces']} parking")
        
        if details:
            st.markdown(" • ".join(details))
        
        if prop.get("description"):
            st.markdown(f"*{prop['description'][:150]}...*")
    
    with col2:
        price = prop.get("price", 0)
        if price > 0:
            st.markdown(f"**💰 ₹{price:,}**")
            
            if prop.get("property_size_sqft", 0) > 0:
                price_per_sqft = price / prop["property_size_sqft"]
                st.markdown(f"₹{price_per_sqft:,.0f}/sqft")
        else:
            st.markdown("**💰 Price on request**")
        
        # Source
        source = prop.get("source", "Database")
        if source == "RAG_Agent":
            st.markdown("🧠 *AI Knowledge*")
            if prop.get("relevance_score"):
                st.markdown(f"Score: {prop['relevance_score']:.2f}")
        else:
            st.markdown("🗃️ *Database*")
        
        if st.button("📋 Details", key=f"details_{index}"):
            st.json(prop)

def display_renovation_result(prop: dict, index: int):
    """Display renovation estimate"""
    st.markdown(f"**{index}. 🔨 {prop.get('title', 'Renovation Estimate')}**")
    
    renovation_details = prop.get("renovation_details", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_cost = renovation_details.get("total_cost", 0)
        st.metric("💰 Total Cost", f"₹{total_cost:,}")
    
    with col2:
        timeline = renovation_details.get("timeline", "Not specified")
        st.markdown(f"**⏱️ Timeline:** {timeline}")
    
    with col3:
        if st.button("📊 Breakdown", key=f"breakdown_{index}"):
            breakdown = renovation_details.get("breakdown", {})
            if breakdown:
                st.json(breakdown)
            else:
                st.info("Detailed breakdown not available")

# Utility functions
def check_api_connection():
    """Check API connection"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_agents_status():
    """Get agents status"""
    try:
        response = requests.get(f"{API_BASE_URL}/agents/status", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_health_status():
    """Get health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def save_search_history(query: str, result: dict):
    """Save search history"""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    st.session_state.search_history.append({
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "results_count": len(result.get("properties", [])),
        "agents_used": result.get("agents_used", [])
    })
    
    # Keep last 10
    st.session_state.search_history = st.session_state.search_history[-10:]

def create_sidebar():
    """Create sidebar with user preferences and quick actions"""
    with st.sidebar:
        st.header("🏠 Quick Actions")
        
        # User ID for memory
        user_id = st.text_input("👤 User ID (for memory)", value=st.session_state.get("user_id", "default_user"))
        st.session_state.user_id = user_id
        
        # Quick search templates
        st.subheader("⚡ Quick Searches")
        if st.button("🏢 Apartments in Hyderabad", use_container_width=True):
            st.session_state.search_query = " apartments in Hyderabad"
        
        if st.button("🏡 Villas in Bangalore", use_container_width=True):
            st.session_state.search_query = "3BHK villa in Bangalore with garden"
        
        if st.button("🔨 Renovation Calculator", use_container_width=True):
            st.session_state.search_query = "renovation cost for 1200 sqft apartment"
        
        if st.button("📊 Market Analysis", use_container_width=True):
            st.session_state.search_query = "market analysis for properties in Hyderabad"
        
        # Memory stats
        st.subheader("🧠 Memory Stats")
        history_count = len(st.session_state.get("search_history", []))
        st.metric("Search History", history_count)
        
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.success("History cleared!")

def memory_interface():
    """Memory and conversation history interface"""
    st.header("🧠 Memory & Conversation History")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Search History")
        
        if "search_history" in st.session_state and st.session_state.search_history:
            for i, search in enumerate(reversed(st.session_state.search_history[-10:])):
                with st.expander(f"🔍 {search['query'][:50]}..." if len(search['query']) > 50 else search['query']):
                    st.write(f"**Time:** {search['timestamp']}")
                    st.write(f"**Results:** {search['results_count']} properties")
                    st.write(f"**Agents:** {', '.join(search['agents_used'])}")
                    
                    if st.button(f"🔄 Repeat Search", key=f"repeat_{i}"):
                        st.session_state.search_query = search['query']
                        st.experimental_rerun()
        else:
            st.info("No search history yet. Start searching to build your memory!")
    
    with col2:
        st.subheader("👤 User Preferences")
        
        # User preferences (stored in session)
        if "user_preferences" not in st.session_state:
            st.session_state.user_preferences = {
                "preferred_locations": [],
                "budget_range": "50-100 lakhs",
                "property_types": [],
                "bhk_preference": "2BHK"
            }
        
        prefs = st.session_state.user_preferences
        
        # Budget preference
        budget_options = ["Under 50 lakhs", "50-100 lakhs", "1-2 crores", "Above 2 crores"]
        prefs["budget_range"] = st.selectbox("💰 Budget Range", budget_options, 
                                           index=budget_options.index(prefs["budget_range"]))
        
        # Location preference
        location_options = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata"]
        prefs["preferred_locations"] = st.multiselect("📍 Preferred Locations", 
                                                     location_options,
                                                     default=prefs["preferred_locations"])
        
        # Property type preference
        property_options = ["Apartment", "Villa", "House", "Studio", "Plot"]
        prefs["property_types"] = st.multiselect("🏠 Property Types",
                                                property_options,
                                                default=prefs["property_types"])
        
        # BHK preference
        bhk_options = ["Studio", "1BHK", "2BHK", "3BHK", "4BHK", "5BHK+"]
        prefs["bhk_preference"] = st.selectbox("🏢 BHK Preference", bhk_options,
                                             index=bhk_options.index(prefs["bhk_preference"]))
        
        if st.button("💾 Save Preferences"):
            st.success("Preferences saved!")
        
        # Export User Memory Section
        st.markdown("---")
        st.subheader("📄 Export Your Data")
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("📥 Download Memory PDF", use_container_width=True):
                generate_user_memory_pdf()
        
        with col4:
            if st.button("📊 Export Search History CSV", use_container_width=True):
                export_search_history_csv()

def report_interface():
    """Report generation interface"""
    st.header("📊 Report Generation")
    
    st.markdown("Generate comprehensive property reports and market analysis")
    
    # Report type selection
    report_type = st.selectbox("📋 Select Report Type", [
        "Property Investment Analysis",
        "Market Trends Report", 
        "Neighborhood Comparison",
        "Property Valuation Report",
        "Rental Yield Analysis"
    ])
    
    # Report parameters
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., Bangalore, Mumbai")
        property_type = st.selectbox("🏠 Property Type", ["All", "Apartment", "Villa", "House", "Studio"])
    
    with col2:
        budget_min = st.number_input("💰 Min Budget (Lakhs)", min_value=0, value=50)
        budget_max = st.number_input("💰 Max Budget (Lakhs)", min_value=0, value=200)
    
    # Additional parameters based on report type
    if report_type == "Property Investment Analysis":
        investment_horizon = st.selectbox("⏰ Investment Horizon", ["1 Year", "3 Years", "5 Years", "10+ Years"])
        risk_appetite = st.selectbox("📈 Risk Appetite", ["Conservative", "Moderate", "Aggressive"])
    
    elif report_type == "Neighborhood Comparison":
        neighborhoods = st.text_area("🏘️ Neighborhoods to Compare", 
                                   placeholder="Enter neighborhoods separated by commas\ne.g., Koramangala, Whitefield, HSR Layout")
    
    # Generate report button
    if st.button("📊 Generate Report", type="primary", use_container_width=True):
        if location:
            generate_report(report_type, {
                "location": location,
                "property_type": property_type,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "report_type": report_type
            })
        else:
            st.warning("Please enter a location")
    
    # Recent reports
    st.subheader("📁 Recent Reports")
    
    if "generated_reports" not in st.session_state:
        st.session_state.generated_reports = []
    
    if st.session_state.generated_reports:
        for i, report in enumerate(st.session_state.generated_reports[-5:]):
            with st.expander(f"📄 {report['title']} - {report['timestamp'][:10]}"):
                st.markdown(report['content'])
                if st.download_button(
                    f"📥 Download",
                    data=report['content'],
                    file_name=f"{report['title'].replace(' ', '_')}.txt",
                    key=f"download_{i}"
                ):
                    st.success("Report downloaded!")
    else:
        st.info("No reports generated yet")
    
    # Intelligent Report Generator Section
    st.markdown("---")
    st.subheader("🤖 AI-Powered Visual Report Generator")
    st.markdown("*Generate personalized reports with charts and visualizations based on your preferences and search history*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**🎯 This intelligent report will include:**")
        st.markdown("• 📊 Visual charts based on your search patterns")
        st.markdown("• 🏠 Property recommendations matching your preferences") 
        st.markdown("• 📈 Market trends analysis for your preferred locations")
        st.markdown("• 💰 Budget analysis and investment insights")
        st.markdown("• 🧠 Personalized insights from your search memory")
    
    with col2:
        if st.button("🤖 Generate AI Visual Report", type="primary", use_container_width=True):
            generate_intelligent_visual_report()

def generate_intelligent_visual_report():
    """Generate intelligent visual report based on user preferences and memory"""
    try:
        with st.spinner("🤖 AI is analyzing your preferences and generating visual report..."):
            # Get user data
            preferences = st.session_state.get("user_preferences", {})
            search_history = st.session_state.get("search_history", [])
            
            # Create intelligent analysis
            st.subheader("🎯 Your Personalized AI Report")
            
            if not search_history:
                st.warning("⚠️ No search history found. Please perform some searches first to generate a meaningful report.")
                return
            
            # Generate visualizations
            generate_search_pattern_charts(search_history)
            generate_preference_analysis_charts(preferences)
            generate_market_insights(search_history, preferences)
            
            # Create downloadable report
            create_downloadable_visual_report(search_history, preferences)
            
    except Exception as e:
        st.error(f"❌ Failed to generate AI report: {str(e)}")

def generate_search_pattern_charts(search_history):
    """Generate charts showing user's search patterns"""
    st.subheader("📊 Your Search Patterns")
    
    if not search_history:
        st.info("No search data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(search_history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Search frequency over time
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_searches = df.groupby('date').size().reset_index()
            daily_searches.columns = ['Date', 'Searches']
            
            fig1 = px.line(daily_searches, x='Date', y='Searches', 
                          title='📈 Daily Search Activity',
                          color_discrete_sequence=['#1f77b4'])
            fig1.update_layout(height=300)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Results distribution
        if 'results_count' in df.columns:
            fig2 = px.histogram(df, x='results_count', 
                               title='📊 Search Results Distribution',
                               color_discrete_sequence=['#ff7f0e'])
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Top search terms
    if 'query' in df.columns:
        st.markdown("**🔥 Your Most Searched Terms:**")
        query_words = []
        for query in df['query']:
            words = str(query).lower().split()
            # Filter common words
            filtered_words = [w for w in words if w not in ['in', 'the', 'for', 'and', 'or', 'with', 'under', 'above']]
            query_words.extend(filtered_words)
        
        if query_words:
            word_freq = pd.Series(query_words).value_counts().head(10)
            fig3 = px.bar(x=word_freq.values, y=word_freq.index, 
                         orientation='h', title='🏷️ Most Searched Keywords',
                         color_discrete_sequence=['#2ca02c'])
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)

def generate_preference_analysis_charts(preferences):
    """Generate charts analyzing user preferences"""
    st.subheader("🎯 Your Preferences Analysis")
    
    if not preferences:
        st.info("💡 Set your preferences in the Memory & History tab to see personalized analysis!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Budget preference visualization
        if 'budget_range' in preferences:
            budget_info = preferences['budget_range']
            st.metric("💰 Preferred Budget", budget_info)
        
        # Location preferences
        if 'preferred_locations' in preferences and preferences['preferred_locations']:
            locations = preferences['preferred_locations']
            fig4 = px.pie(values=[1]*len(locations), names=locations, 
                         title='📍 Preferred Locations Distribution')
            st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        # Property type preferences
        if 'property_types' in preferences and preferences['property_types']:
            prop_types = preferences['property_types']
            fig5 = px.bar(x=prop_types, y=[1]*len(prop_types),
                         title='🏠 Preferred Property Types',
                         color_discrete_sequence=['#d62728'])
            fig5.update_layout(showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        
        # BHK preference
        if 'bhk_preference' in preferences:
            st.metric("🏢 Preferred BHK", preferences['bhk_preference'])

def generate_market_insights(search_history, preferences):
    """Generate market insights based on user data"""
    st.subheader("📈 Market Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔍 Total Searches", len(search_history), "This month")
    
    with col2:
        if search_history:
            avg_results = sum(s.get('results_count', 0) for s in search_history) / len(search_history)
            st.metric("📊 Avg Results", f"{avg_results:.1f}", "Per search")
    
    with col3:
        unique_queries = len(set(s.get('query', '') for s in search_history))
        st.metric("🎯 Unique Searches", unique_queries)
    
    # AI Recommendations
    st.markdown("**🤖 AI Recommendations Based on Your Activity:**")
    
    recommendations = []
    
    # Analyze search patterns for recommendations
    if search_history:
        # Check for budget patterns
        budget_mentions = sum(1 for s in search_history if any(word in s.get('query', '').lower() 
                             for word in ['budget', 'cost', 'price', 'cheap', 'expensive']))
        if budget_mentions > len(search_history) * 0.5:
            recommendations.append("💰 You seem budget-conscious. Consider looking at emerging areas for better value.")
    
    if preferences.get('preferred_locations'):
        recommendations.append(f"📍 Focus on {', '.join(preferences['preferred_locations'][:2])} for consistent results.")
    
    if not recommendations:
        recommendations = [
            "🎯 Continue exploring different areas to find the best deals",
            "📊 Your search patterns show good market research habits",
            "💡 Consider setting specific preferences for more targeted results"
        ]
    
    for i, rec in enumerate(recommendations[:3], 1):
        st.markdown(f"{i}. {rec}")

def create_downloadable_visual_report(search_history, preferences):
    """Create downloadable visual report"""
    st.subheader("📥 Download Your Visual Report")
    
    # Create comprehensive report content
    report_content = f"""
AI-POWERED VISUAL REPORT
========================
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

EXECUTIVE SUMMARY
----------------
• Total Searches Performed: {len(search_history)}
• Unique Search Queries: {len(set(s.get('query', '') for s in search_history))}
• Average Results per Search: {sum(s.get('results_count', 0) for s in search_history) / len(search_history) if search_history else 0:.1f}

USER PREFERENCES
---------------
{json.dumps(preferences, indent=2) if preferences else 'No preferences set yet'}

SEARCH ACTIVITY ANALYSIS
-----------------------
"""
    
    # Add search history details
    for i, search in enumerate(search_history[-10:], 1):
        report_content += f"""
{i}. Search Query: {search.get('query', 'Unknown')}
   Date: {search.get('timestamp', 'Unknown')[:10] if search.get('timestamp') else 'Unknown'}
   Results Found: {search.get('results_count', 0)}
   Agents Activated: {', '.join(search.get('agents_used', []))}
"""
    
    report_content += f"""

AI INSIGHTS & RECOMMENDATIONS
----------------------------
1. Your search patterns indicate a systematic approach to property research
2. Focus on your preferred locations for better targeted results
3. Consider diversifying your search criteria for more opportunities
4. Your activity level shows good market engagement

MARKET ANALYSIS
--------------
Based on your search history, you show interest in diverse property options.
Continue exploring different areas and price ranges to maximize opportunities.

Report generated by AI Real Estate Search Engine
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Create download button
    report_bytes = report_content.encode()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download Full Report (Text)",
            data=report_bytes,
            file_name=f"AI_Visual_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            type="primary"
        )
    
    with col2:
        # Create HTML version with charts
        html_report = create_html_report_with_charts(search_history, preferences)
        st.download_button(
            label="🌐 Download HTML Report",
            data=html_report.encode(),
            file_name=f"AI_Visual_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            type="secondary"
        )
    
    st.success("✅ Visual report generated! Your personalized analysis is ready for download.")
    
    # Show preview
    with st.expander("👀 Preview Report Content"):
        st.text(report_content)

def create_html_report_with_charts(search_history, preferences):
    """Create HTML report with embedded charts"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI-Powered Property Search Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .chart-container {{ margin: 20px 0; padding: 20px; background: #ffffff; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>🤖 AI-Powered Property Search Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    
    <h2>📊 Executive Summary</h2>
    <div class="metric">
        <strong>Total Searches:</strong> {len(search_history)}<br>
        <strong>Unique Queries:</strong> {len(set(s.get('query', '') for s in search_history))}<br>
        <strong>Average Results:</strong> {sum(s.get('results_count', 0) for s in search_history) / len(search_history) if search_history else 0:.1f}
    </div>
    
    <h2>🎯 User Preferences</h2>
    <div class="metric">
        {json.dumps(preferences, indent=2) if preferences else 'No preferences set yet'}
    </div>
    
    <h2>🔍 Recent Search Activity</h2>
"""
    
    # Add recent searches
    for i, search in enumerate(search_history[-5:], 1):
        html_content += f"""
    <div class="metric">
        <strong>Search {i}:</strong> {search.get('query', 'Unknown')}<br>
        <strong>Date:</strong> {search.get('timestamp', 'Unknown')[:10] if search.get('timestamp') else 'Unknown'}<br>
        <strong>Results:</strong> {search.get('results_count', 0)}
    </div>
"""
    
    html_content += """
    <h2>🤖 AI Recommendations</h2>
    <div class="metric">
        1. Continue systematic property research approach<br>
        2. Focus on preferred locations for better targeting<br>
        3. Consider diversifying search criteria<br>
        4. Maintain consistent market engagement
    </div>
    
    <footer style="margin-top: 50px; text-align: center; color: #7f8c8d;">
        <p>Generated by AI Real Estate Search Engine</p>
    </footer>
</body>
</html>
"""
    
    return html_content

def generate_report(report_type: str, params: dict):
    """Generate a report using the backend"""
    try:
        with st.spinner(f"🔄 Generating {report_type}..."):
            
            # Create report query
            query = f"Generate {report_type} for {params['property_type']} in {params['location']} "
            query += f"with budget between {params['budget_min']}-{params['budget_max']} lakhs"
            
            # Call backend
            response = requests.post(f"{API_BASE_URL}/search", json={"query": query}, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Create report
                report_content = f"# {report_type}\n\n"
                report_content += f"**Location:** {params['location']}\n"
                report_content += f"**Property Type:** {params['property_type']}\n"
                report_content += f"**Budget Range:** ₹{params['budget_min']}-{params['budget_max']} Lakhs\n"
                report_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                report_content += "---\n\n"
                report_content += result.get("response_text", "No content generated")
                
                # Save report
                if "generated_reports" not in st.session_state:
                    st.session_state.generated_reports = []
                
                st.session_state.generated_reports.append({
                    "title": f"{report_type} - {params['location']}",
                    "content": report_content,
                    "timestamp": datetime.now().isoformat(),
                    "params": params
                })
                
                # Display report
                st.success("✅ Report Generated Successfully!")
                st.markdown(report_content)
                
            else:
                st.error(f"Failed to generate report: {response.status_code}")
                
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

def settings_interface():
    """Settings and configuration interface"""
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("🔌 API Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        api_url = st.text_input("🌐 Backend API URL", value=API_BASE_URL)
        if st.button("🔗 Test Connection"):
            if check_api_connection():
                st.success("✅ API Connection Successful")
            else:
                st.error("❌ API Connection Failed")
    
    with col2:
        timeout = st.number_input("⏱️ Request Timeout (seconds)", min_value=5, max_value=60, value=15)
    
    # Agent Configuration
    st.subheader("🤖 Agent Settings")
    
    agent_settings = st.session_state.get("agent_settings", {
        "enable_parallel": True,
        "enable_fallback": True,
        "max_retries": 3,
        "preferred_agents": ["structured_data", "rag"]
    })
    
    agent_settings["enable_parallel"] = st.checkbox("⚡ Enable Parallel Agent Execution", 
                                                   value=agent_settings["enable_parallel"])
    
    agent_settings["enable_fallback"] = st.checkbox("🔄 Enable Agent Fallback", 
                                                   value=agent_settings["enable_fallback"])
    
    agent_settings["max_retries"] = st.number_input("🔁 Max Retry Attempts", 
                                                   min_value=1, max_value=5, 
                                                   value=agent_settings["max_retries"])
    
    # System Information
    st.subheader("📊 System Information")
    
    system_info = get_system_info()
    if system_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🤖 Total Agents", system_info.get("total_agents", "Unknown"))
        
        with col2:
            st.metric("✅ Active Agents", system_info.get("active_agents", "Unknown"))
        
        with col3:
            st.metric("🧠 Orchestrator", "Ready" if system_info.get("orchestrator_available") else "Offline")
    
    # Export/Import Settings
    st.subheader("📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Search History"):
            if "search_history" in st.session_state:
                import json
                history_json = json.dumps(st.session_state.search_history, indent=2)
                st.download_button(
                    "📁 Download History",
                    data=history_json,
                    file_name=f"search_history_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            if st.button("⚠️ Confirm Clear All Data", type="primary"):
                st.session_state.clear()
                st.success("All data cleared!")
                st.experimental_rerun()

def get_system_info():
    """Get system information"""
    try:
        agents_status = get_agents_status()
        health_status = get_health_status()
        
        return {
            "total_agents": agents_status.get("total_agents", 0) if agents_status else 0,
            "active_agents": agents_status.get("active_agents", 0) if agents_status else 0,
            "orchestrator_available": health_status.get("orchestrator_available", False) if health_status else False
        }
    except:
        return None

def generate_user_memory_pdf():
    """Generate PDF report of user memory and preferences"""
    try:
        with st.spinner("🔄 Generating your memory report..."):
            # Get user data
            preferences = st.session_state.get("user_preferences", {})
            search_history = st.session_state.get("search_history", [])
            
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
   Date: {search['timestamp'][:10] if 'timestamp' in search else 'Unknown'}
   Results Found: {search.get('results_count', 0)}
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

def export_search_history_csv():
    """Generate CSV export of search history"""
    try:
        search_history = st.session_state.get("search_history", [])
        
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