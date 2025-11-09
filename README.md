# 🏠 ReAltoR Search Engine

Multi-agent real estate search system with intelligent property matching, semantic search, and comprehensive reporting features.

##  Quick Start

### 1. Install Dependencies
```bash
# Install all required packages
conda env create -n realtor-ai-env --file requirements.yml
```

### 2. Environment Configuration (Required)

The project uses a `.env.template` file for secure API key management. **All features require API keys** - the application provides free tier options for testing.

#### Setup Environment Variables:
```bash
# Copy the template file
cp .env.example .env

# Edit .env with your actual API keys
nano .env  # or use any text editor
```

#### Required API Keys (Free Tier Available):
- **🤖 GROQ_API_KEY**: [Get free key](https://console.groq.com/keys) - 25,000 requests/day
- **🔍 SERPER_API_KEY**: [Get free key](https://serper.dev/) - 2,400 searches/month  
- **🌐 TAVILY_API_KEY**: [Get free key](https://tavily.com/) - 950 searches/month
- **🤗 HUGGINGFACE_API_KEY**: [Get free token](https://huggingface.co/settings/tokens) - No limits
- **📊 COHERE_API_KEY**: [Get free key](https://dashboard.cohere.ai/) - 100 requests/month


> **⚠️ Security Note**: Never commit your `.env` file to version control. It contains sensitive API keys and is already included in `.gitignore`.

### 3. Data Setup (First Time Only)

#### Create Structured Database:
```bash
# Run data ingestion to create structured database
python src/data_ingestion.py
```
This creates:
- `realestate.db` - SQLite database with 80+ properties
- Structured data for exact property searches

#### Create Unstructured Vector Database:
```bash
# Stay in src directory
cd src

# Create vector embeddings for semantic search
python data_ingestion.py --create-vectors
```
This creates:
- `vector_db/` directory with FAISS index
- Semantic embeddings for intelligent property matching

### 3. Start the Application

#### Start Backend (Terminal 1):
```bash
# From project root directory
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
Backend will run at: `http://localhost:8000`

#### Start Frontend (Terminal 2):
```bash
# From project root directory  
streamlit run frontend.py --server.port 8501
```
Frontend will run at: `http://localhost:8501`

## 🎯 Using the Application

1. **Open** `http://localhost:8501` in your browser
2. **Search** for properties using natural language
3. **Generate AI Reports** using the visual report generator
4. **Export** your search history and preferences as PDF

## 📊 Features

- **🤖 Multi-Agent System**: 8 specialized AI agents
- **🔍 Intelligent Search**: Both structured and semantic search
- **📈 Visual Reports**: AI-powered charts and analytics
- **🧠 Memory System**: Tracks preferences and search history
- **📄 PDF Export**: Download personalized reports

## 🛠️ Troubleshooting

### Backend won't start:
```bash
# Install uvicorn if missing
pip install uvicorn

# Check if port 8000 is free
lsof -i :8000
```

### Frontend won't start:
```bash
# Install streamlit if missing
pip install streamlit

# Use different port if 8501 is busy
streamlit run frontend.py --server.port 8502
```

### Missing database:
```bash
# Recreate databases
python data_ingestion.py
```

## 📁 Project Structure

```
Real-Estate-Search-Engine/
├── 🚀 Core Application Files
│   ├── backend.py                          # FastAPI server with all 8 agents
│   ├── frontend.py                         # Streamlit UI with AI report generator
│   └── api_gateway.py                      # Backend utility functions
│   
├── 🤖 Multi-Agent System
│   └── agents/
│       ├── query_router.py                 # Intent classification & routing
│       ├── structured_data_agent.py        # SQL database search
│       ├── rag_agent.py                    # Semantic search & fallback
│       ├── web_research_agent.py           # External market data
│       ├── report_generation_agent.py      # Charts & PDF reports
│       ├── renovation_estimation_agent.py  # Cost calculations
│       ├── planner_agent.py               # Task coordination
│       ├── memory_enhanced_planner.py     # Context-aware planning
│       ├── langgraph_orchestrator.py      # Multi-agent workflow
│       └── memory_component.py            # User preference learning
│       
├── 💾 Data Management
│   ├── data/                              # Property data files
│   ├── vector_db/                         # FAISS vector storage
│   │   ├── index.faiss                   # Vector embeddings index
│   │   ├── index.pkl                     # Metadata mappings
│   │   └── docstore.pkl                  # Document storage
│   ├── memory_storage/                    # Session persistence
│   └── real_estate_memory.db             # User memory & preferences
│   
├── 🛠️ Core Components
│   ├── src/                              # Data processing utilities
│   ├── components/                       # Reusable components
│   ├── models/                          # Data models & configurations
│   ├── utils/                           # Utility functions
│   └── config/                          # Configuration management
│   
├── 📋 Configuration & Environment
│   ├── .env                             # Environment variables (keep secure)
│   ├── .env.template                    # Environment template
│   ├── requirements.yml                 # Conda environment file
│   └── .gitignore                      # Git ignore patterns
│   
├── 📊 Documentation & Reports
│   ├── README.md                        # This comprehensive guide
│   ├── ReAltoR.pdf                      # Project documentation
│   
├── 🗂️ Assets & Resources
│   └── assets/                          # Static files & resources

```

### 🎯 Key Components Explained:

**🤖 agents/** - 8 specialized AI agents orchestrated through LangGraph for intelligent real estate search

**💾 Data Layer** - SQLite databases, FAISS vector storage, and persistent memory management

**🛠️ Processing** - Data ingestion, schema management, and utility functions

**📋 Environment** - Secure configuration with environment variables and conda setup

**📊 Documentation** - Comprehensive guides, architecture docs, and generated reports

### 🏗️ Architecture Overview:
- **Frontend**: Streamlit UI with AI visual report generator
- **Backend**: FastAPI with 8 specialized agents
- **Memory**: Persistent user preferences and session management
- **Data**: 80+ properties with vector search capabilities
- **Orchestration**: LangGraph for multi-agent workflows

## 🎉 You're Ready!