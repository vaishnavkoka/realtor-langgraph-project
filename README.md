# 🏠 Real Estate Search Engine

AI-powered multi-agent real estate search system with intelligent property matching, semantic search, and comprehensive reporting features.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install all required packages
conda env create -n case_study --file requirements.yml
```

### 2. Data Setup (First Time Only)

#### Create Structured Database:
```bash
# Navigate to src directory
cd src

# Run data ingestion to create structured database
python data_ingestion.py
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
cd src
python data_ingestion.py
```

## 📁 Project Structure

```
Real-Estate-Search-Engine/
├── backend.py                 # FastAPI server
├── frontend.py               # Streamlit interface
├── requirements.txt          # Dependencies
├── agents/                   # AI agents
├── src/                     # Data processing
│   └── data_ingestion.py    # Database creation
├── data/                    # Databases
│   └── realestate.db       # Property database
└── vector_db/              # Vector embeddings
    └── faiss_index.pkl     # FAISS index
```

## 🎉 You're Ready!

Your AI-powered Real Estate Search Engine is now running with both structured and unstructured data capabilities!