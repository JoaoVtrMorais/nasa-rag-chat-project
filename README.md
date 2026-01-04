# NASA RAG Chat Project 

A Retrieval-Augmented Generation (RAG) system with real-time evaluation capabilities. Create a complete RAG pipeline from document processing to interactive chat interface.

## 🎯 Learning Objectives

- Build document embedding pipelines with ChromaDB and OpenAI
- Implement RAG retrieval systems with semantic search
- Create LLM client integrations with conversation management
- Develop real-time evaluation systems using RAGAS metrics
- Build interactive chat interfaces with Streamlit
- Handle error scenarios and edge cases in production systems

## 📁 Project Structure

```
/
├── chat.py                 # Main Streamlit chat application (TODO-based)
├── embedding_pipeline.py   # ChromaDB embedding pipeline (TODO-based)
├── llm_client.py           # OpenAI LLM client wrapper (TODO-based)
├── rag_client.py           # RAG system client (TODO-based)
├── ragas_evaluator.py      # RAGAS evaluation metrics (TODO-based)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Basic understanding of Python, APIs, and vector databases
- Familiarity with machine learning concepts

### Installation

1. **Navigate to the project folder**:
   ```bash
   cd nasa-rag-chat-project
   ```

2. **Install dependencies**:

   You can use either pip or uv for installation:
   ```bash
   # Using pip
   python -m venv .venv
   source .venv/bin/activate  # On Unix/Mac
   # .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt

   # Or using uv (faster alternative)
   uv sync
   ```

3. **Set up your OpenAI API key**:

   Create a file named .env in the root of the project with the following content:
   ```bash
   OPEN_AI_KEY="your-api-key-here"
   ```

## 📊 Data Requirements

### **Expected Data Structure**
The system expects NASA document data organized in folders:
```
data/
├── apollo11/           # Apollo 11 mission documents
│   ├── *.txt          # Text files with mission data
├── apollo13/           # Apollo 13 mission documents
│   ├── *.txt          # Text files with mission data
└── challenger/         # Challenger mission documents
    ├── *.txt          # Text files with mission data
```

### **Supported Document Types**
- Plain text files (.txt)
- Mission transcripts
- Technical documents
- Audio transcriptions
- Flight plans and procedures

## 🛠️ Usage

### 1. Process documents with the embedding pipeline:

   ```bash
   python embedding_pipeline.py --openai-key YOUR_KEY --data-path ./data
   ```

### 2. Launch the chat interface:

   ```bash
   streamlit run chat.py
   ```

Process documents with the embedding pipeline:

## 🧪 Testing Your Implementation

### **Component Testing**

1. **Test LLM Client**:
   ```python
   from llm_client import generate_response
   response = generate_response(api_key, "What was Apollo 11?", "", [])
   print(response)
   ```

2. **Test RAG Client**:
   ```python
   from rag_client import discover_chroma_backends
   backends = discover_chroma_backends()
   print(backends)
   ```

3. **Test Embedding Pipeline**:
   ```bash
   python embedding_pipeline.py --openai-key YOUR_KEY --stats-only
   ```

4. **Test Evaluation**:
   ```bash
   python ragas_evaluator.py
   ```

## 🔧 Configuration Options

### **Embedding Pipeline**
- Chunk size and overlap settings
- Batch processing parameters
- Update modes for existing documents
- Embedding model selection

### **LLM Client**
- Model selection (GPT-3.5-turbo, GPT-4)
- Temperature and creativity settings
- Maximum token limits
- Conversation history length

### **RAG System**
- Number of documents to retrieve
- Mission-specific filtering options
- Similarity thresholds

### **Evaluation System**
- Metric selection and weighting
- Evaluation frequency settings
- Display preferences
