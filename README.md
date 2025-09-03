# AI-CV-Ranking
A Generative AI-Powered CV Parsing and Ranking System

## Overview

This system uses separate virtual environments to avoid dependency conflicts between vLLM and other project dependencies. It provides contextual search through vector similarity for matching CVs against job descriptions.

## Environment Structure

- **`vllm-env/`** - Contains vLLM and related dependencies for model serving
- **`main-env/`** - Contains the main project dependencies (Streamlit, ChromaDB, etc.)

## Prerequisites

- Python 3.12+ installed
- uv package manager installed
- CUDA-compatible GPU (recommended)
- At least 8GB RAM

## Quick Setup

### Using the Management Script (Recommended)

```bash
# Test both environments
./manage_envs.sh test

# Activate vLLM environment
./manage_envs.sh vllm

# Activate main environment  
./manage_envs.sh main
```

### Manual Setup

#### vLLM Environment
```bash
# Create and setup vLLM environment
uv venv vllm-env --python 3.12 --seed
source vllm-env/bin/activate
uv pip install vllm --torch-backend=auto

# Test installation
vllm --help
```

#### Main Project Environment
```bash
# Create and setup main environment
uv venv main-env --python 3.12 --seed
source main-env/bin/activate
uv pip install -r requirements.txt

# Test installation
python -c "import streamlit; import chromadb; print('OK')"
```

## Running the Application

### Start vLLM Server (Terminal 1)
```bash
./run_vllm.sh
# or manually:
source vllm-env/bin/activate
vllm serve Qwen/Qwen2.5-0.5B --gpu-memory-utilization 0.7 --max-model-len 3072 --max-num-seqs 6 --host 0.0.0.0 --port 8000
```

### Start Main Application (Terminal 2)
```bash
./run_app.sh
# or manually:
source main-env/bin/activate
streamlit run app.py
```

The application will be available at: http://localhost:8501

## Features

### Phase 1 Implementation

- **Job Description Processing**: Extract keywords, skills, and requirements from job descriptions
- **CV Parsing**: Parse PDF CVs and extract structured information using LLM
- **Vector Storage**: Store CV and job embeddings in ChromaDB for semantic search
- **Matching Engine**: Match CVs against job descriptions based on:
  - Skills similarity (semantic matching)
  - Experience years (exact matching)
  - Education level (hierarchical matching)
  - Keywords (text matching)
- **Ranking System**: Rank candidates by overall matching score
- **Export Functionality**: Export results to CSV and JSON formats
- **Analytics Dashboard**: Visualize matching results and insights

### UI Components

- **Job Description Tab**: Input and process job descriptions
- **CV Processing Tab**: Upload and process multiple CV files (PDF only)
- **Matching Results Tab**: View ranked results with detailed analysis
- **Analytics Tab**: Visualize score distributions and component analysis

## Architecture

```
src/
├── models.py           # Pydantic data models
├── job_processor.py    # Job description processing
├── cv_processor.py     # CV parsing and extraction
├── vector_store.py     # ChromaDB integration
├── matching_engine.py  # Matching and ranking logic
└── streamlit_app.py    # Streamlit UI application
```

## Usage Workflow

1. **Start vLLM Server** (Terminal 1): Run `./run_vllm.sh`
2. **Start Streamlit App** (Terminal 2): Run `./run_app.sh`
3. **Open Browser** → http://localhost:8501
4. **Process Job Description** → Enter title and description
5. **Upload CVs** → Select PDF files (max 10)
6. **Run Matching** → Click "Run Matching Analysis"
7. **View Results** → Review ranked candidates
8. **Export Data** → Download CSV/JSON results

## Technical Details

- **LLM**: QWEN 2.5-0.5B served via vLLM
- **Vector Database**: ChromaDB for embeddings storage
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **PDF Processing**: pdfplumber for text extraction
- **UI Framework**: Streamlit
- **Data Validation**: Pydantic models

## Environment Management

- **Test environments**: `./manage_envs.sh test`
- **Clean up**: `./manage_envs.sh clean`
- **Help**: `./manage_envs.sh`

## Troubleshooting

### vLLM Issues
- Ensure CUDA is properly installed
- Check GPU memory availability
- Verify vLLM installation: `source vllm-env/bin/activate && vllm --help`

### Main App Issues
- Check all dependencies: `source main-env/bin/activate && python -c "import streamlit; import chromadb; import sentence_transformers"`
- Ensure vLLM server is running on port 8000
- Check port conflicts (8501 for Streamlit, 8000 for vLLM)

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--gpu-memory-utilization` to 0.5 or 0.6
   - Reduce `--max-model-len` to 2048 or 1536

2. **vLLM Not Starting**
   - Check if port 8000 is available
   - Verify CUDA installation
   - Check GPU memory availability

3. **Import Errors**
   - Ensure virtual environment is activated
   - Run `uv pip install -r requirements.txt`
   - Check Python version compatibility

4. **PDF Processing Issues**
   - Ensure PDF files are not password protected
   - Check file permissions
   - Verify pdfplumber installation

### Performance Optimization

- **For smaller GPUs**: Use quantization with `--quantization awq`
- **For CPU-only**: Remove CUDA-specific flags
- **For faster processing**: Increase `--max-num-seqs` if memory allows

## Verification

### Test vLLM Connection
```bash
curl http://localhost:8000/health
```

### Test API Endpoint
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Limitations (Phase 1)

- PDF files only (no DOC/DOCX support)
- Maximum 10 CVs per session
- No LinkedIn integration
- Basic skill matching (no advanced NLP)
- Local deployment only

## Future Enhancements

- Support for multiple file formats
- Advanced NLP for better skill extraction
- LinkedIn profile verification
- Batch processing capabilities
- Cloud deployment options
- Advanced analytics and reporting

## File Structure

```
AI-CV-Ranking/
├── vllm-env/              # vLLM virtual environment
├── main-env/              # Main project virtual environment
├── src/                   # Source code
│   ├── models.py          # Data models
│   ├── job_processor.py   # Job processing
│   ├── cv_processor.py    # CV processing
│   ├── vector_store.py    # ChromaDB integration
│   ├── matching_engine.py # Matching logic
│   └── streamlit_app.py   # UI application
├── app.py                 # Main entry point
├── run_vllm.sh           # vLLM server script
├── run_app.sh            # Application runner
├── manage_envs.sh        # Environment management script
├── requirements.txt      # Main project dependencies
├── requirements-vllm.txt # vLLM dependencies (reference)
└── README.md            # This documentation
```

## Next Steps

After successful setup:

1. **Test with Sample Data**: Use sample job descriptions and CVs
2. **Customize Matching**: Adjust weights in matching engine
3. **Add More Models**: Experiment with different LLM models
4. **Scale Up**: Process larger batches of CVs
5. **Deploy**: Consider cloud deployment for production use