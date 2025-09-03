# Setup Instructions for AI CV Ranking System

## Prerequisites

- Python 3.8+ installed
- uv package manager installed
- CUDA-compatible GPU (recommended)
- At least 8GB RAM

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install all required packages
uv pip install -r requirements.txt
```

### 2. Start vLLM Server

```bash
# Start the vLLM server (run in background or separate terminal)
./run_vllm.sh
```

**Alternative manual start:**
```bash
source .venv/bin/activate
vllm serve Qwen/Qwen2.5-0.5B \
  --gpu-memory-utilization 0.7 \
  --max-model-len 3072 \
  --max-num-seqs 6 \
  --host 0.0.0.0 \
  --port 8000
```

### 3. Start the Application

```bash
# Start the Streamlit application
./run_app.sh
```

The application will be available at: http://localhost:8501

## Verification

### Test the Implementation

```bash
# Run the test script
python3 test_implementation.py
```

### Test vLLM Connection

```bash
# Test if vLLM is running
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

## Troubleshooting

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

## Usage Workflow

1. **Start vLLM Server** (Terminal 1)
2. **Start Streamlit App** (Terminal 2)
3. **Open Browser** → http://localhost:8501
4. **Process Job Description** → Enter title and description
5. **Upload CVs** → Select PDF files (max 10)
6. **Run Matching** → Click "Run Matching Analysis"
7. **View Results** → Review ranked candidates
8. **Export Data** → Download CSV/JSON results

## File Structure

```
AI-CV-Ranking/
├── src/                    # Source code
│   ├── models.py          # Data models
│   ├── job_processor.py   # Job processing
│   ├── cv_processor.py    # CV processing
│   ├── vector_store.py    # ChromaDB integration
│   ├── matching_engine.py # Matching logic
│   └── streamlit_app.py   # UI application
├── app.py                 # Main entry point
├── run_vllm.sh           # vLLM server script
├── run_app.sh            # Application runner
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Next Steps

After successful setup:

1. **Test with Sample Data**: Use sample job descriptions and CVs
2. **Customize Matching**: Adjust weights in matching engine
3. **Add More Models**: Experiment with different LLM models
4. **Scale Up**: Process larger batches of CVs
5. **Deploy**: Consider cloud deployment for production use
