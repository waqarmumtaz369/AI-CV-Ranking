#!/bin/bash

# AI CV Ranking System - Application Runner
# This script starts the Streamlit application

echo "🚀 Starting AI CV Ranking System..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   uv venv"
    echo "   source .venv/bin/activate"
    echo "   uv pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
. .venv/bin/activate

# Check if vLLM is running
echo "🔍 Checking vLLM connection..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  vLLM server not running. Please start it first:"
    echo "   ./run_vllm.sh"
    echo ""
    echo "   Or run in another terminal:"
    echo "   source .venv/bin/activate"
    echo "   vllm serve Qwen/Qwen2.5-0.5B --gpu-memory-utilization 0.7 --max-model-len 3072 --max-num-seqs 6 --host 0.0.0.0 --port 8000"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Streamlit application
echo "🌐 Starting Streamlit application..."
echo "📱 The application will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
