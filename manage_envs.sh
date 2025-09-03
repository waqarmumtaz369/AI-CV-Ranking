#!/bin/bash

# Environment Management Script for AI CV Ranking System
# This script helps manage the separate vLLM and main project environments

echo "🔧 AI CV Ranking System - Environment Manager"
echo "=============================================="
echo ""

case "$1" in
    "vllm")
        echo "🚀 Starting vLLM environment..."
        if [ ! -d "vllm-env" ]; then
            echo "❌ vLLM environment not found. Creating it..."
            uv venv vllm-env --python 3.12 --seed
            source vllm-env/bin/activate
            uv pip install vllm --torch-backend=auto
            echo "✅ vLLM environment created and ready!"
        else
            echo "📦 Activating vLLM environment..."
            source vllm-env/bin/activate
            echo "✅ vLLM environment activated!"
        fi
        echo ""
        echo "💡 To start vLLM server: ./run_vllm.sh"
        echo "💡 To test vLLM: vllm --help"
        ;;
    
    "main")
        echo "🚀 Starting main project environment..."
        if [ ! -d "main-env" ]; then
            echo "❌ Main environment not found. Creating it..."
            uv venv main-env --python 3.12 --seed
            source main-env/bin/activate
            uv pip install -r requirements.txt
            echo "✅ Main environment created and ready!"
        else
            echo "📦 Activating main environment..."
            source main-env/bin/activate
            echo "✅ Main environment activated!"
        fi
        echo ""
        echo "💡 To start the app: ./run_app.sh"
        echo "💡 To test imports: python -c \"import streamlit; import chromadb; print('OK')\""
        ;;
    
    "test")
        echo "🧪 Testing both environments..."
        echo ""
        
        echo "Testing vLLM environment..."
        if [ -d "vllm-env" ]; then
            source vllm-env/bin/activate
            if vllm --help > /dev/null 2>&1; then
                echo "✅ vLLM environment: OK"
            else
                echo "❌ vLLM environment: FAILED"
            fi
        else
            echo "❌ vLLM environment: NOT FOUND"
        fi
        
        echo ""
        echo "Testing main environment..."
        if [ -d "main-env" ]; then
            source main-env/bin/activate
            if python -c "import streamlit, chromadb, sentence_transformers" > /dev/null 2>&1; then
                echo "✅ Main environment: OK"
            else
                echo "❌ Main environment: FAILED"
            fi
        else
            echo "❌ Main environment: NOT FOUND"
        fi
        ;;
    
    "clean")
        echo "🧹 Cleaning up environments..."
        read -p "This will remove both vllm-env and main-env. Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf vllm-env main-env
            echo "✅ Environments cleaned up!"
        else
            echo "❌ Cleanup cancelled."
        fi
        ;;
    
    *)
        echo "Usage: $0 {vllm|main|test|clean}"
        echo ""
        echo "Commands:"
        echo "  vllm  - Activate or create vLLM environment"
        echo "  main  - Activate or create main project environment"
        echo "  test  - Test both environments"
        echo "  clean - Remove both environments"
        echo ""
        echo "Examples:"
        echo "  $0 vllm    # Activate vLLM environment"
        echo "  $0 main    # Activate main environment"
        echo "  $0 test    # Test both environments"
        echo "  $0 clean   # Remove both environments"
        ;;
esac
