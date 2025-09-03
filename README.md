# AI-CV-Ranking
A Generative AI-Powered CV Parsing and Ranking System 

## Setup

### Using uv (Recommended)

```bash
$ uv venv                    # Create a new virtual environment
$ source .venv/bin/activate  # Activate the virtual environment
$ uv pip install vllm --torch-backend=auto  # Install vLLM with automatic torch backend detection
```

### Install Additional Dependencies

```bash
$ uv pip install -r requirements.txt  # Install all required packages
```

## Quick Start

### 1. Start vLLM Server

```bash
$ ./run_vllm.sh  # Start the vLLM server with QWEN model
```

### 2. Start the Application

```bash
$ ./run_app.sh  # Start the Streamlit application
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

## Usage

1. **Start vLLM Server**: Run `./run_vllm.sh` to start the local LLM server
2. **Launch Application**: Run `./run_app.sh` to start the Streamlit UI
3. **Process Job Description**: Enter job title and description in the first tab
4. **Upload CVs**: Upload PDF CV files in the second tab (max 10 for Phase 1)
5. **Run Matching**: Click "Run Matching Analysis" to get ranked results
6. **Export Results**: Download results in CSV or JSON format

## Technical Details

- **LLM**: QWEN 3-0.6B served via vLLM
- **Vector Database**: ChromaDB for embeddings storage
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **PDF Processing**: pdfplumber for text extraction
- **UI Framework**: Streamlit
- **Data Validation**: Pydantic models

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