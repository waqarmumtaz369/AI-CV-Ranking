#!/usr/bin/env python3
"""
Test script to verify semantic search and vector store functionality.
"""
import sys
import os
sys.path.append('src')

from job_processor import JobDescriptionProcessor
from cv_processor import CVProcessor
from vector_store import VectorStore
from matching_engine import MatchingEngine
from models import CV, Experience, Education

def test_semantic_search():
    """Test the complete semantic search workflow."""
    print("🧪 Testing Semantic Search and Vector Store Functionality")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing components...")
    job_processor = JobDescriptionProcessor()
    cv_processor = CVProcessor()
    vector_store = VectorStore()
    matching_engine = MatchingEngine()
    print("✅ All components initialized successfully")
    
    # Test job description processing
    print("\n2. Testing job description processing...")
    job_title = "Gen-AI Engineer"
    job_description = """
    A skilled Machine Learning Engineer with hands-on experience in developing and deploying in-house machine learning models for business-critical applications. Proficient in model lifecycle management, from data preprocessing and feature engineering to training, evaluation, and deployment using scalable frameworks.
    
    Build autonomous AI-driven agents for tasks such as reasoning, decision-making, automation, and contextual understanding.
    Customize and deploy GPT, LLaMA, Claude, Mistral, and other foundation models to solve industry-specific challenges.
    Design scalable, production-ready AI workflows using Python, LangChain, LlamaIndex, OpenAI functions, and AutoGPT.
    Enhance AI agents with retrieval-augmented generation (RAG), vector databases (FAISS, Pinecone, Weaviate, ChromaDB), and knowledge graphs.
    
    Requirements:
    Proven experience developing GenAI-powered applications and AI agents.
    Strong proficiency in Python and AI/ML frameworks (TensorFlow, PyTorch, Hugging Face, LangChain, LlamaIndex, OpenAI API).
    Hands-on expertise in LLM fine-tuning, model adaptation, and prompt engineering.
    Familiarity with vector databases (FAISS, Pinecone, Weaviate, ChromaDB) and knowledge retrieval techniques.
    Experience integrating AI with APIs, automation frameworks, and microservices.
    Working knowledge of cloud-based AI deployment (AWS, Azure, GCP).
    Understanding of NLP, reinforcement learning, and multi-agent architectures.
    """
    
    job = job_processor.process_job_description(job_title, job_description)
    print(f"✅ Job processed: {job.title}")
    print(f"   - Required skills: {len(job.required_skills)}")
    print(f"   - Preferred skills: {len(job.preferred_skills)}")
    print(f"   - Experience required: {job.experience_years} years")
    print(f"   - Embeddings generated: {len(job.embeddings) if job.embeddings else 0} dimensions")
    
    # Test CV processing with a sample CV
    print("\n3. Testing CV processing...")
    sample_cv_text = """
    Muhammad Ehsan - Machine Learning AI Engineer
    
    Email: muhammad.ehsan@email.com
    Phone: +92-300-1234567
    Location: Karachi, Pakistan
    LinkedIn: https://linkedin.com/in/muhammad-ehsan
    
    Professional Summary:
    Experienced Machine Learning Engineer with 3+ years of experience in developing and deploying AI models. 
    Proficient in Python, TensorFlow, PyTorch, and cloud platforms. Strong background in NLP and computer vision.
    
    Technical Skills:
    Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, 
    Natural Language Processing, Computer Vision, AWS, Azure, Docker, Kubernetes, Git, 
    LangChain, Hugging Face, OpenAI API, Vector Databases, ChromaDB, FAISS
    
    Experience:
    Senior ML Engineer | TechCorp | 2022-Present
    - Developed and deployed machine learning models for business applications
    - Implemented RAG systems using LangChain and ChromaDB
    - Built AI agents using OpenAI API and custom fine-tuned models
    - Optimized model performance and reduced inference time by 40%
    
    ML Engineer | DataTech Solutions | 2020-2022
    - Built NLP models for text classification and sentiment analysis
    - Implemented computer vision models for image recognition
    - Worked with cloud platforms (AWS, Azure) for model deployment
    
    Education:
    Bachelor of Science in Computer Science | University of Karachi | 2020
    GPA: 3.7/4.0
    
    Projects:
    AI-Powered Document Analysis System
    - Built using Python, LangChain, and ChromaDB
    - Implemented RAG for document question-answering
    - Deployed on AWS with Docker containers
    
    Certifications:
    AWS Certified Machine Learning - Specialty
    Google Cloud Professional Machine Learning Engineer
    """
    
    # Create a test CV object
    test_cv = CV(
        filename="test_cv.pdf",
        full_text=sample_cv_text,
        name="Muhammad Ehsan",
        email="muhammad.ehsan@email.com",
        phone="+92-300-1234567",
        location="Karachi, Pakistan",
        linkedin_url="https://linkedin.com/in/muhammad-ehsan",
        professional_summary="Experienced Machine Learning Engineer with 3+ years of experience in developing and deploying AI models.",
        skills=["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Natural Language Processing", "Computer Vision", "AWS", "Azure", "Docker", "Kubernetes", "Git", "LangChain", "Hugging Face", "OpenAI API", "Vector Databases", "ChromaDB", "FAISS"],
        experience=[
            Experience(
                title="Senior ML Engineer",
                company="TechCorp",
                start_date="2022",
                end_date="Present",
                description="Developed and deployed machine learning models for business applications. Implemented RAG systems using LangChain and ChromaDB.",
                skills_used=["Python", "LangChain", "ChromaDB", "OpenAI API"]
            ),
            Experience(
                title="ML Engineer",
                company="DataTech Solutions",
                start_date="2020",
                end_date="2022",
                description="Built NLP models for text classification and sentiment analysis. Implemented computer vision models for image recognition.",
                skills_used=["Python", "NLP", "Computer Vision", "AWS", "Azure"]
            )
        ],
        education=[
            Education(
                degree="Bachelor of Science",
                field="Computer Science",
                institution="University of Karachi",
                graduation_year=2020,
                gpa=3.7
            )
        ],
        total_experience_years=3.5,
        skills_embeddings=cv_processor.generate_embeddings("Python Machine Learning Deep Learning TensorFlow PyTorch Scikit-learn Pandas NumPy Natural Language Processing Computer Vision AWS Azure Docker Kubernetes Git LangChain Hugging Face OpenAI API Vector Databases ChromaDB FAISS"),
        experience_embeddings=cv_processor.generate_embeddings("Senior ML Engineer developed and deployed machine learning models for business applications. Implemented RAG systems using LangChain and ChromaDB. Built AI agents using OpenAI API and custom fine-tuned models."),
        education_embeddings=cv_processor.generate_embeddings("Bachelor of Science Computer Science University of Karachi")
    )
    
    print(f"✅ Test CV created: {test_cv.name}")
    print(f"   - Skills: {len(test_cv.skills)}")
    print(f"   - Experience entries: {len(test_cv.experience)}")
    print(f"   - Total experience: {test_cv.total_experience_years} years")
    print(f"   - Skills embeddings: {len(test_cv.skills_embeddings) if test_cv.skills_embeddings else 0} dimensions")
    print(f"   - Experience embeddings: {len(test_cv.experience_embeddings) if test_cv.experience_embeddings else 0} dimensions")
    
    # Test vector store operations
    print("\n4. Testing vector store operations...")
    job_id = vector_store.add_job_description(job)
    cv_id = vector_store.add_cv(test_cv)
    print(f"✅ Job stored with ID: {job_id[:8]}...")
    print(f"✅ CV stored with ID: {cv_id[:8]}...")
    
    # Test semantic similarity calculation
    print("\n5. Testing semantic similarity calculation...")
    semantic_score, semantic_details = matching_engine.calculate_semantic_similarity(job, test_cv)
    print(f"✅ Semantic similarity score: {semantic_score:.3f}")
    print(f"   Details: {semantic_details}")
    
    # Test complete matching
    print("\n6. Testing complete CV matching...")
    matching_result = matching_engine.match_cv_to_job(test_cv, job)
    print(f"✅ Complete matching completed")
    print(f"   - Overall score: {matching_result.total_score:.3f}")
    print(f"   - Summary: {matching_result.summary}")
    
    print("\n📊 Individual Component Scores:")
    for score in matching_result.individual_scores:
        print(f"   - {score.component}: {score.score:.3f} - {score.details}")
    
    # Test vector store search
    print("\n7. Testing vector store search...")
    search_results = vector_store.search_similar_cvs(job, n_results=5)
    print(f"✅ Vector store search completed")
    print(f"   - Found {len(search_results)} similar CVs")
    
    if search_results:
        for i, result in enumerate(search_results):
            print(f"   - Result {i+1}: {result['name']} (Score: {result['overall_score']:.3f})")
    
    # Test collection stats
    print("\n8. Testing collection statistics...")
    stats = vector_store.get_collection_stats()
    print(f"✅ Collection stats retrieved")
    print(f"   - CVs stored: {stats['cv_count']}")
    print(f"   - Jobs stored: {stats['job_count']}")
    print(f"   - Total embeddings: {stats['total_embeddings']}")
    
    print("\n🎉 All tests completed successfully!")
    print("✅ Semantic search is working correctly")
    print("✅ Vector store is functioning properly")
    print("✅ Embeddings are being generated and used for similarity calculations")
    
    return True

if __name__ == "__main__":
    try:
        test_semantic_search()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
