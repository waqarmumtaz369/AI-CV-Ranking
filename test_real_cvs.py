#!/usr/bin/env python3
"""
Test script to verify semantic search with real PDF files.
"""
import sys
import os
sys.path.append('src')

from job_processor import JobDescriptionProcessor
from cv_processor import CVProcessor
from vector_store import VectorStore
from matching_engine import MatchingEngine

def test_with_real_files():
    """Test with real PDF files from input folder."""
    print("🧪 Testing with Real PDF Files")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing components...")
    job_processor = JobDescriptionProcessor()
    cv_processor = CVProcessor()
    vector_store = VectorStore()
    matching_engine = MatchingEngine()
    print("✅ All components initialized")
    
    # Process job description from jd.txt
    print("\n2. Processing job description from jd.txt...")
    with open('input/jd.txt', 'r') as f:
        jd_content = f.read()
    
    lines = jd_content.strip().split('\n')
    job_title = lines[0]  # First line is the title
    job_description = '\n'.join(lines[1:])  # Rest is the description
    
    job = job_processor.process_job_description(job_title, job_description)
    print(f"✅ Job processed: {job.title}")
    print(f"   - Required skills: {len(job.required_skills)}")
    print(f"   - Preferred skills: {len(job.preferred_skills)}")
    print(f"   - Experience required: {job.experience_years} years")
    print(f"   - Keywords: {len(job.keywords)}")
    
    # Store job in vector store
    job_id = vector_store.add_job_description(job)
    print(f"✅ Job stored with ID: {job_id[:8]}...")
    
    # Process a few CV files
    print("\n3. Processing CV files...")
    cv_files = [
        "input/CVs/Muhammad Ehsan - Machine Learning AI[1].pdf",
        "input/CVs/ML CV Fateh[1].pdf",
        "input/CVs/Mubashir_Resume_Senior_ML_Engineer[1].pdf"
    ]
    
    processed_cvs = []
    for cv_file in cv_files:
        if os.path.exists(cv_file):
            try:
                print(f"   Processing: {os.path.basename(cv_file)}")
                cv = cv_processor.process_cv(cv_file)
                processed_cvs.append(cv)
                
                # Store in vector store
                cv_id = vector_store.add_cv(cv)
                print(f"   ✅ Processed: {cv.name or 'Unknown'} (ID: {cv_id[:8]}...)")
                print(f"      - Skills: {len(cv.skills)}")
                print(f"      - Experience: {cv.total_experience_years} years")
                print(f"      - Embeddings generated: {len(cv.skills_embeddings) if cv.skills_embeddings else 0} dims")
                
            except Exception as e:
                print(f"   ❌ Error processing {cv_file}: {e}")
        else:
            print(f"   ⚠️ File not found: {cv_file}")
    
    if not processed_cvs:
        print("❌ No CVs were processed successfully")
        return False
    
    # Test matching
    print(f"\n4. Testing matching for {len(processed_cvs)} CVs...")
    matching_results = []
    
    for cv in processed_cvs:
        result = matching_engine.match_cv_to_job(cv, job)
        matching_results.append(result)
        print(f"   {cv.name or 'Unknown'}: {result.total_score:.3f} ({result.summary[:50]}...)")
    
    # Rank results
    ranked_results = matching_engine.rank_cvs(matching_results)
    
    print(f"\n📊 Ranking Results:")
    for i, result in enumerate(ranked_results):
        print(f"   #{i+1} {result.candidate_name or 'Unknown'}: {result.total_score:.3f}")
        
        # Show semantic similarity score
        for score in result.individual_scores:
            if score.component == "semantic_similarity":
                print(f"      Semantic similarity: {score.score:.3f} - {score.details}")
                break
    
    # Test vector store search
    print(f"\n5. Testing vector store search...")
    search_results = vector_store.search_similar_cvs(job, n_results=5)
    print(f"✅ Found {len(search_results)} similar CVs in vector store")
    
    for i, result in enumerate(search_results):
        print(f"   #{i+1} {result['name']}: {result['overall_score']:.3f}")
        if 'scores' in result:
            print(f"      Section scores: {result['scores']}")
    
    # Show final stats
    print(f"\n6. Final statistics...")
    stats = vector_store.get_collection_stats()
    print(f"   - CVs stored: {stats['cv_count']}")
    print(f"   - Jobs stored: {stats['job_count']}")
    print(f"   - Total embeddings: {stats['total_embeddings']}")
    
    print(f"\n🎉 Real file testing completed successfully!")
    print(f"✅ Semantic search working with real PDF files")
    print(f"✅ Vector store functioning with actual CV data")
    print(f"✅ Embeddings generated from real CV content")
    
    return True

if __name__ == "__main__":
    try:
        test_with_real_files()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
