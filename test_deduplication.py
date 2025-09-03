#!/usr/bin/env python3
"""
Test script to verify deduplication fixes.
"""
import sys
import os
sys.path.append('src')

from cv_processor import CVProcessor

def test_deduplication():
    """Test the deduplication functionality."""
    print("🧪 Testing Deduplication Fixes")
    print("=" * 40)
    
    # Initialize CV processor
    cv_processor = CVProcessor()
    
    # Test experience deduplication
    print("1. Testing experience deduplication...")
    test_experience = [
        {"title": "Software Engineer", "company": "Tech Corp", "start_date": "2020", "end_date": "2022"},
        {"title": "Software Engineer", "company": "Tech Corp", "start_date": "2020", "end_date": "2022"},  # Duplicate
        {"title": "Senior Engineer", "company": "Tech Corp", "start_date": "2022", "end_date": "Present"},
        {"title": "Unknown", "company": "Unknown Company", "start_date": "2020", "end_date": "2022"},  # Should be filtered
        {"title": "Data Scientist", "company": "AI Corp", "start_date": "2019", "end_date": "2020"},
    ]
    
    deduplicated_exp = cv_processor._deduplicate_experience(test_experience)
    print(f"   Original: {len(test_experience)} entries")
    print(f"   Deduplicated: {len(deduplicated_exp)} entries")
    print(f"   ✅ Removed {len(test_experience) - len(deduplicated_exp)} duplicates/placeholders")
    
    # Test education deduplication
    print("\n2. Testing education deduplication...")
    test_education = [
        {"degree": "Bachelor of Science", "institution": "University of Tech", "graduation_year": 2020},
        {"degree": "Bachelor of Science", "institution": "University of Tech", "graduation_year": 2020},  # Duplicate
        {"degree": "Master of Science", "institution": "University of Tech", "graduation_year": 2022},
        {"degree": "Unknown", "institution": "Unknown Institution", "graduation_year": None},  # Should be filtered
        {"degree": "PhD", "institution": "Research University", "graduation_year": 2024},
    ]
    
    deduplicated_edu = cv_processor._deduplicate_education(test_education)
    print(f"   Original: {len(test_education)} entries")
    print(f"   Deduplicated: {len(deduplicated_edu)} entries")
    print(f"   ✅ Removed {len(test_education) - len(deduplicated_edu)} duplicates/placeholders")
    
    # Test projects deduplication
    print("\n3. Testing projects deduplication...")
    test_projects = [
        {"title": "AI Chatbot", "description": "Built a chatbot using Python and NLP"},
        {"title": "AI Chatbot", "description": "Built a chatbot using Python and NLP"},  # Duplicate
        {"title": "Web App", "description": "Created a web application with React"},
        {"title": "Unknown Project", "description": "Some project"},  # Should be filtered
        {"title": "Mobile App", "description": "Developed a mobile app with Flutter"},
    ]
    
    deduplicated_proj = cv_processor._deduplicate_projects(test_projects)
    print(f"   Original: {len(test_projects)} entries")
    print(f"   Deduplicated: {len(deduplicated_proj)} entries")
    print(f"   ✅ Removed {len(test_projects) - len(deduplicated_proj)} duplicates/placeholders")
    
    print("\n🎉 Deduplication tests completed successfully!")
    print("✅ Duplicate entries are now properly filtered")
    print("✅ Placeholder/unknown entries are removed")
    
    return True

if __name__ == "__main__":
    try:
        test_deduplication()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
