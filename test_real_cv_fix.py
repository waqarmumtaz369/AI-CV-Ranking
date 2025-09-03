#!/usr/bin/env python3
"""
Test script to verify the fixes work with real CV data.
"""
import sys
import os
sys.path.append('src')

from cv_processor import CVProcessor

def test_real_cv_processing():
    """Test CV processing with deduplication fixes."""
    print("🧪 Testing Real CV Processing with Deduplication Fixes")
    print("=" * 60)
    
    # Initialize CV processor
    cv_processor = CVProcessor()
    
    # Test with one of the CV files
    cv_file = "input/CVs/Bilal resume[1].pdf"
    
    if not os.path.exists(cv_file):
        print(f"❌ CV file not found: {cv_file}")
        return False
    
    print(f"1. Processing CV: {cv_file}")
    try:
        cv = cv_processor.process_cv(cv_file)
        
        print(f"✅ CV processed successfully: {cv.name}")
        print(f"   - Skills: {len(cv.skills)}")
        print(f"   - Experience entries: {len(cv.experience)}")
        print(f"   - Education entries: {len(cv.education)}")
        print(f"   - Projects: {len(cv.projects)}")
        print(f"   - Certifications: {len(cv.certifications)}")
        
        # Show sample experience entries (should be deduplicated)
        print(f"\n2. Sample Experience Entries (first 3):")
        for i, exp in enumerate(cv.experience[:3]):
            print(f"   {i+1}. {exp.title} at {exp.company} ({exp.start_date} - {exp.end_date})")
        
        # Show sample education entries (should be deduplicated)
        print(f"\n3. Sample Education Entries (first 3):")
        for i, edu in enumerate(cv.education[:3]):
            print(f"   {i+1}. {edu.degree} from {edu.institution} ({edu.graduation_year})")
        
        # Check for duplicates
        exp_titles = [exp.title for exp in cv.experience if exp.title]
        edu_degrees = [edu.degree for edu in cv.education if edu.degree]
        
        exp_duplicates = len(exp_titles) - len(set(exp_titles))
        edu_duplicates = len(edu_degrees) - len(set(edu_degrees))
        
        print(f"\n4. Duplicate Check:")
        print(f"   - Experience duplicates: {exp_duplicates}")
        print(f"   - Education duplicates: {edu_duplicates}")
        
        if exp_duplicates == 0 and edu_duplicates == 0:
            print("   ✅ No duplicates found - deduplication working!")
        else:
            print("   ⚠️ Some duplicates still present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing CV: {e}")
        return False

if __name__ == "__main__":
    try:
        test_real_cv_processing()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
