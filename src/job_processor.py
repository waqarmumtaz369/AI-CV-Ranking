"""
Job Description processing module.
"""
import re
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .models import JobDescription


class JobDescriptionProcessor:
    """Processes job descriptions to extract keywords and generate embeddings."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the processor with an embedding model."""
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from job description text."""
        # Common technical skills patterns
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Go|Rust|PHP|Ruby|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Node\.?js|Django|Flask|Spring|Laravel|Express)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|GitHub|GitLab)\b',
            r'\b(?:SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra)\b',
            r'\b(?:Machine Learning|AI|Deep Learning|TensorFlow|PyTorch|Scikit-learn)\b',
            r'\b(?:Data Science|Analytics|Pandas|NumPy|Matplotlib|Seaborn)\b',
            r'\b(?:DevOps|CI/CD|Microservices|REST|GraphQL|API|Web Development)\b',
            r'\b(?:Project Management|Agile|Scrum|Kanban|Leadership|Communication)\b'
        ]
        
        keywords = set()
        text_lower = text.lower()
        
        # Extract skills using patterns
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.update(matches)
        
        # Extract common words (excluding stop words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words with 3+ characters
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        keywords.update([word for word in words if word not in stop_words])
        
        return list(keywords)
    
    def extract_skills(self, text: str) -> tuple[List[str], List[str]]:
        """Extract required and preferred skills from job description."""
        required_skills = []
        preferred_skills = []
        
        # Look for "required" and "preferred" sections
        required_section = re.search(r'(?:required|must have|essential).*?(?=\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
        preferred_section = re.search(r'(?:preferred|nice to have|bonus).*?(?=\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
        
        if required_section:
            required_text = required_section.group(0)
            required_skills = self.extract_keywords(required_text)
        
        if preferred_section:
            preferred_text = preferred_section.group(0)
            preferred_skills = self.extract_keywords(preferred_text)
        
        # If no specific sections found, extract from entire text
        if not required_skills and not preferred_skills:
            all_skills = self.extract_keywords(text)
            # Split roughly 70/30 for required/preferred
            split_point = int(len(all_skills) * 0.7)
            required_skills = all_skills[:split_point]
            preferred_skills = all_skills[split_point:]
        
        return required_skills, preferred_skills
    
    def extract_experience_years(self, text: str) -> Optional[int]:
        """Extract required years of experience from job description."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?relevant\s*experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?professional\s*experience',
            r'minimum\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_education_level(self, text: str) -> Optional[str]:
        """Extract required education level from job description."""
        education_patterns = [
            r'\b(bachelor\'?s?|b\.?s\.?|b\.?e\.?)\b',
            r'\b(master\'?s?|m\.?s\.?|m\.?e\.?|mba)\b',
            r'\b(phd|doctorate|doctoral)\b',
            r'\b(associate|diploma|certificate)\b'
        ]
        
        for pattern in education_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return None
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the job description text."""
        return self.embedding_model.encode(text).tolist()
    
    def process_job_description(self, title: str, description: str) -> JobDescription:
        """Process a complete job description and return structured data."""
        # Extract information
        required_skills, preferred_skills = self.extract_skills(description)
        keywords = self.extract_keywords(description)
        experience_years = self.extract_experience_years(description)
        education_level = self.extract_education_level(description)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(description)
        
        return JobDescription(
            title=title,
            description=description,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_years=experience_years,
            education_level=education_level,
            keywords=keywords,
            embeddings=embeddings
        )
