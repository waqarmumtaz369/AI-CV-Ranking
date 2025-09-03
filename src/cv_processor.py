"""
CV/Resume processing module with PDF parsing and LLM-based extraction.
"""
import os
import re
import json
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
import pdfplumber
from sentence_transformers import SentenceTransformer
from .models import CV, Experience, Education


class CVProcessor:
    """Processes CV/Resume files to extract structured information."""
    
    def __init__(self, vllm_url: str = "http://localhost:8000", embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the CV processor."""
        self.vllm_url = vllm_url
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF {pdf_path}: {str(e)}")
    
    def call_vllm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call the vLLM API to process text."""
        try:
            payload = {
                "model": "Qwen/Qwen3-0.6B",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                raise Exception(f"vLLM API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling vLLM API: {str(e)}")
    
    def extract_structured_data(self, cv_text: str) -> Dict[str, Any]:
        """Extract structured data from CV text using LLM."""
        prompt = f"""
        Please extract the following information from this CV/Resume text and return it as a JSON object:

        CV Text:
        {cv_text}

        Please extract and return a JSON object with the following structure:
        {{
            "name": "Full name of the candidate",
            "email": "Email address if found",
            "phone": "Phone number if found",
            "linkedin_url": "LinkedIn profile URL if found",
            "skills": ["skill1", "skill2", "skill3"],
            "experience": [
                {{
                    "title": "Job title",
                    "company": "Company name",
                    "start_date": "Start date (YYYY-MM or YYYY)",
                    "end_date": "End date (YYYY-MM or YYYY or 'Present')",
                    "description": "Job description",
                    "skills_used": ["skill1", "skill2"]
                }}
            ],
            "education": [
                {{
                    "degree": "Degree type (e.g., Bachelor's, Master's)",
                    "field": "Field of study",
                    "institution": "Institution name",
                    "graduation_year": "Graduation year (YYYY)",
                    "gpa": "GPA if mentioned"
                }}
            ]
        }}

        Important notes:
        - Only include information that is clearly stated in the CV
        - For dates, use YYYY-MM format when possible, or just YYYY
        - For skills, include technical skills, programming languages, tools, frameworks
        - For experience, calculate duration in months if possible
        - Return only valid JSON, no additional text
        """
        
        try:
            response = self.call_vllm(prompt, max_tokens=1500)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                return json.loads(response)
                
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to basic extraction if LLM fails
            return self._fallback_extraction(cv_text)
    
    def _fallback_extraction(self, cv_text: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns if LLM fails."""
        data = {
            "name": None,
            "email": None,
            "phone": None,
            "linkedin_url": None,
            "skills": [],
            "experience": [],
            "education": []
        }
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', cv_text)
        if email_match:
            data["email"] = email_match.group(0)
        
        # Extract phone
        phone_match = re.search(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', cv_text)
        if phone_match:
            data["phone"] = phone_match.group(0)
        
        # Extract LinkedIn URL
        linkedin_match = re.search(r'https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9-]+/?', cv_text)
        if linkedin_match:
            data["linkedin_url"] = linkedin_match.group(0)
        
        # Extract basic skills (common technical terms)
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Go|Rust|PHP|Ruby|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Node\.?js|Django|Flask|Spring|Laravel|Express)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|GitHub|GitLab)\b',
            r'\b(?:SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra)\b',
            r'\b(?:Machine Learning|AI|Deep Learning|TensorFlow|PyTorch|Scikit-learn)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, cv_text, re.IGNORECASE)
            data["skills"].extend(matches)
        
        data["skills"] = list(set(data["skills"]))  # Remove duplicates
        
        return data
    
    def calculate_experience_duration(self, experience_list: List[Dict[str, Any]]) -> float:
        """Calculate total years of experience from experience list."""
        total_months = 0
        
        for exp in experience_list:
            start_date = exp.get("start_date", "")
            end_date = exp.get("end_date", "Present")
            
            try:
                # Parse start date
                if start_date:
                    if "-" in start_date:
                        start_year, start_month = start_date.split("-")
                        start_months = int(start_year) * 12 + int(start_month)
                    else:
                        start_months = int(start_date) * 12
                else:
                    continue
                
                # Parse end date
                if end_date.lower() == "present" or end_date.lower() == "current":
                    end_months = datetime.now().year * 12 + datetime.now().month
                elif end_date:
                    if "-" in end_date:
                        end_year, end_month = end_date.split("-")
                        end_months = int(end_year) * 12 + int(end_month)
                    else:
                        end_months = int(end_date) * 12
                else:
                    continue
                
                duration = end_months - start_months
                if duration > 0:
                    total_months += duration
                    
            except (ValueError, IndexError):
                continue
        
        return round(total_months / 12, 1)
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        return self.embedding_model.encode(text).tolist()
    
    def process_cv(self, pdf_path: str) -> CV:
        """Process a CV file and return structured data."""
        filename = os.path.basename(pdf_path)
        
        # Extract text from PDF
        cv_text = self.extract_text_from_pdf(pdf_path)
        
        # Extract structured data using LLM
        structured_data = self.extract_structured_data(cv_text)
        
        # Create Experience objects
        experience_objects = []
        for exp_data in structured_data.get("experience", []):
            experience_objects.append(Experience(
                title=exp_data.get("title", ""),
                company=exp_data.get("company", ""),
                start_date=exp_data.get("start_date"),
                end_date=exp_data.get("end_date"),
                description=exp_data.get("description", ""),
                skills_used=exp_data.get("skills_used", [])
            ))
        
        # Create Education objects
        education_objects = []
        for edu_data in structured_data.get("education", []):
            education_objects.append(Education(
                degree=edu_data.get("degree", ""),
                field=edu_data.get("field", ""),
                institution=edu_data.get("institution", ""),
                graduation_year=edu_data.get("graduation_year"),
                gpa=edu_data.get("gpa")
            ))
        
        # Calculate total experience
        total_experience = self.calculate_experience_duration(structured_data.get("experience", []))
        
        # Generate embeddings for different sections
        skills_text = " ".join(structured_data.get("skills", []))
        experience_text = " ".join([exp.get("description", "") for exp in structured_data.get("experience", [])])
        education_text = " ".join([f"{edu.get('degree', '')} {edu.get('field', '')}" for edu in structured_data.get("education", [])])
        
        skills_embeddings = self.generate_embeddings(skills_text) if skills_text else None
        experience_embeddings = self.generate_embeddings(experience_text) if experience_text else None
        education_embeddings = self.generate_embeddings(education_text) if education_text else None
        
        return CV(
            filename=filename,
            full_text=cv_text,
            name=structured_data.get("name"),
            email=structured_data.get("email"),
            phone=structured_data.get("phone"),
            linkedin_url=structured_data.get("linkedin_url"),
            skills=structured_data.get("skills", []),
            experience=experience_objects,
            education=education_objects,
            total_experience_years=total_experience,
            skills_embeddings=skills_embeddings,
            experience_embeddings=experience_embeddings,
            education_embeddings=education_embeddings
        )
