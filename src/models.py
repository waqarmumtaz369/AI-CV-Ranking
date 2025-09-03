"""
Pydantic models for the CV ranking system.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class JobDescription(BaseModel):
    """Job Description model with extracted information."""
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Full job description text")
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    experience_years: Optional[int] = Field(None, ge=0, le=50, description="Required years of experience")
    education_level: Optional[str] = Field(None, description="Required education level")
    location: Optional[str] = Field(None, description="Job location")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    embeddings: Optional[List[float]] = Field(None, description="Job description embeddings")
    
    @field_validator('required_skills', 'preferred_skills', 'keywords')
    @classmethod
    def skills_must_not_be_empty(cls, v):
        if v and any(not skill.strip() for skill in v):
            raise ValueError('Skills cannot be empty strings')
        return [skill.strip() for skill in v if skill.strip()]


class Education(BaseModel):
    """Education information model."""
    degree: Optional[str] = Field(None, description="Degree type (e.g., Bachelor's, Master's)")
    field: Optional[str] = Field(None, description="Field of study")
    institution: Optional[str] = Field(None, description="Institution name")
    graduation_year: Optional[int] = Field(None, description="Graduation year")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA if available")
    
    @field_validator('graduation_year', mode='before')
    @classmethod
    def validate_graduation_year(cls, v):
        """Handle placeholder text and convert to integer."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Handle placeholder text
            if any(placeholder in v.lower() for placeholder in ['graduation year', 'yyyy', 'year', 'placeholder']):
                return None
            # Try to extract year from string
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', v)
            if year_match:
                return int(year_match.group())
            # Try to convert directly
            try:
                return int(v)
            except ValueError:
                return None
        return None
    
    @field_validator('gpa', mode='before')
    @classmethod
    def validate_gpa(cls, v):
        """Handle placeholder text and convert to float."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # Handle placeholder text
            if any(placeholder in v.lower() for placeholder in ['gpa if mentioned', 'gpa', 'placeholder', 'not mentioned']):
                return None
            # Try to extract number from string
            import re
            gpa_match = re.search(r'\b\d+\.?\d*\b', v)
            if gpa_match:
                try:
                    gpa_val = float(gpa_match.group())
                    # Ensure GPA is in reasonable range
                    if 0.0 <= gpa_val <= 4.0:
                        return gpa_val
                except ValueError:
                    pass
            # Try to convert directly
            try:
                gpa_val = float(v)
                if 0.0 <= gpa_val <= 4.0:
                    return gpa_val
            except ValueError:
                pass
        return None


class Experience(BaseModel):
    """Work experience model."""
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date (None for current)")
    duration_months: Optional[int] = Field(None, ge=0, description="Duration in months")
    description: Optional[str] = Field(None, description="Job description")
    skills_used: List[str] = Field(default_factory=list, description="Skills used in this role")


class Project(BaseModel):
    """Project information model."""
    title: Optional[str] = Field(None, description="Project title")
    project_type: Optional[str] = Field(None, description="Type: Personal/Academic/Open Source/Freelance")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies used")
    description: Optional[str] = Field(None, description="Project description")
    github_url: Optional[str] = Field(None, description="GitHub repository URL")
    demo_url: Optional[str] = Field(None, description="Demo/live URL")


class Certification(BaseModel):
    """Certification information model."""
    name: Optional[str] = Field(None, description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    date_earned: Optional[str] = Field(None, description="Date earned")
    expiry_date: Optional[str] = Field(None, description="Expiry date if applicable")


class CV(BaseModel):
    """CV/Resume model with extracted information."""
    filename: str = Field(..., description="Original filename")
    full_text: str = Field(..., description="Full extracted text")
    name: Optional[str] = Field(None, description="Candidate name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    portfolio_url: Optional[str] = Field(None, description="Portfolio website URL")
    location: Optional[str] = Field(None, description="Location (city, country)")
    professional_summary: Optional[str] = Field(None, description="Professional summary/objective")
    skills: List[str] = Field(default_factory=list, description="Technical skills")
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    education: List[Education] = Field(default_factory=list, description="Education")
    projects: List[Project] = Field(default_factory=list, description="Projects")
    certifications: List[Certification] = Field(default_factory=list, description="Certifications")
    awards: List[str] = Field(default_factory=list, description="Awards and achievements")
    publications: List[str] = Field(default_factory=list, description="Publications and talks")
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    total_experience_years: Optional[float] = Field(None, ge=0, description="Total years of experience")
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    
    # Embeddings for different sections
    skills_embeddings: Optional[List[float]] = Field(None, description="Skills embeddings")
    experience_embeddings: Optional[List[float]] = Field(None, description="Experience embeddings")
    education_embeddings: Optional[List[float]] = Field(None, description="Education embeddings")
    projects_embeddings: Optional[List[float]] = Field(None, description="Projects embeddings")


class MatchingScore(BaseModel):
    """Individual matching score component."""
    component: str = Field(..., description="Matching component (skills, experience, education)")
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
    details: str = Field(..., description="Detailed explanation of the score")
    matched_items: List[str] = Field(default_factory=list, description="Matched items")


class MatchingResult(BaseModel):
    """Final matching result for a CV against a job description."""
    cv_filename: str = Field(..., description="CV filename")
    candidate_name: Optional[str] = Field(None, description="Candidate name")
    total_score: float = Field(..., ge=0.0, le=1.0, description="Overall matching score")
    individual_scores: List[MatchingScore] = Field(..., description="Individual component scores")
    summary: str = Field(..., description="Overall matching summary")
    rank: Optional[int] = Field(None, description="Ranking position")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingStatus(BaseModel):
    """Processing status for UI updates."""
    stage: str = Field(..., description="Current processing stage")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress percentage")
    message: str = Field(..., description="Status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
