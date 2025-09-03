"""
Pydantic models for the CV ranking system.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
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
    
    @validator('required_skills', 'preferred_skills', 'keywords')
    def skills_must_not_be_empty(cls, v):
        if v and any(not skill.strip() for skill in v):
            raise ValueError('Skills cannot be empty strings')
        return [skill.strip() for skill in v if skill.strip()]


class Education(BaseModel):
    """Education information model."""
    degree: str = Field(..., description="Degree type (e.g., Bachelor's, Master's)")
    field: str = Field(..., description="Field of study")
    institution: str = Field(..., description="Institution name")
    graduation_year: Optional[int] = Field(None, description="Graduation year")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA if available")


class Experience(BaseModel):
    """Work experience model."""
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date (None for current)")
    duration_months: Optional[int] = Field(None, ge=0, description="Duration in months")
    description: str = Field(..., description="Job description")
    skills_used: List[str] = Field(default_factory=list, description="Skills used in this role")


class CV(BaseModel):
    """CV/Resume model with extracted information."""
    filename: str = Field(..., description="Original filename")
    full_text: str = Field(..., description="Full extracted text")
    name: Optional[str] = Field(None, description="Candidate name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    skills: List[str] = Field(default_factory=list, description="Technical skills")
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    education: List[Education] = Field(default_factory=list, description="Education")
    total_experience_years: Optional[float] = Field(None, ge=0, description="Total years of experience")
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    
    # Embeddings for different sections
    skills_embeddings: Optional[List[float]] = Field(None, description="Skills embeddings")
    experience_embeddings: Optional[List[float]] = Field(None, description="Experience embeddings")
    education_embeddings: Optional[List[float]] = Field(None, description="Education embeddings")


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
