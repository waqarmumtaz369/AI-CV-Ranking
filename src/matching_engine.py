"""
Matching engine for CV ranking system.
"""
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from .models import JobDescription, CV, MatchingResult, MatchingScore


class MatchingEngine:
    """Engine for matching CVs against job descriptions."""
    
    def __init__(self):
        """Initialize the matching engine."""
        pass
    
    def calculate_skill_similarity(self, job_skills: List[str], cv_skills: List[str]) -> Tuple[float, List[str]]:
        """Calculate skill similarity between job requirements and CV skills."""
        if not job_skills or not cv_skills:
            return 0.0, []
        
        job_skills_lower = [skill.lower().strip() for skill in job_skills]
        cv_skills_lower = [skill.lower().strip() for skill in cv_skills]
        
        # Common skill aliases for better matching
        skill_aliases = {
            'javascript': ['js', 'node.js', 'nodejs'],
            'python': ['py', 'python3'],
            'c++': ['cpp', 'c plus plus'],
            'c#': ['csharp', 'c sharp'],
            'react': ['reactjs', 'react.js'],
            'angular': ['angularjs', 'angular.js'],
            'vue': ['vuejs', 'vue.js'],
            'aws': ['amazon web services'],
            'azure': ['microsoft azure'],
            'gcp': ['google cloud'],
            'machine learning': ['ml'],
            'artificial intelligence': ['ai'],
            'data science': ['datascience'],
            'deep learning': ['deeplearning'],
            'natural language processing': ['nlp'],
            'computer vision': ['cv'],
            'postgresql': ['postgres'],
            'mongodb': ['mongo'],
            'elasticsearch': ['elastic search'],
            'github': ['git hub'],
            'gitlab': ['git lab'],
            'microservices': ['micro services'],
            'rest api': ['restful api'],
            'graphql': ['graph ql'],
            'typescript': ['ts'],
            'spring boot': ['springboot'],
            'red hat': ['redhat'],
            'macos': ['mac os'],
            'vscode': ['visual studio code', 'vs code'],
            'intellij': ['intellij idea'],
            'power bi': ['powerbi'],
            'tableau': ['tableau bi']
        }
        
        matched_skills = []
        total_score = 0.0
        
        for job_skill in job_skills_lower:
            best_match_score = 0.0
            best_match_skill = None
            
            # Check for exact match first
            if job_skill in cv_skills_lower:
                matched_skills.append(job_skill)
                total_score += 1.0
                continue
            
            # Check for alias matches
            job_aliases = skill_aliases.get(job_skill, [])
            for alias in job_aliases:
                if alias in cv_skills_lower:
                    matched_skills.append(alias)
                    total_score += 0.9
                    break
            else:
                # Check for partial matches
                for cv_skill in cv_skills_lower:
                    # Exact match
                    if job_skill == cv_skill:
                        match_score = 1.0
                    # Partial match using sequence matcher
                    else:
                        match_score = SequenceMatcher(None, job_skill, cv_skill).ratio()
                    
                    # Check for keyword overlap
                    job_words = set(job_skill.split())
                    cv_words = set(cv_skill.split())
                    if job_words.intersection(cv_words):
                        match_score = max(match_score, 0.7)
                    
                    # Check for substring matches
                    if job_skill in cv_skill or cv_skill in job_skill:
                        match_score = max(match_score, 0.8)
                    
                    if match_score > best_match_score and match_score > 0.4:  # Lowered threshold for better matching
                        best_match_score = match_score
                        best_match_skill = cv_skill
                
                if best_match_skill:
                    matched_skills.append(best_match_skill)
                    total_score += best_match_score
        
        # Calculate average score
        similarity_score = total_score / len(job_skills) if job_skills else 0.0
        
        return similarity_score, matched_skills
    
    def calculate_experience_match(self, required_years: Optional[int], cv_years: Optional[float]) -> Tuple[float, str]:
        """Calculate experience matching score."""
        if not required_years:
            return 1.0, "No experience requirement specified"
        
        if not cv_years:
            return 0.0, "No experience information found in CV"
        
        if cv_years >= required_years:
            # Bonus for exceeding requirements
            excess_ratio = min(cv_years / required_years, 2.0)  # Cap at 2x
            score = min(1.0, 0.8 + (excess_ratio - 1.0) * 0.2)
            return score, f"CV has {cv_years} years (required: {required_years})"
        else:
            # Penalty for not meeting requirements
            ratio = cv_years / required_years
            score = max(0.0, ratio * 0.8)  # Max 80% if under requirement
            return score, f"CV has {cv_years} years (required: {required_years})"
    
    def calculate_education_match(self, required_education: Optional[str], cv_education: List[Any]) -> Tuple[float, str]:
        """Calculate education matching score."""
        if not required_education:
            return 1.0, "No education requirement specified"
        
        if not cv_education:
            return 0.0, "No education information found in CV"
        
        required_lower = required_education.lower()
        
        # Education hierarchy (higher is better)
        education_hierarchy = {
            "certificate": 1,
            "diploma": 2,
            "associate": 3,
            "bachelor": 4,
            "master": 5,
            "phd": 6,
            "doctorate": 6
        }
        
        required_level = 0
        for key, level in education_hierarchy.items():
            if key in required_lower:
                required_level = level
                break
        
        if required_level == 0:
            return 1.0, f"Unrecognized education requirement: {required_education}"
        
        # Find highest education level in CV
        cv_highest_level = 0
        cv_highest_degree = ""
        
        for edu in cv_education:
            degree_lower = edu.degree.lower() if hasattr(edu, 'degree') else str(edu).lower()
            for key, level in education_hierarchy.items():
                if key in degree_lower and level > cv_highest_level:
                    cv_highest_level = level
                    cv_highest_degree = edu.degree if hasattr(edu, 'degree') else str(edu)
                    break
        
        if cv_highest_level == 0:
            return 0.0, "No recognized education level found in CV"
        
        if cv_highest_level >= required_level:
            score = 1.0
            message = f"CV meets education requirement: {cv_highest_degree} (required: {required_education})"
        else:
            score = cv_highest_level / required_level
            message = f"CV education below requirement: {cv_highest_degree} (required: {required_education})"
        
        return score, message
    
    def calculate_keyword_match(self, job_keywords: List[str], cv_text: str) -> Tuple[float, List[str]]:
        """Calculate keyword matching score."""
        if not job_keywords or not cv_text:
            return 0.0, []
        
        cv_text_lower = cv_text.lower()
        matched_keywords = []
        
        for keyword in job_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in cv_text_lower:
                matched_keywords.append(keyword)
        
        score = len(matched_keywords) / len(job_keywords) if job_keywords else 0.0
        
        return score, matched_keywords
    
    def calculate_overall_score(self, individual_scores: List[MatchingScore]) -> float:
        """Calculate overall matching score from individual components."""
        if not individual_scores:
            return 0.0
        
        # Weighted average (skills are most important)
        weights = {
            "skills": 0.4,
            "experience": 0.3,
            "education": 0.2,
            "keywords": 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score in individual_scores:
            weight = weights.get(score.component, 0.1)
            weighted_sum += score.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def match_cv_to_job(self, cv: CV, job: JobDescription) -> MatchingResult:
        """Match a CV against a job description."""
        individual_scores = []
        
        # Skills matching
        required_skills = job.required_skills + job.preferred_skills
        skills_score, matched_skills = self.calculate_skill_similarity(required_skills, cv.skills)
        individual_scores.append(MatchingScore(
            component="skills",
            score=skills_score,
            details=f"Matched {len(matched_skills)} out of {len(required_skills)} required skills",
            matched_items=matched_skills
        ))
        
        # Experience matching
        exp_score, exp_details = self.calculate_experience_match(job.experience_years, cv.total_experience_years)
        individual_scores.append(MatchingScore(
            component="experience",
            score=exp_score,
            details=exp_details,
            matched_items=[]
        ))
        
        # Education matching
        edu_score, edu_details = self.calculate_education_match(job.education_level, cv.education)
        individual_scores.append(MatchingScore(
            component="education",
            score=edu_score,
            details=edu_details,
            matched_items=[]
        ))
        
        # Keyword matching
        keyword_score, matched_keywords = self.calculate_keyword_match(job.keywords, cv.full_text)
        individual_scores.append(MatchingScore(
            component="keywords",
            score=keyword_score,
            details=f"Matched {len(matched_keywords)} out of {len(job.keywords)} keywords",
            matched_items=matched_keywords
        ))
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(individual_scores)
        
        # Generate summary
        summary = self._generate_summary(overall_score, individual_scores, cv, job)
        
        return MatchingResult(
            cv_filename=cv.filename,
            candidate_name=cv.name,
            total_score=overall_score,
            individual_scores=individual_scores,
            summary=summary
        )
    
    def _generate_summary(self, overall_score: float, individual_scores: List[MatchingScore], cv: CV, job: JobDescription) -> str:
        """Generate a summary of the matching result."""
        if overall_score >= 0.8:
            strength = "Excellent"
        elif overall_score >= 0.6:
            strength = "Good"
        elif overall_score >= 0.4:
            strength = "Fair"
        else:
            strength = "Poor"
        
        # Find the strongest and weakest components
        scores_by_component = {score.component: score.score for score in individual_scores}
        strongest = max(scores_by_component, key=scores_by_component.get)
        weakest = min(scores_by_component, key=scores_by_component.get)
        
        summary = f"{strength} match ({overall_score:.1%}). "
        summary += f"Strongest area: {strongest} ({scores_by_component[strongest]:.1%}). "
        summary += f"Area for improvement: {weakest} ({scores_by_component[weakest]:.1%})."
        
        if cv.total_experience_years and job.experience_years:
            if cv.total_experience_years >= job.experience_years:
                summary += f" Experience requirement met ({cv.total_experience_years} years)."
            else:
                summary += f" Experience gap: {job.experience_years - cv.total_experience_years:.1f} years short."
        
        return summary
    
    def rank_cvs(self, matching_results: List[MatchingResult]) -> List[MatchingResult]:
        """Rank CVs by their matching scores."""
        # Sort by total score (descending)
        ranked_results = sorted(matching_results, key=lambda x: x.total_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(ranked_results):
            result.rank = i + 1
        
        return ranked_results
