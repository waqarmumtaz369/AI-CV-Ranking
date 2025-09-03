"""
ChromaDB vector store implementation for CV ranking system.
"""
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
from models import CV, JobDescription


class VectorStore:
    """ChromaDB-based vector store for CV and job description embeddings."""
    
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the vector store."""
        self.persist_directory = persist_directory
        # Initialize sentence transformer for real embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"✅ Loaded embedding model: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not load embedding model {embedding_model}: {e}")
                print("Falling back to hash-based embeddings")
                self.embedding_model = None
        else:
            print("Warning: sentence-transformers not available, using hash-based embeddings")
            self.embedding_model = None
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collections
        self.cv_collection = self.client.get_or_create_collection(
            name="cvs",
            metadata={"description": "CV embeddings and metadata"}
        )
        
        self.job_collection = self.client.get_or_create_collection(
            name="jobs",
            metadata={"description": "Job description embeddings and metadata"}
        )
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate real embeddings using sentence transformers."""
        if not text or not text.strip():
            return [0.0] * 384
        
        # Use real embeddings if available
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                # Fall back to hash-based embeddings
                pass
        
        # Fallback to hash-based embeddings
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to a list of floats (simulating embeddings)
        embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(32, len(text_hash)), 2)]
        # Pad or truncate to 384 dimensions (standard embedding size)
        while len(embedding) < 384:
            embedding.append(0.0)
        return embedding[:384]
    
    def add_cv(self, cv: CV) -> str:
        """Add a CV to the vector store."""
        cv_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = {
            "filename": cv.filename,
            "original_filename": cv.original_filename or cv.filename,
            "name": cv.name or "Unknown",
            "email": cv.email or "",
            "phone": cv.phone or "",
            "linkedin_url": cv.linkedin_url or "",
            "total_experience_years": cv.total_experience_years or 0,
            "skills_count": len(cv.skills),
            "experience_count": len(cv.experience),
            "education_count": len(cv.education),
            "processed_at": cv.processed_at.isoformat()
        }
        
        # Add skills embeddings
        if cv.skills_embeddings:
            self.cv_collection.add(
                ids=[f"{cv_id}_skills"],
                embeddings=[cv.skills_embeddings],
                documents=[" ".join(cv.skills)],
                metadatas=[{**metadata, "section": "skills"}]
            )
        
        # Add experience embeddings
        if cv.experience_embeddings:
            experience_text = " ".join([exp.description or "" for exp in cv.experience])
            self.cv_collection.add(
                ids=[f"{cv_id}_experience"],
                embeddings=[cv.experience_embeddings],
                documents=[experience_text],
                metadatas=[{**metadata, "section": "experience"}]
            )
        
        # Add education embeddings
        if cv.education_embeddings:
            education_text = " ".join([f"{edu.degree or ''} {edu.field or ''}" for edu in cv.education])
            self.cv_collection.add(
                ids=[f"{cv_id}_education"],
                embeddings=[cv.education_embeddings],
                documents=[education_text],
                metadatas=[{**metadata, "section": "education"}]
            )
        
        # Add projects embeddings
        if cv.projects_embeddings:
            projects_text = " ".join([f"{proj.title or ''} {proj.description or ''}" for proj in cv.projects])
            self.cv_collection.add(
                ids=[f"{cv_id}_projects"],
                embeddings=[cv.projects_embeddings],
                documents=[projects_text],
                metadatas=[{**metadata, "section": "projects"}]
            )
        
        return cv_id
    
    def add_job_description(self, job: JobDescription) -> str:
        """Add a job description to the vector store."""
        job_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = {
            "title": job.title,
            "experience_years": job.experience_years or 0,
            "education_level": job.education_level or "",
            "location": job.location or "",
            "required_skills_count": len(job.required_skills),
            "preferred_skills_count": len(job.preferred_skills),
            "keywords_count": len(job.keywords)
        }
        
        # Add job description embeddings
        if job.embeddings:
            self.job_collection.add(
                ids=[job_id],
                embeddings=[job.embeddings],
                documents=[job.description],
                metadatas=[metadata]
            )
        
        return job_id
    
    def search_similar_cvs(self, job: JobDescription, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for CVs similar to the job description."""
        if not job.embeddings:
            return []
        
        # Search in each section
        results = []
        
        # Search skills
        skills_results = self.cv_collection.query(
            query_embeddings=[job.embeddings],
            n_results=n_results,
            where={"section": "skills"}
        )
        
        # Search experience
        experience_results = self.cv_collection.query(
            query_embeddings=[job.embeddings],
            n_results=n_results,
            where={"section": "experience"}
        )
        
        # Search education
        education_results = self.cv_collection.query(
            query_embeddings=[job.embeddings],
            n_results=n_results,
            where={"section": "education"}
        )
        
        # Search projects
        projects_results = self.cv_collection.query(
            query_embeddings=[job.embeddings],
            n_results=n_results,
            where={"section": "projects"}
        )
        
        # Combine and deduplicate results
        all_results = {}
        
        for section, section_results in [
            ("skills", skills_results),
            ("experience", experience_results),
            ("education", education_results),
            ("projects", projects_results)
        ]:
            if section_results["ids"] and section_results["ids"][0]:
                for i, cv_id in enumerate(section_results["ids"][0]):
                    base_id = cv_id.split("_")[0]  # Remove section suffix
                    distance = section_results["distances"][0][i]
                    metadata = section_results["metadatas"][0][i]
                    
                    if base_id not in all_results:
                        all_results[base_id] = {
                            "cv_id": base_id,
                            "filename": metadata["filename"],
                            "name": metadata["name"],
                            "total_experience_years": metadata["total_experience_years"],
                            "scores": {},
                            "metadata": metadata
                        }
                    
                    # Convert distance to similarity score (1 - distance)
                    similarity_score = max(0, 1 - distance)
                    all_results[base_id]["scores"][section] = similarity_score
        
        # Calculate overall scores
        for cv_data in all_results.values():
            scores = cv_data["scores"]
            if scores:
                cv_data["overall_score"] = sum(scores.values()) / len(scores)
            else:
                cv_data["overall_score"] = 0
        
        # Sort by overall score
        sorted_results = sorted(all_results.values(), key=lambda x: x["overall_score"], reverse=True)
        
        return sorted_results[:n_results]
    
    def get_cv_by_id(self, cv_id: str) -> Optional[Dict[str, Any]]:
        """Get CV data by ID."""
        try:
            # Get all sections for this CV
            results = self.cv_collection.get(
                where={"$contains": {"filename": cv_id.split("_")[0]}}
            )
            
            if results["ids"]:
                return {
                    "cv_id": cv_id,
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                    "documents": results["documents"] if results["documents"] else []
                }
        except Exception:
            pass
        
        return None
    
    def clear_all_data(self):
        """Clear all data from the vector store."""
        try:
            self.client.delete_collection("cvs")
            self.client.delete_collection("jobs")
            
            # Recreate collections
            self.cv_collection = self.client.create_collection(
                name="cvs",
                metadata={"description": "CV embeddings and metadata"}
            )
            
            self.job_collection = self.client.create_collection(
                name="jobs",
                metadata={"description": "Job description embeddings and metadata"}
            )
        except Exception as e:
            print(f"Error clearing data: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collections."""
        try:
            cv_count = self.cv_collection.count()
            job_count = self.job_collection.count()
            
            return {
                "cv_count": cv_count,
                "job_count": job_count,
                "total_embeddings": cv_count + job_count
            }
        except Exception:
            return {
                "cv_count": 0,
                "job_count": 0,
                "total_embeddings": 0
            }
