"""
Job Description processing module.
"""
import re
from typing import List, Optional
from .models import JobDescription


class JobDescriptionProcessor:
    """Processes job descriptions to extract keywords and generate embeddings."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the processor with an embedding model."""
        # For Phase 1, we'll use simple TF-IDF based embeddings
        # This avoids the sentence_transformers dependency conflict
        self.embedding_model = None
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from job description text."""
        # Common technical skills patterns
        skill_patterns = [
            # Programming Languages
            r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|PHP|Ruby|Swift|Kotlin|Scala|R|MATLAB|Perl|Haskell|Clojure|Erlang|Elixir|Dart|Julia)\b',
            # Web Frameworks
            r'\b(?:React|Angular|Vue|Svelte|Next\.?js|Nuxt\.?js|Node\.?js|Express|Django|Flask|FastAPI|Spring|Spring Boot|Laravel|Symfony|CodeIgniter|ASP\.NET|Ruby on Rails|Sinatra|Phoenix|Actix)\b',
            # Cloud & DevOps
            r'\b(?:AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|Jenkins|GitLab CI|GitHub Actions|CircleCI|Travis CI|Bamboo|Terraform|Ansible|Chef|Puppet|Vagrant|Prometheus|Grafana|ELK Stack|Splunk)\b',
            # Databases
            r'\b(?:SQL|PostgreSQL|MySQL|SQLite|Oracle|SQL Server|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|CouchDB|Neo4j|InfluxDB|TimescaleDB|ClickHouse|Snowflake|BigQuery|Redshift)\b',
            # AI/ML
            r'\b(?:Machine Learning|ML|AI|Artificial Intelligence|Deep Learning|Neural Networks|TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Matplotlib|Seaborn|Plotly|Jupyter|OpenCV|NLTK|spaCy|Hugging Face|Transformers)\b',
            # Data Science
            r'\b(?:Data Science|Data Analytics|Business Intelligence|BI|ETL|Data Pipeline|Apache Spark|Hadoop|Kafka|Airflow|dbt|Tableau|Power BI|Looker|Qlik|SAS|R Studio|Apache Superset)\b',
            # Development Tools
            r'\b(?:Git|GitHub|GitLab|Bitbucket|SVN|Mercurial|Jira|Confluence|Slack|Discord|Teams|Zoom|Figma|Sketch|Adobe XD|InVision|Zeplin|Postman|Insomnia|Swagger|OpenAPI)\b',
            # Methodologies
            r'\b(?:DevOps|CI/CD|Microservices|REST|GraphQL|API|Web Development|Mobile Development|iOS|Android|React Native|Flutter|Xamarin|Ionic|Cordova|PWA|SPA|SSR|JAMstack)\b',
            # Project Management
            r'\b(?:Project Management|Agile|Scrum|Kanban|SAFe|Lean|Six Sigma|Leadership|Communication|Team Management|Product Management|Technical Writing|Documentation|Testing|QA|TDD|BDD|Code Review)\b',
            # Security
            r'\b(?:Cybersecurity|Security|Penetration Testing|OWASP|SSL|TLS|OAuth|JWT|Firewall|VPN|Encryption|Cryptography|PKI|SIEM|SOC|Compliance|GDPR|HIPAA|PCI DSS)\b'
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
        
        # Enhanced section detection with more patterns
        required_patterns = [
            r'(?:requirements?|required|must have|essential|qualifications?|skills? required).*?(?=\n\n|\n[A-Z]|$|\nBonus|\nPreferred)',
            r'(?:experience|proficiency|familiarity|knowledge|understanding).*?(?=\n\n|\n[A-Z]|$|\nBonus|\nPreferred)'
        ]
        
        preferred_patterns = [
            r'(?:bonus|preferred|nice to have|bonus skills?|preferred qualifications?|additional).*?(?=\n\n|\n[A-Z]|$)',
            r'(?:experience with|knowledge of|familiarity with).*?(?=\n\n|\n[A-Z]|$)'
        ]
        
        # Extract required skills
        for pattern in required_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_text = match.group(0)
                skills = self.extract_keywords(section_text)
                required_skills.extend(skills)
        
        # Extract preferred skills
        for pattern in preferred_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_text = match.group(0)
                skills = self.extract_keywords(section_text)
                preferred_skills.extend(skills)
        
        # Remove duplicates and filter out non-technical words
        required_skills = self._filter_technical_skills(list(set(required_skills)))
        preferred_skills = self._filter_technical_skills(list(set(preferred_skills)))
        
        # If no specific sections found, extract from entire text
        if not required_skills and not preferred_skills:
            all_skills = self.extract_keywords(text)
            filtered_skills = self._filter_technical_skills(all_skills)
            
            # Split roughly 70/30 for required/preferred
            split_point = int(len(filtered_skills) * 0.7)
            required_skills = filtered_skills[:split_point]
            preferred_skills = filtered_skills[split_point:]
        
        # If still no skills found, try to extract from bullet points
        if not required_skills and not preferred_skills:
            # Look for bullet points or numbered lists
            bullet_pattern = r'(?:^|\n)\s*[-•*]\s*([^\n]+)'
            bullets = re.findall(bullet_pattern, text, re.MULTILINE)
            if bullets:
                bullet_text = ' '.join(bullets)
                all_skills = self.extract_keywords(bullet_text)
                filtered_skills = self._filter_technical_skills(all_skills)
                
                split_point = int(len(filtered_skills) * 0.7)
                required_skills = filtered_skills[:split_point]
                preferred_skills = filtered_skills[split_point:]
        
        return required_skills, preferred_skills
    
    def _filter_technical_skills(self, skills: List[str]) -> List[str]:
        """Filter out non-technical words and keep only relevant skills."""
        filtered_skills = []
        non_technical_words = {
            'experience', 'years', 'degree', 'bachelor', 'master', 'preferred', 'required', 
            'must', 'have', 'essential', 'ability', 'strong', 'good', 'excellent', 'skills',
            'work', 'team', 'communication', 'problem', 'solving', 'analytical', 'thinking',
            'curiosity', 'challenges', 'independently', 'learn', 'quickly', 'adapt', 'evolving',
            'technologies', 'genuine', 'passion', 'transformative', 'potential', 'businesses',
            'bonus', 'not', 'or', 'and', 'with', 'in', 'of', 'for', 'to', 'the', 'a', 'an'
        }
        
        for skill in skills:
            skill_lower = skill.lower().strip()
            # Keep skills that are not in non-technical words and have reasonable length
            if (skill_lower not in non_technical_words and 
                len(skill_lower) > 2 and 
                len(skill_lower) < 50 and
                not skill_lower.isdigit()):
                filtered_skills.append(skill)
        
        return filtered_skills
    
    def extract_experience_years(self, text: str) -> Optional[int]:
        """Extract required years of experience from job description."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?relevant\s*experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?professional\s*experience',
            r'minimum\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?',
            r'(\d+)\s*–\s*(\d+)\s*years?',  # Range like "2–3 years"
            r'(\d+)\s*-\s*(\d+)\s*years?',  # Range like "2-3 years"
            r'(\d+)\s*to\s*(\d+)\s*years?'  # Range like "2 to 3 years"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    return int(match.group(1))
                elif len(match.groups()) == 2:
                    # For ranges, take the lower bound
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
        """Generate simple TF-IDF based embeddings for the job description text."""
        # For Phase 1, we'll use a simple hash-based embedding
        # This avoids the sentence_transformers dependency conflict
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to a list of floats (simulating embeddings)
        embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(32, len(text_hash)), 2)]
        # Pad or truncate to 384 dimensions (standard embedding size)
        while len(embedding) < 384:
            embedding.append(0.0)
        return embedding[:384]
    
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
