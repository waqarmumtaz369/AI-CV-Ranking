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
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
from models import CV, Experience, Education, Project, Certification


class CVProcessor:
    """Processes CV/Resume files to extract structured information."""
    
    def __init__(self, vllm_url: str = "http://localhost:8000", embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the CV processor."""
        self.vllm_url = vllm_url
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
                "model": "Qwen/Qwen2.5-0.5B",
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
        You are an expert CV parser. Extract comprehensive information from this CV text and return ONLY a valid JSON object.

        CV TEXT:
        {cv_text}

        Extract and return this COMPLETE JSON structure with REAL data from the CV:
        {{
            "name": "John Smith",
            "email": "john@email.com", 
            "phone": "+1234567890",
            "linkedin_url": "https://linkedin.com/in/johnsmith",
            "github_url": "https://github.com/johnsmith",
            "portfolio_url": "https://johnsmith.dev",
            "location": "San Francisco, CA",
            "professional_summary": "Software Engineer with 3+ years in backend systems, specializing in Python and distributed microservices.",
            "skills": ["Python", "JavaScript", "React", "AWS", "Docker"],
            "experience": [
                {{
                    "title": "Senior Software Engineer",
                    "company": "Tech Corp",
                    "start_date": "2020-01",
                    "end_date": "2023-12",
                    "description": "Led development of microservices architecture, improved API response time by 40%",
                    "skills_used": ["Python", "React", "AWS", "Docker"]
                }}
            ],
            "education": [
                {{
                    "degree": "Bachelor of Science",
                    "field": "Computer Science", 
                    "institution": "Stanford University",
                    "graduation_year": 2019,
                    "gpa": 3.7
                }}
            ],
            "projects": [
                {{
                    "title": "E-commerce Platform",
                    "project_type": "Personal",
                    "tech_stack": ["React", "Node.js", "MongoDB"],
                    "description": "Full-stack e-commerce application with payment integration",
                    "github_url": "https://github.com/johnsmith/ecommerce",
                    "demo_url": "https://ecommerce-demo.com"
                }}
            ],
            "certifications": [
                {{
                    "name": "AWS Certified Solutions Architect",
                    "issuer": "Amazon Web Services",
                    "date_earned": "2022-06",
                    "expiry_date": "2025-06"
                }}
            ],
            "awards": ["Best Hackathon Project 2021", "Dean's List 2018-2019"],
            "publications": ["Building Scalable Microservices", "AI in Web Development"],
            "languages": ["English (Native)", "Spanish (Fluent)"]
        }}

        EXTRACTION RULES:
        1. HEADER/CONTACT INFO: Extract name (usually at top), email, phone, LinkedIn, GitHub, portfolio URLs, location
        2. PROFESSIONAL SUMMARY: Look for 2-3 line overview of expertise and career focus
        3. TECHNICAL SKILLS: Extract programming languages, frameworks, tools, databases, platforms
        4. PROFESSIONAL EXPERIENCE: Extract job title, company, dates, key responsibilities with impact/metrics, skills used
        5. PROJECTS: Extract project title, type (Personal/Academic/Open Source/Freelance), tech stack, description, GitHub/demo URLs
        6. EDUCATION: Extract degree, field, institution, graduation year, GPA, honors
        7. CERTIFICATIONS: Extract certification name, issuer, date earned, expiry date
        8. ADDITIONAL SECTIONS: Extract awards, publications, languages, achievements

        IMPORTANT:
        - Extract ONLY REAL data from the CV, not placeholder examples
        - Use null for missing information
        - For skills, include both technical and soft skills mentioned
        - For experience, focus on achievements with metrics when available
        - For projects, prioritize those with GitHub links or detailed descriptions
        - Return ONLY the JSON object, no other text or explanations
        """
        
        try:
            response = self.call_vllm(prompt, max_tokens=2000)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                llm_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                llm_data = json.loads(response)
            
            # Clean up placeholder text
            llm_data = self._clean_placeholder_text(llm_data)
            
            # HYBRID APPROACH: Combine LLM and fallback extraction
            fallback_data = self._fallback_extraction(cv_text)
            hybrid_data = self._combine_extraction_results(llm_data, fallback_data)
            
            return hybrid_data
                
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to basic extraction if LLM fails
            return self._fallback_extraction(cv_text)
    
    def _clean_placeholder_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up placeholder text from LLM response."""
        # List of common placeholder texts
        placeholder_patterns = [
            'actual_full_name_from_cv',
            'full name of the candidate',
            'actual_email@domain.com',
            'actual_phone_number',
            'actual_linkedin_url',
            'john smith',
            'john@email.com',
            '+1234567890',
            'https://linkedin.com/in/johnsmith',
            'abc corp',
            'university of xyz',
            'tech company inc',
            'example',
            'placeholder',
            'n/a',
            'not specified',
            'not mentioned'
        ]
        
        # Clean name field
        if 'name' in data and data['name']:
            name = str(data['name']).lower().strip()
            if any(pattern in name for pattern in placeholder_patterns):
                data['name'] = None
        
        # Clean email field
        if 'email' in data and data['email']:
            email = str(data['email']).lower().strip()
            if any(pattern in email for pattern in placeholder_patterns):
                data['email'] = None
        
        # Clean phone field
        if 'phone' in data and data['phone']:
            phone = str(data['phone']).lower().strip()
            if any(pattern in phone for pattern in placeholder_patterns):
                data['phone'] = None
        
        # Clean LinkedIn field
        if 'linkedin_url' in data and data['linkedin_url']:
            linkedin = str(data['linkedin_url']).lower().strip()
            if any(pattern in linkedin for pattern in placeholder_patterns):
                data['linkedin_url'] = None
        
        # Clean skills list
        if 'skills' in data and data['skills']:
            cleaned_skills = []
            for skill in data['skills']:
                skill_str = str(skill).lower().strip()
                if not any(pattern in skill_str for pattern in placeholder_patterns):
                    cleaned_skills.append(skill)
            data['skills'] = cleaned_skills
        
        # Clean experience list
        if 'experience' in data and data['experience']:
            cleaned_experience = []
            for exp in data['experience']:
                if isinstance(exp, dict):
                    # Clean company name
                    if 'company' in exp and exp['company']:
                        company = str(exp['company']).lower().strip()
                        if any(pattern in company for pattern in placeholder_patterns):
                            exp['company'] = None
                    
                    # Clean job title
                    if 'title' in exp and exp['title']:
                        title = str(exp['title']).lower().strip()
                        if any(pattern in title for pattern in placeholder_patterns):
                            exp['title'] = None
                    
                    cleaned_experience.append(exp)
            data['experience'] = cleaned_experience
        
        # Clean education list
        if 'education' in data and data['education']:
            cleaned_education = []
            for edu in data['education']:
                if isinstance(edu, dict):
                    # Clean institution name
                    if 'institution' in edu and edu['institution']:
                        institution = str(edu['institution']).lower().strip()
                        if any(pattern in institution for pattern in placeholder_patterns):
                            edu['institution'] = None
                    
                    # Clean degree
                    if 'degree' in edu and edu['degree']:
                        degree = str(edu['degree']).lower().strip()
                        if any(pattern in degree for pattern in placeholder_patterns):
                            edu['degree'] = None
                    
                    cleaned_education.append(edu)
            data['education'] = cleaned_education
        
        return data
    
    def _combine_extraction_results(self, llm_data: Dict[str, Any], fallback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine LLM and fallback extraction results for better accuracy."""
        combined_data = {}
        
        # For each field, prefer LLM data if it's meaningful, otherwise use fallback
        for field in ['name', 'email', 'phone', 'linkedin_url', 'github_url', 'portfolio_url', 'location', 'professional_summary']:
            llm_value = llm_data.get(field)
            fallback_value = fallback_data.get(field)
            
            # Prefer LLM value if it's not None and not placeholder text
            if llm_value and not self._is_placeholder_text(llm_value):
                combined_data[field] = llm_value
            elif fallback_value:
                combined_data[field] = fallback_value
            else:
                combined_data[field] = None
        
        # For lists, combine and deduplicate
        for field in ['skills', 'awards', 'publications', 'languages']:
            llm_list = llm_data.get(field, [])
            fallback_list = fallback_data.get(field, [])
            
            # Combine and remove duplicates
            combined_list = list(set(llm_list + fallback_list))
            # Remove placeholder items
            combined_list = [item for item in combined_list if not self._is_placeholder_text(item)]
            combined_data[field] = combined_list
        
        # For complex objects (experience, education, projects, certifications)
        # Prefer LLM data but supplement with fallback data
        for field in ['experience', 'education', 'projects', 'certifications']:
            llm_list = llm_data.get(field, [])
            fallback_list = fallback_data.get(field, [])
            
            # Start with LLM data
            combined_list = llm_list.copy()
            
            # Add fallback data if LLM data is insufficient
            if len(combined_list) < len(fallback_list):
                # Add missing items from fallback
                for fallback_item in fallback_list:
                    if not self._item_exists_in_list(fallback_item, combined_list):
                        combined_list.append(fallback_item)
            
            combined_data[field] = combined_list
        
        return combined_data
    
    def _is_placeholder_text(self, text: str) -> bool:
        """Check if text is placeholder/generic text."""
        if not text:
            return True
        
        placeholder_patterns = [
            'john smith', 'jane doe', 'example', 'placeholder', 'n/a', 'not specified',
            'not mentioned', 'actual_', 'full name of', 'your name', 'candidate name',
            'john@email.com', 'example@email.com', 'your.email@domain.com',
            '+1234567890', 'your phone', 'phone number', 'your linkedin',
            'linkedin.com/in/yourname', 'github.com/yourusername'
        ]
        
        text_lower = str(text).lower().strip()
        return any(pattern in text_lower for pattern in placeholder_patterns)
    
    def _item_exists_in_list(self, item: Dict[str, Any], item_list: List[Dict[str, Any]]) -> bool:
        """Check if an item already exists in a list of dictionaries."""
        if not item or not item_list:
            return False
        
        # Simple comparison based on key fields
        item_key = self._get_item_key(item)
        for existing_item in item_list:
            if self._get_item_key(existing_item) == item_key:
                return True
        return False
    
    def _get_item_key(self, item: Dict[str, Any]) -> str:
        """Generate a key for item comparison."""
        if 'title' in item and 'company' in item:
            # Experience item
            return f"{item.get('title', '')}_{item.get('company', '')}"
        elif 'degree' in item and 'institution' in item:
            # Education item
            return f"{item.get('degree', '')}_{item.get('institution', '')}"
        elif 'name' in item:
            # Certification item
            return item.get('name', '')
        elif 'title' in item:
            # Project item
            return item.get('title', '')
        else:
            return str(item)
    
    def _fallback_extraction(self, cv_text: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns if LLM fails."""
        data = {
            "name": None,
            "email": None,
            "phone": None,
            "linkedin_url": None,
            "github_url": None,
            "portfolio_url": None,
            "location": None,
            "professional_summary": None,
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "awards": [],
            "publications": [],
            "languages": []
        }
        
        # Extract name (usually at the top of the CV)
        lines = cv_text.split('\n')
        for i, line in enumerate(lines[:15]):  # Check first 15 lines
            line = line.strip()
            if len(line) > 3 and len(line) < 50:  # Reasonable name length
                # Skip common CV headers and patterns
                skip_patterns = [
                    'resume', 'cv', 'curriculum vitae', 'contact', 'email', 'phone', 'address',
                    'objective', 'summary', 'profile', 'experience', 'education', 'skills',
                    'projects', 'certifications', 'languages', 'references', 'linkedin',
                    'github', 'portfolio', 'website', 'www.', 'http', 'https', '@',
                    'tel:', 'phone:', 'email:', 'mobile:', 'address:', 'location:',
                    'date of birth', 'dob', 'nationality', 'marital status'
                ]
                
                if not any(pattern in line.lower() for pattern in skip_patterns):
                    # Check if it looks like a name (contains letters, spaces, dots, hyphens)
                    if re.match(r'^[A-Za-z\s\.\-\']+$', line):
                        # Must have at least 2 words and reasonable length
                        words = line.split()
                        if len(words) >= 2 and len(words) <= 4:
                            # Check if words are reasonable length (not too short/long)
                            if all(2 <= len(word) <= 20 for word in words):
                                data["name"] = line
                                break
        
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
        
        # Extract GitHub URL
        github_match = re.search(r'https?://(?:www\.)?github\.com/[A-Za-z0-9-]+/?', cv_text)
        if github_match:
            data["github_url"] = github_match.group(0)
        
        # Extract Portfolio URL (common patterns)
        portfolio_patterns = [
            r'https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[A-Za-z0-9.-]*)?',  # General URLs
            r'www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[A-Za-z0-9.-]*)?'  # www URLs
        ]
        for pattern in portfolio_patterns:
            portfolio_match = re.search(pattern, cv_text)
            if portfolio_match and 'linkedin.com' not in portfolio_match.group(0) and 'github.com' not in portfolio_match.group(0):
                data["portfolio_url"] = portfolio_match.group(0)
                break
        
        # Extract location (city, country patterns) - look in first few lines
        location_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})',  # City, State (most common)
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # City, Country
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})'  # City, State, Country
        ]
        
        # Look for location in first 10 lines (header area)
        header_lines = '\n'.join(lines[:10])
        for pattern in location_patterns:
            location_match = re.search(pattern, header_lines)
            if location_match:
                data["location"] = location_match.group(0)
                break
        
        # Extract professional summary (look for objective/summary sections)
        summary_patterns = [
            r'(?:objective|summary|profile|about)[:\s]*([^\n]+(?:\n[^\n]+){0,2})',
            r'(?:professional\s+summary|career\s+objective)[:\s]*([^\n]+(?:\n[^\n]+){0,2})'
        ]
        for pattern in summary_patterns:
            summary_match = re.search(pattern, cv_text, re.IGNORECASE)
            if summary_match:
                data["professional_summary"] = summary_match.group(1).strip()
                break
        
        # Extract comprehensive skills (technical terms)
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
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, cv_text, re.IGNORECASE)
            data["skills"].extend(matches)
        
        data["skills"] = list(set(data["skills"]))  # Remove duplicates
        
        # Extract work experience information
        experience_patterns = [
            r'(?:Software Engineer|Developer|Programmer|Analyst|Consultant|Manager|Lead|Senior|Junior|Intern|Trainee|Assistant|Specialist|Architect|Designer|Administrator|Coordinator|Executive|Director|Head|Chief|President|Vice President|VP|CEO|CTO|CFO|COO|Founder|Co-founder|Owner|Freelancer|Contractor|Consultant)',
            r'(?:Experience|Work History|Employment|Professional Experience|Career|Positions|Roles|Jobs)',
            r'(?:Company|Organization|Corporation|Inc|LLC|Ltd|GmbH|Pvt|Private|Public|Government|Non-profit|Startup|Enterprise)',
            r'(?:202[0-9]|201[0-9]|202[0-9]|Present|Current|Ongoing|Till Date|To Date)',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        ]
        
        # Look for experience sections
        experience_sections = []
        for pattern in experience_patterns:
            matches = re.finditer(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                # Get context around the match
                start = max(0, match.start() - 200)
                end = min(len(cv_text), match.end() + 200)
                context = cv_text[start:end]
                experience_sections.append(context)
        
        # Extract experience entries from sections
        experience_entries = []
        for section in experience_sections:
            # Look for job titles and companies
            job_title_pattern = r'([A-Z][A-Za-z\s]+(?:Engineer|Developer|Analyst|Manager|Consultant|Specialist|Architect|Designer|Administrator|Coordinator|Executive|Director|Lead|Senior|Junior|Intern))'
            company_pattern = r'([A-Z][A-Za-z\s&]+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Technologies|Systems|Solutions|Services|Group|International|Global))'
            
            job_titles = re.findall(job_title_pattern, section)
            companies = re.findall(company_pattern, section)
            
            # Look for date patterns
            date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{4}|\d{1,2}/\d{4}|\d{4}-\d{4}'
            dates = re.findall(date_pattern, section, re.IGNORECASE)
            
            # Create experience entries
            for i, title in enumerate(job_titles):
                company = companies[i] if i < len(companies) else "Unknown Company"
                start_date = dates[i] if i < len(dates) else None
                end_date = dates[i + 1] if i + 1 < len(dates) else "Present"
                
                experience_entries.append({
                    "title": title.strip(),
                    "company": company.strip(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "description": f"Worked as {title} at {company}",
                    "skills_used": []
                })
        
        data["experience"] = experience_entries
        
        # Extract education information
        education_patterns = [
            r'(?:Bachelor|Master|PhD|Doctorate|Associate|Diploma|Certificate)\s*(?:of\s*)?(?:Science|Arts|Engineering|Business|Computer Science|Information Technology|Software Engineering|Data Science|Machine Learning|Artificial Intelligence|Cybersecurity|Network Engineering|Database Administration|Web Development|Mobile Development|DevOps|Cloud Computing|Project Management|Marketing|Finance|Accounting|Economics|Mathematics|Physics|Chemistry|Biology|Psychology|Sociology|Political Science|History|Literature|Philosophy|Education|Medicine|Law|Nursing|Pharmacy|Dentistry|Veterinary|Architecture|Civil Engineering|Mechanical Engineering|Electrical Engineering|Chemical Engineering|Industrial Engineering|Aerospace Engineering|Biomedical Engineering|Environmental Engineering|Materials Engineering|Nuclear Engineering|Petroleum Engineering|Systems Engineering|Software Engineering|Computer Engineering|Information Systems|Information Technology|Computer Science|Data Science|Machine Learning|Artificial Intelligence|Cybersecurity|Network Engineering|Database Administration|Web Development|Mobile Development|DevOps|Cloud Computing|Project Management|Marketing|Finance|Accounting|Economics|Mathematics|Physics|Chemistry|Biology|Psychology|Sociology|Political Science|History|Literature|Philosophy|Education|Medicine|Law|Nursing|Pharmacy|Dentistry|Veterinary|Architecture|Civil Engineering|Mechanical Engineering|Electrical Engineering|Chemical Engineering|Industrial Engineering|Aerospace Engineering|Biomedical Engineering|Environmental Engineering|Materials Engineering|Nuclear Engineering|Petroleum Engineering|Systems Engineering)',
            r'(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|PhD|Ph\.?D\.?|Associate|Diploma|Certificate)',
        ]
        
        # Extract graduation years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, cv_text)
        
        # Extract GPA
        gpa_pattern = r'\b(?:GPA|CGPA|Grade Point Average)[:\s]*(\d+\.?\d*)\b'
        gpa_match = re.search(gpa_pattern, cv_text, re.IGNORECASE)
        gpa = None
        if gpa_match:
            try:
                gpa_val = float(gpa_match.group(1))
                if 0.0 <= gpa_val <= 4.0:
                    gpa = gpa_val
            except ValueError:
                pass
        
        # Try to extract education entries
        education_entries = []
        for pattern in education_patterns:
            matches = re.finditer(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                degree_text = match.group(0)
                # Try to find associated institution and year
                context_start = max(0, match.start() - 100)
                context_end = min(len(cv_text), match.end() + 100)
                context = cv_text[context_start:context_end]
                
                # Extract institution name (look for university, college, institute)
                institution_pattern = r'([A-Z][A-Za-z\s&]+(?:University|College|Institute|School|Academy|Polytechnic))'
                institution_match = re.search(institution_pattern, context)
                institution = institution_match.group(1).strip() if institution_match else "Unknown Institution"
                
                # Extract graduation year from context
                year_match = re.search(year_pattern, context)
                graduation_year = int(year_match.group(0)) if year_match else None
                
                education_entries.append({
                    "degree": degree_text,
                    "field": "Unknown Field",
                    "institution": institution,
                    "graduation_year": graduation_year,
                    "gpa": gpa
                })
        
        data["education"] = education_entries
        
        # Extract projects
        project_patterns = [
            r'(?:project|portfolio|work)[:\s]*([^\n]+(?:\n[^\n]+){0,3})',
            r'([A-Z][A-Za-z\s]+(?:Project|App|Platform|System|Website|Application))[:\s]*([^\n]+(?:\n[^\n]+){0,2})'
        ]
        projects = []
        for pattern in project_patterns:
            matches = re.finditer(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                project_text = match.group(0)
                # Try to extract GitHub URL from project context
                github_match = re.search(r'https?://(?:www\.)?github\.com/[A-Za-z0-9-]+/?', project_text)
                github_url = github_match.group(0) if github_match else None
                
                projects.append({
                    "title": match.group(1) if len(match.groups()) > 0 else "Project",
                    "project_type": "Personal",  # Default assumption
                    "tech_stack": [],
                    "description": project_text,
                    "github_url": github_url,
                    "demo_url": None
                })
        data["projects"] = projects
        
        # Extract certifications
        cert_patterns = [
            r'([A-Z][A-Za-z\s]+(?:Certified|Certification|Certificate))[:\s]*([A-Z][A-Za-z\s]+)',
            r'(?:certification|certificate)[:\s]*([^\n]+)'
        ]
        certifications = []
        for pattern in cert_patterns:
            matches = re.finditer(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                certifications.append({
                    "name": match.group(1) if len(match.groups()) > 0 else match.group(0),
                    "issuer": match.group(2) if len(match.groups()) > 1 else "Unknown",
                    "date_earned": None,
                    "expiry_date": None
                })
        data["certifications"] = certifications
        
        # Extract awards
        award_patterns = [
            r'(?:award|achievement|honor|recognition)[:\s]*([^\n]+)',
            r'([A-Z][A-Za-z\s]+(?:Award|Prize|Recognition|Honor))[:\s]*([^\n]+)'
        ]
        awards = []
        for pattern in award_patterns:
            matches = re.finditer(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                awards.append(match.group(1) if len(match.groups()) > 0 else match.group(0))
        data["awards"] = list(set(awards))  # Remove duplicates
        
        # Extract publications
        pub_patterns = [
            r'(?:publication|paper|article|blog|talk)[:\s]*([^\n]+)',
            r'"([^"]+)"[:\s]*(?:published|presented|talk)'
        ]
        publications = []
        for pattern in pub_patterns:
            matches = re.finditer(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                publications.append(match.group(1))
        data["publications"] = list(set(publications))  # Remove duplicates
        
        # Extract languages
        lang_patterns = [
            r'(?:language|languages)[:\s]*([^\n]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[:\s]*(?:native|fluent|intermediate|beginner)'
        ]
        languages = []
        for pattern in lang_patterns:
            matches = re.finditer(pattern, cv_text, re.IGNORECASE)
            for match in matches:
                languages.append(match.group(1))
        data["languages"] = list(set(languages))  # Remove duplicates
        
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
    
    def _log_extracted_data(self, filename: str, structured_data: Dict[str, Any], total_experience: float):
        """Log extracted CV data to console for debugging."""
        print(f"\n{'='*60}")
        print(f"📄 PROCESSED CV: {filename}")
        print(f"{'='*60}")
        print(f"Name: {structured_data.get('name', 'Not found')}")
        print(f"Email: {structured_data.get('email', 'Not found')}")
        print(f"Phone: {structured_data.get('phone', 'Not found')}")
        print(f"Location: {structured_data.get('location', 'Not found')}")
        print(f"LinkedIn: {structured_data.get('linkedin_url', 'Not found')}")
        print(f"GitHub: {structured_data.get('github_url', 'Not found')}")
        print(f"Portfolio: {structured_data.get('portfolio_url', 'Not found')}")
        print(f"Experience: {total_experience} years")
        print(f"Professional Summary: {structured_data.get('professional_summary', 'Not found')}")
        
        # Skills
        skills = structured_data.get('skills', [])
        print(f"Skills ({len(skills)}): {', '.join(skills) if skills else 'None found'}")
        
        # Experience
        experience = structured_data.get('experience', [])
        print(f"Experience Entries ({len(experience)}):")
        for i, exp in enumerate(experience):
            print(f"  {i+1}. {exp.get('title', 'Unknown')} at {exp.get('company', 'Unknown')} ({exp.get('start_date', 'Unknown')} - {exp.get('end_date', 'Present')})")
            if exp.get('description'):
                print(f"     Description: {exp.get('description', '')[:100]}...")
        
        # Education
        education = structured_data.get('education', [])
        print(f"Education Entries ({len(education)}):")
        for i, edu in enumerate(education):
            print(f"  {i+1}. {edu.get('degree', 'Unknown')} in {edu.get('field', 'Unknown')} from {edu.get('institution', 'Unknown')} ({edu.get('graduation_year', 'Unknown')})")
        
        # Projects
        projects = structured_data.get('projects', [])
        print(f"Projects ({len(projects)}):")
        for i, proj in enumerate(projects):
            print(f"  {i+1}. {proj.get('title', 'Unknown')} - {proj.get('description', '')[:100]}...")
        
        # Certifications
        certifications = structured_data.get('certifications', [])
        print(f"Certifications ({len(certifications)}):")
        for i, cert in enumerate(certifications):
            print(f"  {i+1}. {cert.get('name', 'Unknown')} from {cert.get('issuer', 'Unknown')}")
        
        # Awards
        awards = structured_data.get('awards', [])
        if awards:
            print(f"Awards ({len(awards)}): {', '.join(awards)}")
        
        # Publications
        publications = structured_data.get('publications', [])
        if publications:
            print(f"Publications ({len(publications)}): {', '.join(publications)}")
        
        # Languages
        languages = structured_data.get('languages', [])
        if languages:
            print(f"Languages ({len(languages)}): {', '.join(languages)}")
        
        print(f"{'='*60}\n")
    
    def process_cv(self, pdf_path: str, original_filename: Optional[str] = None) -> CV:
        """Process a CV file and return structured data."""
        # Use original filename if provided, otherwise use the PDF path basename
        filename = original_filename if original_filename else os.path.basename(pdf_path)
        
        # Extract text from PDF
        cv_text = self.extract_text_from_pdf(pdf_path)
        
        # Extract structured data using LLM
        structured_data = self.extract_structured_data(cv_text)
        
        # Create Experience objects
        experience_objects = []
        for exp_data in structured_data.get("experience", []):
            # Only create experience objects if we have meaningful data
            if exp_data.get("title") or exp_data.get("company") or exp_data.get("description"):
                experience_objects.append(Experience(
                    title=exp_data.get("title"),
                    company=exp_data.get("company"),
                    start_date=exp_data.get("start_date"),
                    end_date=exp_data.get("end_date"),
                    description=exp_data.get("description"),
                    skills_used=exp_data.get("skills_used", [])
                ))
        
        # Create Education objects
        education_objects = []
        for edu_data in structured_data.get("education", []):
            # Only create education objects if we have meaningful data
            if edu_data.get("degree") or edu_data.get("institution") or edu_data.get("field"):
                education_objects.append(Education(
                    degree=edu_data.get("degree"),
                    field=edu_data.get("field"),
                    institution=edu_data.get("institution"),
                    graduation_year=edu_data.get("graduation_year"),
                    gpa=edu_data.get("gpa")
                ))
        
        # Create Project objects
        project_objects = []
        for proj_data in structured_data.get("projects", []):
            # Only create project objects if we have meaningful data
            if proj_data.get("title") or proj_data.get("description"):
                project_objects.append(Project(
                    title=proj_data.get("title"),
                    project_type=proj_data.get("project_type"),
                    tech_stack=proj_data.get("tech_stack", []),
                    description=proj_data.get("description"),
                    github_url=proj_data.get("github_url"),
                    demo_url=proj_data.get("demo_url")
                ))
        
        # Create Certification objects
        certification_objects = []
        for cert_data in structured_data.get("certifications", []):
            # Only create certification objects if we have meaningful data
            if cert_data.get("name") or cert_data.get("issuer"):
                certification_objects.append(Certification(
                    name=cert_data.get("name"),
                    issuer=cert_data.get("issuer"),
                    date_earned=cert_data.get("date_earned"),
                    expiry_date=cert_data.get("expiry_date")
                ))
        
        # Calculate total experience
        total_experience = self.calculate_experience_duration(structured_data.get("experience", []))
        
        # Generate embeddings for different sections
        skills_text = " ".join(structured_data.get("skills", []))
        experience_text = " ".join([exp.get("description", "") for exp in structured_data.get("experience", [])])
        education_text = " ".join([f"{edu.get('degree', '')} {edu.get('field', '')}" for edu in structured_data.get("education", [])])
        projects_text = " ".join([f"{proj.get('title', '')} {proj.get('description', '')}" for proj in structured_data.get("projects", [])])
        
        skills_embeddings = self.generate_embeddings(skills_text) if skills_text else None
        experience_embeddings = self.generate_embeddings(experience_text) if experience_text else None
        education_embeddings = self.generate_embeddings(education_text) if education_text else None
        projects_embeddings = self.generate_embeddings(projects_text) if projects_text else None
        
        # Final validation - ensure no placeholder text
        final_name = structured_data.get("name")
        if final_name and any(placeholder in final_name.lower() for placeholder in ['actual_full_name', 'full name of', 'example', 'placeholder']):
            final_name = None
        
        final_email = structured_data.get("email")
        if final_email and any(placeholder in final_email.lower() for placeholder in ['actual_email', 'example', 'placeholder']):
            final_email = None
        
        # Log extracted data to console
        self._log_extracted_data(original_filename or os.path.basename(pdf_path), structured_data, total_experience)
        
        return CV(
            filename=filename,
            original_filename=original_filename,
            full_text=cv_text,
            name=final_name,
            email=final_email,
            phone=structured_data.get("phone"),
            linkedin_url=structured_data.get("linkedin_url"),
            github_url=structured_data.get("github_url"),
            portfolio_url=structured_data.get("portfolio_url"),
            location=structured_data.get("location"),
            professional_summary=structured_data.get("professional_summary"),
            skills=structured_data.get("skills", []),
            experience=experience_objects,
            education=education_objects,
            projects=project_objects,
            certifications=certification_objects,
            awards=structured_data.get("awards", []),
            publications=structured_data.get("publications", []),
            languages=structured_data.get("languages", []),
            total_experience_years=total_experience,
            skills_embeddings=skills_embeddings,
            experience_embeddings=experience_embeddings,
            education_embeddings=education_embeddings,
            projects_embeddings=projects_embeddings
        )
