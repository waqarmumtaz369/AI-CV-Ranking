**Step 1 - Process the Job Description (JD)**
- Extract Keywords
- Generate Embeddings 
- Create a Pydantic Object (from the extracted keywords)

**Step 2 - Process the Collection of Curriculum Vitae (CVs)**
- PDF file -> Textual Data -> Process through LLM
- Extract the LinkedIn Profile Link from the processed data in the previous step
- Aggregate the duration of experience from the CV 
- Store the Processed Data

**Step 3 - Cross-Verify with the LinkedIn Profile (I want to skip this in the first iteration)**
- Crawl the LinkedIn Profile 
- Aggregate the duration of experience from the LinkedIn Profile  
- Cross-verify the LinkedIn Data with the data stored 
- Apply rules like intersection or rejection, and modify the candidate stored data

**Step 4 - Create a Matching Matrix**
- That shows the properties (like skills, years of experience, etc.) matching of every CV with the reference job description
- Calculate and assign the sum of the assigned numbers 
- Show the top `N` matching CVs
