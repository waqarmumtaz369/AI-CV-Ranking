For the first phase, here are the actions of implmentation 
We will not strive for performance or speed here 
We will not strive for extensive implementation with a lot of edge use cases 
Our main foucs is to deliver a working prototype with enough progress that can be matured and new features added later 

## Major Tech Components
- Use uv venv and activate it using `source .venv/bin/activate`
- vllm is already installed and feel free to add required packages when needed
- Use streamlit for UI prototyping 
- Use chroma db for all the vector operations 
- Use local model served through vllm currently it's QWEN (run_vllm.sh can serve the model)

## UI Flow
- A Title
- A textbox to paste the Job Description(JD)
- An option to select a folder path having one or more CVs (process multiple CVs but limit to 5-10 for Phase 1 to keep it manageable)
- An action button to start the matching process
- Any ui indicator that shows the ongoing processing 
- A place holder to display the results 
- An option to export all the results in such a suitable format that can be assessed later (even if we delete the intermediate data and clear all chroma db, it must make sense in deciison making)

## Logical Flow
- Refer to the-task.md and tentative-flow.md documents for more implementation details 
- Use your common sense to reduce the intensity of a task with respect to the fact that it is first phase, not the complete MVP  
- Implement in a modular form that can be scaled up later and easily testable 
- Use prompts (promt engineering) where applicable
- User will paste the Job Description (upload feature can be added later)
- User will select the folder path where one or more CV files are already there 
- Process Job Description and store that in a suitable way (in memory or in a separate file but not in chroma db) that can help matching process later  
- Parse the CV text, send to vLLM-served model for structured data extraction (skills, experience, education), then store in Chroma with embeddings
- Dont store the complete CV as a single vector. Rather store one resume/cv in such a key value where values are the more than one embeddings separate for skills education etc. for better searching and matching in later stages)
- Match on: skills (semantic similarity), experience years, education level
- Use Chroma for semantic similarity + exact matching for structured data
- Calculate match scores using semantic similarity (Chroma) + exact matching for experience/education requirements
- Show the matching results on UI in a suitable way and prepare the data to be exported for later usage 