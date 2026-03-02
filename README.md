#  AI Feature Prioritization Copilot

An AI-powered product intelligence tool that transforms raw user feedback into prioritized roadmap decisions and auto-generated Product Requirements Documents (PRDs).

🔗 **Live App:**  
https://ai-feature-prioritization-copilot-3oowkxvenpgd3rn8zm3mpz.streamlit.app/

---

##  Overview

Product teams often struggle to convert large volumes of unstructured user feedback into clear, data-driven roadmap decisions. This project automates that workflow using NLP and LLMs.

The application:

- analyzes raw feedback  
- detects key product themes  
- prioritizes features using the RICE framework  
- generates structured PRDs using Groq LLM  

---

##  Key Features

-  AI theme detection from user feedback  
-  Embedding-based clustering (Sentence Transformers)  
-  RICE-based feature prioritization  
-  Recommended next feature selection  
-  Automatic PRD generation via Groq  
-  Live deployment with Streamlit Cloud  
-  Interactive dashboard with charts  

---

##  Tech Stack

- Python  
- Streamlit  
- Sentence Transformers  
- Scikit-learn  
- Pandas  
- Matplotlib  
- Groq LLM API  

---

##  How It Works

1. User uploads feedback CSV  
2. Text is converted into embeddings  
3. KMeans groups feedback into themes  
4. Themes are ranked using RICE scoring  
5. Top feature is selected  
6. Groq LLM generates a structured PRD  
7. Results displayed in interactive UI  

