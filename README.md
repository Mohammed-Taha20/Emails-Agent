# 📧 Email Sentiment Analysis with LangChain & NVIDIA LLM

## ⚡ Key Features
- Uses **Meta LLaMA 3.1 (8B Instruct)** via `ChatNVIDIA`.  
- Converts unstructured email text into structured JSON with fields:  
  - `sentiment` (*positive* / *negative*)  
  - `location` (*city* or `"unknown"`)  
  - `product_category` (*furniture, appliances, clothing, kitchenware, beauty, toys, groceries, services, other*)  
- Implements a **LangChain Tool** to compute the most negative product category and location.  
- Integrates with **LangGraph React Agent** for reasoning and tool usage.  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **LangChain** (Core + NVIDIA AI Endpoints)  
- **LangGraph** (React Agent)  
- **Pydantic** for schema validation  
- **JSON Output Parsing**  

---

## 📂 Project Structure
┣ 📄 emails.json # Input dataset (list of customer emails)
┣ 📄 main.py # Main script (this project)
┣ 📄 requirements.txt # Python dependencies
┗ 📄 README.md # Project documentation


## Create a virtual environment & install dependencies:
-create an `.env` file and fill it with your key
- run: `pip install -r requirements.txt`


