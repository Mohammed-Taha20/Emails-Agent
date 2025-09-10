# 📧 Email Sentiment Analysis with LangChain & NVIDIA LLM

## 📌 Project Description
This project analyzes customer emails using **LangChain**, **NVIDIA AI Endpoints**, and **LangGraph** to extract structured insights.  
The pipeline performs:  
1. **Sentiment Analysis** – Determines whether each email is *positive* or *negative*.  
2. **Entity Extraction** – Identifies the **location (city)** and **product category** mentioned in each email.  
3. **Aggregated Insights** – Finds the **most negatively mentioned product category** and the **location with the highest number of negative mentions**.  

The final output is a structured JSON object plus a summary of key negative trends.  

---

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


