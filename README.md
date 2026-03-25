# 🚀 Text-to-SQL NLP Engine: Zero-Shot Generalization via Strategic Data Mixing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![NLP](https://img.shields.io/badge/NLP-Transformers%20%7C%20Llama%203B-success)

## 📌 Project Overview
Translating natural language flight inquiries into executable SQL queries (ATIS dataset) is severely hindered by complex nested data structures and the need for generalization across unseen query types (Query-Splits). 

This project tackles the **Zero-Shot classification failure** problem in Text-to-SQL tasks. Instead of merely tweaking model architectures, this project introduces a **data-centric approach**, utilizing strategic data-mixing and rigorous ablation studies to force model generalization.

## 🛠️ Technical Architecture & Pipeline
* **Robust ETL Pipeline:** Developed a Python-based data pipeline to ingest, clean, and transform raw JSON datasets into structured SQL templates, systematically resolving complex alignment issues.
* **Model Benchmarking:** Implemented and evaluated multiple architectures from scratch using **PyTorch**, including LSTMs, Feedforward networks, and Transformer-based Seq2Seq models with Attention mechanisms.
* **LLM Prompt Engineering:** Engineered Few-Shot prompts for **Llama 3B**, leveraging in-context learning to autonomously diagnose reconstruction errors and evaluate pipeline bottlenecks.

## 📊 Key Findings & Business Impact
Through rigorous data ablation studies focusing on the ratio of question-splits vs. query-splits, the following breakthroughs were achieved:
* **Overcame Zero-Shot Failures:** By intelligently blending linguistically rich datasets (e.g., a 50/50 mix), the pipeline successfully forced model generalization on completely unseen SQL templates, transforming the baseline accuracy from **0% to over 23.05%**.
* **Exponential Performance Surge:** Systematically debugging alignment errors and optimizing data mixtures surged overall model accuracy by **24x** compared to initial baselines (peaking at a leading 38.25% for Transformers).
* **Data-Centric Proof:** Proved that scaling in-domain data and strategic data mixing yield significantly higher performance gains than standard architectural hyperparameter tuning.

## 💻 Tech Stack
* **Languages & Frameworks:** Python, PyTorch, SQL
* **Techniques:** Natural Language Processing (NLP), Deep Learning, Prompt Engineering, ETL Data Pipelines, Data Ablation Studies
* **Models:** Transformers, LSTM, Seq2Seq + Attention, Llama 3B
