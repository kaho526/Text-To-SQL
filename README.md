# 🚀 Text-to-SQL NLP Engine: Generalisation to Unseen Templates, and a Leakage Audit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![NLP](https://img.shields.io/badge/NLP-Transformers%20%7C%20Text--to--SQL-success)

## 📌 Project Overview
Translating natural-language flight questions into SQL (ATIS dataset) is easy when test questions reuse SQL templates seen in training (question split), and much harder when every test template is new (query split).

This group course project (Natural Language Processing, University of Sydney, 2025) asked whether a data-centric approach, blending question-split data into query-split training, could help a template-based model generalise to unseen SQL templates. A later audit showed that it could not, and why.

## 🛠️ Technical Architecture & Pipeline
* **Data Pipeline:** Built a Python pipeline that turns the raw ATIS JSON into question–SQL pairs grouped by SQL template.
* **Models:** Built linear, feedforward and LSTM classifiers in **PyTorch** (a teammate built the Transformer). Each maps a question to one of 944 SQL templates and tags the tokens that fill the template's slots.
* **Data-Quantity Ablation:** On the question split, the LSTM rose from 17.00% to 27.06% exact match as training data grew from 25% to 100%.
* **Data-Mixing Experiments:** Blended question-split data into query-split training for the LSTM at three ratios (25%, 50% and 75% question-split data).
* **LLM Prompting:** Built a few-shot prompting and exact-match evaluation pipeline for **Phi-4-reasoning-plus** through the OpenRouter API (0, 5 and 40 shots), the only part of the project that generates free-form SQL.

## 🔍 Leakage Audit
The original write-up reported query-split accuracy rising from **0% to 23.05%** after mixing. Re-checking it traced the gain to training–test overlap between the two dataset splits, not generalisation:

* **Source:** the query split itself is clean (its training set shares 0 of the 82 test templates). But the question split assigns individual sentences to splits, so its training and dev sets contain sentences from templates the query split holds out for test.
* **Scale:** the 50/50 mix put 47 of the 82 test templates into training, affecting 298 of 347 test examples.
* **Re-scored:** on test templates that never appear in training, accuracy is **0%** in both LSTM runs with saved results. Every correct answer came from a leaked template.

| Run | Reported (all test) | Seen templates | Unseen templates only |
|---|---|---|---|
| 50% question-split mix | 80/347 = 23.05% | 80/298 = 26.85% | **0/49 = 0.00%** |
| 25% question-split mix | 51/347 = 14.70% | 51/285 = 17.89% | **0/62 = 0.00%** |

This is what the design should produce: a template classifier can only output templates it was trained on, so mixing data cannot fix its structural limit on a query split.

The leak also reaches individual sentences, not just templates: 301 of the 347 query-split test sentences appear verbatim in the question split's train (269) or dev (32) set. In the 50% run, 77 of the 80 correct answers were sentences of that kind.

### Leak-free rerun (October 2026)
To confirm, I reran the experiment with one change: before training, every example whose SQL template is held out for the query-split test set was removed from the mixed training data and from the dev set. A leaky control on the same code checks that the setup reproduces the original result.

| Run (LSTM, query-split test) | Removed from training | Training examples | Test accuracy |
|---|---|---|---|
| 50% mix, leaky control | 0 | 3,629 | 89/347 = 25.65% |
| 50% mix, leak-free | 128 train + 32 dev | 3,470 | **0/347 = 0.00%** |
| 25% mix, leak-free | 72 train + 32 dev | 3,730 | **0/347 = 0.00%** |

The leak-free models still trained normally, reaching about 36% on the question-split dev set (0.361 for both), so the 0% is the template classifier's structural limit, not a broken run. The control's 25.65% sits close to the original 23.05%.

How the rerun differs from the original: the experiment script was edited after the May 2025 runs, so the rerun matches the original run's saved settings (two-layer bidirectional LSTM, 256-dimensional embeddings and hidden state, 30 epochs, 50/50 and 25/75 mixes with the question-split dev set). Learning rate and dropout weren't recorded, so the script's defaults were used (1e-4 and 0.2). PyTorch was not seeded, so each figure comes from a single run.

## 📊 Key Findings
* **Data mixing doesn't create generalisation:** the query-split gains came entirely from leaked templates. With the leak removed, accuracy on the query-split test set is 0% at both mixing ratios.
* **More data helps on the question split:** LSTM 17.00% → 27.06% from 25% to 100% of the training data.
* **Best saved Transformer run (teammate's model):** 41.83% (187/447) exact match on the question split, where test templates also appear in training.

## ⚠️ Limitations
* Exact match marks semantically equivalent SQL as wrong.
* The few-shot prompting outputs weren't saved, and the script scored the dev set, so no LLM results are reported here.
* Some earlier ablation runs weren't saved, so only results with saved outputs are reported.

## 🧭 What I'd Do Next
* Try a model that generates SQL token by token, which can in principle produce templates it has never seen
* Report execution accuracy alongside exact match
* Save every run's predictions and metrics, so any result can be re-scored later

## 💻 Tech Stack
* **Languages & Frameworks:** Python, PyTorch, SQL
* **Techniques:** NLP, deep learning, prompt engineering, data ablation studies, leakage auditing
* **Models:** linear, feedforward, LSTM and Transformer template classifiers; Phi-4-reasoning-plus (prompting)
