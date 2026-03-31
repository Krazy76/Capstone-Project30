# Capstone-Project30
# Project 30 – Automated Multi-level Information Discovery for Financial Intelligence

## Overview
This project builds a data engineering and auto-labelling pipeline for financial news articles. The goal is to transform raw unstructured JSON news data into a structured Silver Dataset that can support downstream financial intelligence tasks.

The pipeline covers:
- automated JSON ingestion
- dataset cleaning and deduplication
- rule-based pre-labelling
- local LLM auto-labelling using Mistral
- final Silver Dataset construction

## Project Pipeline
### 1. JSON Ingestion
The raw financial news articles were stored as JSON files across multiple folders.  
The ingestion notebook recursively reads all JSON files and extracts the main article fields and metadata into a tabular dataset.

### 2. Dataset Cleaning
The cleaned dataset removes weak rows and duplicate articles, and standardizes the main text fields for later labelling.

### 3. Pre-labelling
A rule-based pre-labelling step assigns initial Macro, Industry, and Entity signals using:
- keyword matching
- entity metadata from the source JSON files

This step also creates a `needs_llm` flag to identify unresolved articles that should be passed to the local LLM.

### 4. Local LLM Labelling
Only rows marked with `needs_llm = 1` are sent to the local Mistral model.  
The model returns JSON labels for:
- `macro_tags`
- `industry_tags`
- `entity_tags`

### 5. Silver Dataset Construction
The labelled outputs are merged back into the cleaned dataset to form the final Silver Dataset.

## Main Notebooks
- `Automated-json-ingestion.ipynb` – raw JSON ingestion
- `creating-prelabeled_dataset.ipynb` – rule-based pre-labelling and unresolved row selection
- `Test-for-need-llm-labelling.ipynb` – local Mistral labelling of unresolved rows

## Outputs
Main outputs produced in this phase include:
- cleaned raw dataset
- prelabelled dataset
- labelled Mistral batches
- merged Silver Dataset

## Notes on Large Files
The final `silver_dataset_final.csv` is not stored directly in this repository because the file is too large for standard GitHub file tracking. A smaller sample or alternative storage method should be used for sharing the full dataset.

## Tools Used
- Python
- pandas
- Ollama
- Mistral
- Jupyter Notebook

## Current Status
Phase 2 Data Engineering and Auto-Labelling pipeline has been implemented, including:
- automated ingestion
- cleaning
- pre-labelling
- local LLM labelling
- Silver Dataset generation
