# NLP Assignment Part 2: Semantic Analysis & Word Embeddings

This repository contains the source code and result files for the second NLP assignment, focusing on the Romanian language.

## 1. Overview of the Code

### Part 1: RoWordNet Semantic Analysis (`rowordnet_assignment.py`)
This script uses the **RoWordNet (Romanian WordNet)** library to explore the hierarchical structure and semantic relations of Romanian words.
- **Goal:** For 3 nouns, 3 verbs, and 2 adjectives, find every sense (Synset) and its related semantic data.
- **What it does:**
    - Extracts the **Definition** and the list of **Literals** (synonyms) for each sense.
    - Calculates **PNO Scores** (Positive, Negative, Objective) for every sense by mapping RoWordNet IDs to the SentiWordNet database using NLTK.
    - Crawls the **Semantic Hierarchy** to display relations like `hypernyms`, `hyponyms`, `meronyms`, etc., including both the Synset ID and the actual Romanian words for the related concept.
- **Output:** The detailed analysis is saved in **`rowordnet_results.txt`**.

### Part 2: COROLA Word Embeddings (`corola_assignment.py`)
This script implements distributional semantics using the **COROLA Corpus** (300-dimension vector space).
- **Goal:** Analyze 10 input words across three categories: raw word form, base lemma, and lemma + POS tag.
- **What it does:**
    - Uses **spaCy (Romanian model)** to accurately lemmatize and identify the Part-of-Speech (POS) tag for each input word.
    - Calculates the **10 Nearest Neighbors** for the Word, Lemma, and Lemma+POS versions of the word using **Cosine Similarity**.
    - Displays similarity scores for each neighbor (0.0 to 1.0).
    - Performs **Vector Analogies** (arithmetic) to show semantic directions (e.g., `rege - bărbat + femeie = regină`).
- **Output:** The analysis results are saved in **`corola_results_updated.txt`**.

---

## 2. Requirements
- **Python Libraries:** `rowordnet`, `gensim`, `spacy`, `nltk`, `numpy`.
- **Model Files:** `corola.300.20.vec` (included in the workspace).
- **AI Models:** Romanian spaCy model (`ro_core_news_sm`).

---

## 3. Results Significance
- **RoWordNet results** provide a dictionary-style breakdown of word senses and their place in the hierarchy of language.
- **COROLA results** show how computer models understand word meaning through usage patterns, demonstrating that the model can find synonyms (Concept neighbors) and relationship patterns (Analogies).