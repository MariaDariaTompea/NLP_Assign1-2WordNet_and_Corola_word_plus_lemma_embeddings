# NLP Assignment 1

This project demonstrates Natural Language Processing (NLP) on English and Romanian sentences using two popular Python libraries: **spaCy** and **NLTK**. It analyzes sample sentences and extracts linguistic features such as tokens, lemmas, part-of-speech tags, dependencies, and named entities.

## Files
- `spacy_processor.py`: Processes sentences using spaCy.
- `nltk_processor.py`: Processes sentences using NLTK.
- `sample_sentences.py`: Contains English and Romanian sample sentences.
- `spacy_results.txt`: Output of spaCy analysis.
- `nltk_results.txt`: Output of NLTK analysis.
- `requirements.txt`: Required Python packages.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download spaCy models (if not already installed):
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download ro_core_news_sm
   ```
3. Run the scripts:
   ```bash
   python spacy_processor.py
   python nltk_processor.py
   ```

## Function Descriptions

### spacy_processor.py
- **load_models()**: Loads spaCy language models for English and Romanian.
- **process_sentence(nlp, sentence, language)**: Processes a sentence with spaCy, extracting tokens, lemmas, POS tags, dependencies, and named entities.
- **print_results(results, sentence_num, language)**: Prints the analysis results for a sentence in a readable format.
- **save_results_to_file(all_results, filename)**: Saves all analysis results to a text file.
- **main()**: Orchestrates loading models, processing all sentences, printing, and saving results.

### nltk_processor.py
- **download_nltk_resources()**: Downloads required NLTK resources.
- **process_sentence_nltk(sentence, language)**: Processes a sentence with NLTK, extracting tokens, stems, lemmas, POS tags, chunks, and named entities.
- **print_results_nltk(results, sentence_num, language)**: Prints the analysis results for a sentence in a readable format.
- **save_results_to_file(all_results, filename)**: Saves all analysis results to a text file.
- **main()**: Orchestrates downloading resources, processing all sentences, printing, and saving results.

### sample_sentences.py
- **ENGLISH_SENTENCES**: List of English sentences about Japanese culture.
- **ROMANIAN_SENTENCES**: List of Romanian sentences about Japanese culture.

---

This project is intended for educational purposes to illustrate basic NLP tasks in Python using spaCy and NLTK.
