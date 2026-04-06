"""
Assignment Part 2: COROLA Word Embeddings Analysis
"""
import os
from gensim.models import KeyedVectors
import spacy
import numpy as np

# Path to the COROLA embeddings file
MODEL_PATH = "corola.300.20.vec"
OUTPUT_FILE = "corola_results_updated.txt"

# Load Romanian spaCy model for lemmatization and POS tagging
print("Loading spaCy Romanian model...")
try:
    nlp = spacy.load("ro_core_news_sm")
except:
    os.system("python -m spacy download ro_core_news_sm")
    nlp = spacy.load("ro_core_news_sm")

def load_model(path):
    """
    Încarcă modelul de embedding-uri COROLA în format word2vec. 
    Dacă fișierul nu se află în locația curentă, îl caută recursiv în folderele părinte.
    
    Input:
        path (str): Calea către fișierul .vec.
    
    Output:
        KeyedVectors: Obiectul Gensim care conține vectorii de cuvinte.
    """
    if not os.path.exists(path):
        # Check if it is in the parent directory
        parent_dir = os.path.dirname(os.getcwd())
        potential_path = os.path.join(parent_dir, path)
        if os.path.exists(potential_path):
            path = potential_path
        else:
            # Check if it's inside a folder named the same thing (as seen in list_dir)
            nested_path = os.path.join(parent_dir, path, path)
            if os.path.exists(nested_path):
                path = nested_path

    print(f"Loading embeddings from {path}...")
    return KeyedVectors.load_word2vec_format(path, binary=False)

def get_info(word):
    """
    Folosește spaCy pentru a identifica lema și partea de vorbire (POS) pentru un cuvânt dat.
    
    Input:
        word (str): Cuvântul original.
    
    Output:
        tuple: (lemă, POS, lemă_POS).
    """
    doc = nlp(word)
    token = doc[0]
    return token.lemma_, token.pos_, f"{token.lemma_}_{token.pos_}"

# Input words (10 inputs: nouns, verbs, adjectives)
INPUT_WORDS = [
    "bucurie", "succes", "om",     # Nouns
    "alerga", "mânca", "vorbi",    # Verbs
    "frumos", "bun", "mare", "rece" # Adjectives
]

def main():
    """
    Încarcă modelul COROLA, trece prin cuvintele de input și extrage vecinii apropiați
    pentru Formă, Lemă și Lemă+POS, apoi salvează rezultatele.
    
    Input:
        Niciunul.
    
    Output:
        Niciunul (scrie rezultatele în 'corola_results_updated.txt').
    """
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write("NLP Assignment Part 2: COROLA Word Embeddings\n")
        out.write("Word, Lemma, and Lemma+POS Neighbors\n")
        out.write("="*60 + "\n\n")

        for word in INPUT_WORDS:
            lemma, pos, lemma_pos = get_info(word)
            out.write(f"Input Original: {word}\n")
            out.write(f"Lemma: {lemma} | POS: {pos} | Lemma+POS: {lemma_pos}\n")
            out.write("-" * 40 + "\n")

            # 1. Word Neighbors
            out.write(f"[1] Nearest Neighbors (Word Form: '{word}'):\n")
            if word in model:
                neighbors = model.most_similar(word, topn=10)
                for n, s in neighbors:
                    out.write(f"  - {n} ({s:.4f})\n")
            else:
                out.write("  - Word not in model vocabulary.\n")
            
            # 2. Lemma Neighbors
            out.write(f"\n[2] Nearest Neighbors (Lemma: '{lemma}'):\n")
            if lemma in model and lemma != word:
                neighbors = model.most_similar(lemma, topn=10)
                for n, s in neighbors:
                    out.write(f"  - {n} ({s:.4f})\n")
            elif lemma == word:
                out.write("  - Same as Word Form.\n")
            else:
                out.write(f"  - Lemma '{lemma}' not in model vocabulary.\n")

            # 3. Lemma + POS neighbors
            out.write(f"\n[3] Nearest Neighbors (Lemma+POS: '{lemma_pos}'):\n")
            out.write(f"  - [Note] Using word-model '{lemma}' as representative for Lemma+POS category.\n")
            if lemma in model:
                 out.write(f"  - (Results same as Lemma results in this unified word model)\n")
            else:
                 out.write(f"  - Lemma representation not available.\n")

            out.write("\n" + "="*40 + "\n\n")

        # Analogies section
        out.write("\nAnalogies (Vector Arithmetic)\n")
        out.write("-" * 40 + "\n")
        analogies = [
            ("rege", "bărbat", "femeie", "rege - bărbat + femeie"),
            ("Paris", "Franța", "România", "Paris - Franța + România"),
            ("bun", "bine", "rău", "bun - bine + rău")
        ]

        for a, b, c, text in analogies:
            out.write(f"Analogy: {text}\n")
            try:
                res = model.most_similar(positive=[a, c], negative=[b], topn=3)
                for n, s in res:
                    out.write(f"  -> {n} ({s:.4f})\n")
            except Exception as e:
                 out.write(f"  Error: {e}\n")
            out.write("\n")

    print(f"COROLA results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
