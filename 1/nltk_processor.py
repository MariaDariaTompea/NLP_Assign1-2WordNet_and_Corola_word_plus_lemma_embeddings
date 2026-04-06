"""
NLP Processing using NLTK
Performs: tokenization, stemming, POS tagging, lemmatization, chunking, NER
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk import RegexpParser
from sample_sentences import ENGLISH_SENTENCES, ROMANIAN_SENTENCES


def download_nltk_resources():
    """Download required NLTK resources"""
    print("Downloading NLTK resources...")
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'wordnet',
        'omw-1.4'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    print("NLTK resources downloaded!\n")


def process_sentence_nltk(sentence, language):
    """Process a single sentence with NLTK"""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    results = {
        'original': sentence,
        'tokens': [],
        'stems': [],
        'lemmas': [],
        'pos_tags': [],
        'chunks': [],
        'entities': []
    }
    
    # Tokenization
    tokens = word_tokenize(sentence)
    results['tokens'] = tokens
    
    # Stemming (only for English)
    if language == "English":
        results['stems'] = [stemmer.stem(token) for token in tokens]
        
        # Lemmatization
        results['lemmas'] = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        results['stems'] = ["N/A (Romanian)"] * len(tokens)
        results['lemmas'] = ["N/A (Romanian)"] * len(tokens)
    
    # POS Tagging
    results['pos_tags'] = pos_tag(tokens)
    
    # Named Entity Recognition (only for English with good accuracy)
    if language == "English":
        try:
            ne_tree = ne_chunk(results['pos_tags'])
            for subtree in ne_tree:
                if hasattr(subtree, 'label'):
                    entity_text = " ".join([token for token, pos in subtree.leaves()])
                    results['entities'].append({
                        'text': entity_text,
                        'label': subtree.label()
                    })
        except Exception as e:
            print(f"Error in NER: {e}")
    
    # Chunking (Noun Phrases)
    grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN.*>+}
        VP: {<VB.*><NP|PP|CLAUSE>+$}
        PP: {<IN><NP>}
    """
    cp = RegexpParser(grammar)
    try:
        chunk_tree = cp.parse(results['pos_tags'])
        for subtree in chunk_tree:
            if hasattr(subtree, 'label'):
                chunk_text = " ".join([token for token, pos in subtree.leaves()])
                results['chunks'].append({
                    'text': chunk_text,
                    'type': subtree.label()
                })
    except Exception as e:
        print(f"Error in chunking: {e}")
    
    return results


def print_results_nltk(results, sentence_num, language):
    """Print formatted NLTK results"""
    print(f"\n{'='*80}")
    print(f"{language.upper()} SENTENCE {sentence_num}")
    print(f"{'='*80}")
    print(f"Original: {results['original']}")
    
    print(f"\nTokens: {results['tokens']}")
    
    if language == "English":
        print(f"\nStems: {results['stems']}")
        print(f"\nLemmas: {results['lemmas']}")
    
    print("\nPOS Tags:")
    for i, (token, pos) in enumerate(results['pos_tags'], 1):
        print(f"  {i}. {token}: {pos}")
    
    if results['chunks']:
        print("\nChunks:")
        for chunk in results['chunks']:
            print(f"  [{chunk['type']}] {chunk['text']}")
    else:
        print("\nChunks: None found")
    
    if results['entities']:
        print("\nNamed Entities:")
        for ent in results['entities']:
            print(f"  {ent['text']}: {ent['label']}")
    else:
        print("\nNamed Entities: None found")


def save_results_to_file(all_results, filename="nltk_results.txt"):
    """Save all NLTK results to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for lang, lang_results in all_results.items():
            f.write(f"\n{'#'*80}\n")
            f.write(f"# {lang.upper()} SENTENCES - NLTK ANALYSIS\n")
            f.write(f"{'#'*80}\n")
            
            for i, result in enumerate(lang_results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"SENTENCE {i}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Original: {result['original']}\n")
                
                f.write(f"\nTokens: {result['tokens']}\n")
                
                if "N/A" not in str(result['stems'][0]):
                    f.write(f"\nStems: {result['stems']}\n")
                    f.write(f"\nLemmas: {result['lemmas']}\n")
                
                f.write("\nPOS Tags:\n")
                for j, (token, pos) in enumerate(result['pos_tags'], 1):
                    f.write(f"  {j}. {token}: {pos}\n")
                
                if result['chunks']:
                    f.write("\nChunks:\n")
                    for chunk in result['chunks']:
                        f.write(f"  [{chunk['type']}] {chunk['text']}\n")
                else:
                    f.write("\nChunks: None found\n")
                
                if result['entities']:
                    f.write("\nNamed Entities:\n")
                    for ent in result['entities']:
                        f.write(f"  {ent['text']}: {ent['label']}\n")
                else:
                    f.write("\nNamed Entities: None found\n")
    
    print(f"\n\nResults saved to {filename}")


def main():
    """Main processing function for NLTK"""
    download_nltk_resources()
    
    all_results = {
        'English': [],
        'Romanian': []
    }
    
    # Process English sentences
    print("\n" + "="*80)
    print("PROCESSING ENGLISH SENTENCES WITH NLTK")
    print("="*80)
    for i, sentence in enumerate(ENGLISH_SENTENCES, 1):
        results = process_sentence_nltk(sentence, "English")
        all_results['English'].append(results)
        print_results_nltk(results, i, "English")
    
    # Process Romanian sentences
    print("\n\n" + "="*80)
    print("PROCESSING ROMANIAN SENTENCES WITH NLTK")
    print("="*80)
    for i, sentence in enumerate(ROMANIAN_SENTENCES, 1):
        results = process_sentence_nltk(sentence, "Romanian")
        all_results['Romanian'].append(results)
        print_results_nltk(results, i, "Romanian")
    
    # Save results to file
    save_results_to_file(all_results)


if __name__ == "__main__":
    main()
