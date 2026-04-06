"""
NLP Processing using Spacy
Performs: tokenization, POS tagging, lemmatization, dependency parsing, NER
"""

import spacy
from sample_sentences import ENGLISH_SENTENCES, ROMANIAN_SENTENCES

from pathlib import Path
from spacy import displacy
import re


def load_models():
    """Load Spacy language models"""
    print("Loading Spacy models...")
    try:
        nlp_en = spacy.load("en_core_web_sm")
        nlp_ro = spacy.load("ro_core_news_sm")
        print("Models loaded successfully!\n")
        return nlp_en, nlp_ro
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please install models using:")
        print("  python -m spacy download en_core_web_sm")
        print("  python -m spacy download ro_core_news_sm")
        return None, None


def process_sentence(nlp, sentence, language):
    """Process a single sentence with Spacy"""
    doc = nlp(sentence)
    
    results = {
        'original': sentence,
        'tokens': [],
        'lemmas': [],
        'pos_tags': [],
        'dependencies': [],
        'entities': []
    }
    
    # Tokenization, Lemmatization, POS Tagging
    for token in doc:
        results['tokens'].append(token.text)
        results['lemmas'].append(token.lemma_)
        results['pos_tags'].append((token.text, token.pos_, token.tag_))
        
        # Dependency relations
        results['dependencies'].append({
            'text': token.text,
            'dep': token.dep_,
            'head': token.head.text,
            'head_pos': token.head.pos_
        })
    
    # Named Entity Recognition
    for ent in doc.ents:
        results['entities'].append({
            'text': ent.text,
            'label': ent.label_,
            'description': spacy.explain(ent.label_)
        })
    
    return results


def print_results(results, sentence_num, language):
    """Print formatted results"""
    print(f"\n{'='*80}")
    print(f"{language.upper()} SENTENCE {sentence_num}")
    print(f"{'='*80}")
    print(f"Original: {results['original']}")
    
    print(f"\nTokens: {results['tokens']}")
    
    print(f"\nLemmas: {results['lemmas']}")
    
    print("\nPOS Tags:")
    for i, (token, pos, tag) in enumerate(results['pos_tags'], 1):
        print(f"  {i}. {token}: {pos} ({tag})")
    
    print("\nDependency Relations:")
    for dep in results['dependencies']:
        print(f"  {dep['text']} --[{dep['dep']}]--> {dep['head']} ({dep['head_pos']})")
    
    if results['entities']:
        print("\nNamed Entities:")
        for ent in results['entities']:
            print(f"  {ent['text']}: {ent['label']} - {ent['description']}")
    else:
        print("\nNamed Entities: None found")


def save_results_to_file(all_results, filename="spacy_results.txt"):
    """Save all results to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for lang, lang_results in all_results.items():
            f.write(f"\n{'#'*80}\n")
            f.write(f"# {lang.upper()} SENTENCES - SPACY ANALYSIS\n")
            f.write(f"{'#'*80}\n")
            
            for i, result in enumerate(lang_results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"SENTENCE {i}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Original: {result['original']}\n")
                
                f.write(f"\nTokens: {result['tokens']}\n")
                f.write(f"\nLemmas: {result['lemmas']}\n")
                
                f.write("\nPOS Tags:\n")
                for j, (token, pos, tag) in enumerate(result['pos_tags'], 1):
                    f.write(f"  {j}. {token}: {pos} ({tag})\n")
                
                f.write("\nDependency Relations:\n")
                for dep in result['dependencies']:
                    f.write(f"  {dep['text']} --[{dep['dep']}]--> {dep['head']} ({dep['head_pos']})\n")
                
                if result['entities']:
                    f.write("\nNamed Entities:\n")
                    for ent in result['entities']:
                        f.write(f"  {ent['text']}: {ent['label']} - {ent['description']}\n")
                else:
                    f.write("\nNamed Entities: None found\n")
    
    print(f"\n\nResults saved to {filename}")


def main():
    """Main processing function"""
    # Load models
    nlp_en, nlp_ro = load_models()
    if nlp_en is None or nlp_ro is None:
        return
    
    all_results = {
        'English': [],
        'Romanian': []
    }
    
    # Process English sentences
    print("\n" + "="*80)
    print("PROCESSING ENGLISH SENTENCES WITH SPACY")
    print("="*80)
    for i, sentence in enumerate(ENGLISH_SENTENCES, 1):
        results = process_sentence(nlp_en, sentence, "English")
        all_results['English'].append(results)
        print_results(results, i, "English")
    
    # Process Romanian sentences
    print("\n\n" + "="*80)
    print("PROCESSING ROMANIAN SENTENCES WITH SPACY")
    print("="*80)
    for i, sentence in enumerate(ROMANIAN_SENTENCES, 1):
        results = process_sentence(nlp_ro, sentence, "Romanian")
        all_results['Romanian'].append(results)
        print_results(results, i, "Romanian")
    
    # Save results to file
    save_results_to_file(all_results)

    # Save dependency visualizations for all examples
    save_dep_all(nlp_en, ENGLISH_SENTENCES, "en")
    save_dep_all(nlp_ro, ROMANIAN_SENTENCES, "ro")

def save_dep_all(nlp_model, sentences, prefix):
    out_dir = Path("deps_spacy")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(sentences, start=1):
        doc = nlp_model(s)
        html = displacy.render(doc, style="dep", jupyter=False)
        fname_html = out_dir / f"{prefix}_{i}.html"
        fname_html.write_text(html, encoding="utf-8")
        print(f"Wrote dependency HTML to {fname_html.resolve()}")

        # extract svg from displaCy HTML and save as .svg
        m = re.search(r"(<svg[\s\S]*?</svg>)", html)
        if m:
            svg = m.group(1)
            fname_svg = out_dir / f"{prefix}_{i}.svg"
            fname_svg.write_text(svg, encoding="utf-8")
            print(f"Wrote dependency SVG to {fname_svg.resolve()}")
        else:
            print("No SVG found in displaCy HTML; saved HTML only.")


if __name__ == "__main__":
    main()
