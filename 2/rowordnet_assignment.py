"""
Assignment Part 1: RoWordNet / WordNet Analysis
"""
import rowordnet
import os
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as pwn

# Ensure nltk data is downloaded
try:
    nltk.data.find('corpora/sentiwordnet')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('sentiwordnet')
    nltk.download('wordnet')

# Initialize RoWordNet
try:
    wn = rowordnet.RoWordNet()
except Exception as e:
    print(f"Error loading RoWordNet: {e}")
    exit(1)

# Words to analyze (3 nouns, 3 verbs, 2 adjectives)
WORDS = {
    'noun': ['casă', 'om', 'timp'],
    'verb': ['alerga', 'mânca', 'vorbi'],
    'adj': ['frumos', 'bun']
}

OUTPUT_FILE = 'rowordnet_results.txt'

def get_pno_scores(sid):
    """
    Calculează scorurile PNO (Pozitiv, Negativ, Obiectiv) folosind SentiWordNet.
    
    Input:
        sid (str): ID-ul synset-ului formatat conform RoWordNet (ex: 'ENG30-08078020-n').
    
    Output:
        str: Un șir de caractere care conține scorurile formatate (ex: 'P=0.0, N=0.0, O=1.0').
    """
    # sid format is often 'ENG30-offset-p'
    try:
        parts = sid.split('-')
        offset = int(parts[1])
        pos = parts[2]
        
        synset = pwn.synset_from_pos_and_offset(pos, offset)
        senti = swn.senti_synset(synset.name())
        
        pos_s = senti.pos_score()
        neg_s = senti.neg_score()
        obj_s = 1.0 - (pos_s + neg_s)
        return f"P={pos_s}, N={neg_s}, O={obj_s}"
    except:
        return "P=0.0, N=0.0, O=1.0"

def get_relations(synset_id):
    """
    Extrage toate relațiile semantice externe ale unui synset dat.
    
    Input:
        synset_id (str): ID-ul synset-ului pentru care se caută relațiile.
    
    Output:
        list: O listă de tupluri (target_id, relation_name).
    """
    relations = wn.outbound_relations(synset_id)
    return relations

def analyze():
    """
    Funcția principală care analizează cuvintele predefinite (substantive, verbe, adjective),
    extrage sensurile, definițiile, literalele și relațiile din RoWordNet și scrie rezultatele într-un fișier.
    
    Input:
        Niciunul (folosește variabila globală WORDS).
    
    Output:
        Niciunul (scrie rezultatele în 'rowordnet_results.txt').
    """
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write("NLP Assignment Part 1: RoWordNet Analysis\n")
        out.write("="*60 + "\n\n")

        for pos, list_words in WORDS.items():
            for word in list_words:
                out.write(f"Cuvânt: {word} ({pos})\n")
                out.write("-" * 40 + "\n")
                
                synset_ids = wn.synsets(literal=word)
                
                if not synset_ids:
                    out.write(f"  Nu s-au găsit synset-uri pentru '{word}'.\n\n")
                    continue
                
                for i, sid in enumerate(synset_ids, 1):
                    synset = wn.synset(sid)
                    out.write(f"  Sense {i} (ID: {sid}):\n")
                    out.write(f"    Definiție: {synset.definition}\n")
                    literals = synset.literals
                    out.write(f"    Literals: {', '.join(literals)}\n")
                    
                    # PNO Scores (Positive, Negative, Objective)
                    scores = get_pno_scores(sid)
                    out.write(f"    PNO Scores: {scores}\n")
                    
                    out.write(f"    Relații semantice:\n")
                    rels = get_relations(sid)
                    if not rels:
                        out.write("      Nicio relație găsită.\n")
                    else:
                        for rel in rels:
                            target_sid = rel[0]
                            rel_type = rel[1]
                            try:
                                target_synset = wn.synset(target_sid)
                                target_literals = ", ".join(target_synset.literals)
                                out.write(f"      - {rel_type}: {target_sid} ({target_literals})\n")
                            except:
                                out.write(f"      - {rel_type}: {target_sid}\n")
                    out.write("\n")
                out.write("\n")

    print(f"RoWordNet results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze()
