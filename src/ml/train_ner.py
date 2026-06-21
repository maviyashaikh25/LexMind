import spacy, json, random, os
from spacy.training import Example
from spacy.util import minibatch, compounding
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


CUSTOM_LABELS = [
    "PETITIONER", "RESPONDENT", "JUDGE", "COURT",
    "STATUTE", "SECTION", "CITATION", "DATE"
]

def map_label(tag_name: str) -> str:
    """Map opennyaiorg/InLegalNER labels to the project expected labels."""
    prefix = ""
    if tag_name.startswith("B-"):
        prefix = "B-"
        label = tag_name[2:]
    elif tag_name.startswith("I-"):
        prefix = "I-"
        label = tag_name[2:]
    else:
        return "O"
        

    if label == "PROVISION":
        label = "SECTION"
    elif label == "PRECEDENT":
        label = "CITATION"
        
    if label in CUSTOM_LABELS:
        return f"{prefix}{label}"
    else:
        return "O"

def load_ner_data_from_hf(split="train") -> list:
    """Load hf-tuner/indian-legal-ner dataset and convert to spaCy training format."""
    hf_split = "validation" if split in ["validation", "test"] else "train"
    print(f"Loading '{hf_split}' split of hf-tuner/indian-legal-ner from Hugging Face...")
    dataset = load_dataset("hf-tuner/indian-legal-ner")
    label_list = dataset[hf_split].features["ner_tags"].feature.names
    
    data = []
    for row in dataset[hf_split]:
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        
        tags = []
        for tag_id in ner_tags:
            tag_name = label_list[tag_id]
            tags.append(map_label(tag_name))
            
        text = " ".join(tokens)
        
        # Compute token start and end offsets in the joined text
        offsets = []
        current_offset = 0
        for token in tokens:
            token_start = text.find(token, current_offset)
            if token_start == -1:
                token_start = current_offset
            token_end = token_start + len(token)
            offsets.append((token_start, token_end))
            current_offset = token_end + 1
            
        entities = []
        active_ent = None  # [start_char, label]
        
        for idx, tag in enumerate(tags):
            if tag == "O":
                if active_ent is not None:
                    # End active entity
                    entities.append((active_ent[0], offsets[idx-1][1], active_ent[1]))
                    active_ent = None
            elif tag.startswith("B-"):
                if active_ent is not None:
                    # End previous entity
                    entities.append((active_ent[0], offsets[idx-1][1], active_ent[1]))
                # Start new entity
                label = tag.split("-")[-1]
                active_ent = [offsets[idx][0], label]
            elif tag.startswith("I-"):
                label = tag.split("-")[-1]
                if active_ent is not None:
                    if active_ent[1] == label:
                        # Continue active entity
                        pass
                    else:
                        # Label mismatch, end previous, start new
                        entities.append((active_ent[0], offsets[idx-1][1], active_ent[1]))
                        active_ent = [offsets[idx][0], label]
                else:
                    # No active entity, treat I- tag as B-
                    active_ent = [offsets[idx][0], label]
                    
        if active_ent is not None:
            entities.append((active_ent[0], offsets[-1][1], active_ent[1]))
            
        data.append((text, {"entities": entities}))
    return data

def get_mock_ner_data() -> list:
    import random
    petitioners = ["Amit Patel", "Rajesh Sharma", "Kamal Nath", "Sita Devi"]
    respondents = ["Union of India", "State of Maharashtra", "M/s XYZ Ltd.", "DCP Delhi"]
    judges = ["Justice D.Y. Chandrachud", "Justice Sanjiv Khanna", "Justice Hrishikesh Roy", "Justice B.V. Nagarathna"]
    courts = ["Supreme Court of India", "Delhi High Court", "Bombay High Court", "District Court of Pune"]
    statutes = ["Indian Penal Code", "Code of Criminal Procedure", "Code of Civil Procedure", "Constitution of India"]
    sections = ["Section 302", "Section 420", "Section 376", "Section 498A", "Section 307"]
    dates = ["12th January 2020", "15 August 2021", "26 January 2022", "5th May 2023"]
    citations = ["2021 SEC 456", "2020 SCC 789", "AIR 2019 SC 123", "2023 INSC 987"]
    
    data = []
    for _ in range(200):
        p = random.choice(petitioners)
        r = random.choice(respondents)
        j = random.choice(judges)
        c = random.choice(courts)
        s = random.choice(statutes)
        sec = random.choice(sections)
        d = random.choice(dates)
        cit = random.choice(citations)
        
        templates = [
            (f"Before the {c}, the petitioner {p} argued against {r}.", 
             [("COURT", c), ("PETITIONER", p), ("RESPONDENT", r)]),
            (f"Judgment was delivered by {j} in the case of {p} v. {r}.", 
             [("JUDGE", j), ("PETITIONER", p), ("RESPONDENT", r)]),
            (f"The accused was charged under {sec} of the {s}.", 
             [("SECTION", sec), ("STATUTE", s)]),
            (f"This petition was filed in {c} on {d}.", 
             [("COURT", c), ("DATE", d)]),
            (f"In citation {cit}, the {c} allowed the appeal.", 
             [("CITATION", cit), ("COURT", c)]),
            (f"Justice {j} observed that {r} failed to comply with the order.", 
             [("JUDGE", j), ("RESPONDENT", r)]),
            (f"Under the provisions of {sec}, the petitioner {p} sought bail.", 
             [("SECTION", sec), ("PETITIONER", p)])
        ]
        
        text, entities_meta = random.choice(templates)
        entities = []
        for label, val in entities_meta:
            start = text.find(val)
            if start != -1:
                end = start + len(val)
                entities.append((start, end, label))
        data.append((text, {"entities": entities}))
    return data

def train_ner(output_dir: str, n_iter: int = 5):
    """Resume training of spaCy model on mapped Indian Legal NER data."""

    print("Loading base model 'en_core_web_sm'...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Base model 'en_core_web_sm' not found. Downloading...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")


    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")


    for label in CUSTOM_LABELS:
        ner.add_label(label)


    try:
        train_data = load_ner_data_from_hf("train")
        try:
            val_data = load_ner_data_from_hf("validation")
        except Exception:
            val_data = load_ner_data_from_hf("test")
    except Exception as e:
        print(f"Failed to load InLegalNER from Hugging Face: {e}")
        print("Falling back to generating local mock dataset for training...")
        mock_data = get_mock_ner_data()
        random.shuffle(mock_data)
        split_idx = int(len(mock_data) * 0.8)
        train_data = mock_data[:split_idx]
        val_data = mock_data[split_idx:]

    random.shuffle(train_data)
    train_data = train_data[:500]
    random.shuffle(val_data)
    val_data = val_data[:100]
    

    optimizer = nlp.resume_training()
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    print(f"Starting training on {len(train_data)} samples for {n_iter} epochs...")
    with nlp.disable_pipes(*other_pipes):
        for epoch in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], sgd=optimizer, losses=losses, drop=0.3)
            print(f"Epoch {epoch+1}/{n_iter} — NER loss: {losses.get('ner', 0):.4f}")

    # Evaluate the model
    print("Evaluating model...")
    from spacy.scorer import Scorer
    scorer = Scorer()
    examples = []
    for text, annotations in val_data:
        doc = nlp(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    scores = scorer.score(examples)

    metrics = {
        "precision": scores.get("ents_p", 0.0) or 0.0,
        "recall": scores.get("ents_r", 0.0) or 0.0,
        "f1_score": scores.get("ents_f", 0.0) or 0.0,
        "per_label_metrics": {}
    }

    per_type = scores.get("ents_per_type", {})
    for label, label_scores in per_type.items():
        metrics["per_label_metrics"][label] = {
            "precision": label_scores.get("p", 0.0) or 0.0,
            "recall": label_scores.get("r", 0.0) or 0.0,
            "f1_score": label_scores.get("f", 0.0) or 0.0
        }

    print("\n--- NER Evaluation Metrics ---")
    print(f"Overall Precision: {metrics['precision']:.4f}")
    print(f"Overall Recall:    {metrics['recall']:.4f}")
    print(f"Overall F1-Score:  {metrics['f1_score']:.4f}")
    print("\nPer-Label Metrics:")
    for label, m in metrics["per_label_metrics"].items():
        print(f"  {label:<12} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1_score']:.4f}")
    print("------------------------------\n")

    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Model successfully saved to {output_dir}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")


def clean_party_name(name: str) -> str:
    import re
    # Remove common suffixes and wrappers
    name = re.sub(r'\s*\.\.\.\s*(?:Appellant|Respondent|Petitioner|Accused|Complainant)s?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\b(?:Appellant|Respondent|Petitioner|Accused|Complainant)s?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*(?:and|&)\s+(?:Anr|Ors|another|others)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\b(?:Anr|Ors|another|others)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(s\)', '', name, flags=re.IGNORECASE)
    # Remove any trailing / leading punctuation or leftover &
    name = name.strip().strip(",").strip(".").strip("&").strip()
    return name

def is_deceased_or_victim(name: str, full_text: str) -> bool:
    import re
    name_escaped = re.escape(name)
    for match in re.finditer(name_escaped, full_text, re.IGNORECASE):
        start_idx = match.start()
        context_before = full_text[max(0, start_idx - 40):start_idx].lower()
        if any(w in context_before for w in ["deceased", "victim", "dead", "murdered"]):
            return True
    return False

def remove_generic_duplicates(values: list) -> list:
    sorted_vals = sorted(list(set(values)), key=len, reverse=True)
    keep = []
    for val in sorted_vals:
        if not any(val.lower() in existing.lower() and val.lower() != existing.lower() for existing in keep):
            keep.append(val)
    result = []
    for val in values:
        if val in keep and val not in result:
            result.append(val)
    return result

def extract_entities(text: str, nlp=None) -> dict:
    """Extract entities using the trained legal NER model."""
    import re
    if nlp is None:
        model_path = Path(__file__).resolve().parent.parent.parent / "models" / "legal_ner"
        try:
            nlp = spacy.load(model_path)
        except OSError:
            print(f"Trained model not found at {model_path}. Loading fallback 'en_core_web_sm'...")
            nlp = spacy.load("en_core_web_sm")
            
    doc = nlp(text)
    raw_entities = {}
    for ent in doc.ents:
        raw_entities.setdefault(ent.label_, []).append(ent.text)
        
    # Standard labels expected
    expected_labels = ["PETITIONER", "RESPONDENT", "JUDGE", "COURT", "STATUTE", "SECTION", "CITATION", "DATE"]
    cleaned = {label: [] for label in expected_labels}
    
    # Heuristic parsing of headers for high accuracy in party & court extraction
    header_petitioner = None
    header_respondent = None
    header_court = None
    
    lines = [l.strip() for l in text.split("\n")[:20] if l.strip()]
    for i, line in enumerate(lines):
        # Only parse actual header lines, not body paragraphs
        if len(line) < 120:
            # Court detection in header
            if "IN THE" in line.upper() and "COURT" in line.upper():
                c_name = line.upper().replace("IN THE", "").strip()
                header_court = c_name.title()
            
            # Appellant / Petitioner detection
            if "APPELLANT" in line.upper() or "PETITIONER" in line.upper():
                part = re.split(r'\.\.\.|versus|vs\.', line, flags=re.IGNORECASE)[0].strip()
                if part:
                    header_petitioner = part
            
            # Respondent detection
            if "RESPONDENT" in line.upper():
                part = re.split(r'\.\.\.|versus|vs\.', line, flags=re.IGNORECASE)[0].strip()
                if part:
                    header_respondent = part
                
    if not header_petitioner or not header_respondent:
        for i, line in enumerate(lines):
            if len(line) < 120 and line.lower() in ["versus", "vs", "vs."]:
                if i > 0 and not header_petitioner:
                    header_petitioner = re.split(r'\.\.\.', lines[i-1])[0].strip()
                if i < len(lines) - 1 and not header_respondent:
                    header_respondent = re.split(r'\.\.\.', lines[i+1])[0].strip()

    # Clean the header party names for comparison
    header_pet_clean = clean_party_name(header_petitioner) if header_petitioner else ""
    header_res_clean = clean_party_name(header_respondent) if header_respondent else ""

    # Find cited case parties in the body text
    cited_parties = set()
    citation_matches = re.findall(
        r'\b([A-Z][A-Za-z0-9\s\.\&\u0080-\uFFFF\-]{2,50})\s+(?:v\.|versus|vs\.|vs)\s+([A-Z][A-Za-z0-9\s\.\&\u0080-\uFFFF\-]{2,50})\b',
        text
    )
    for p1, p2 in citation_matches:
        p1_c = clean_party_name(p1)
        p2_c = clean_party_name(p2)
        if len(p1_c) > 3:
            cited_parties.add(p1_c.lower())
        if len(p2_c) > 3:
            cited_parties.add(p2_c.lower())

    # Clean and filter each label
    for label, values in raw_entities.items():
        if label not in cleaned:
            continue
        for val in values:
            val_clean = val.strip().strip(",").strip(".").strip()
            if not val_clean:
                continue
                
            # Heuristics for Bug 3 (filtering wrong entities)
            
            # --- JUDGE FILTERS ---
            if label == "JUDGE":
                if re.search(r'\d', val_clean):
                    continue
                if any(x in val_clean.lower() for x in ["pm", "am", "o'clock", "night", "morning", "at approximately"]):
                    continue
                if any(x in val_clean.lower() for x in ["court", "appeal", "judgement", "judgment", "prosecution", "witness", "police", "appellant"]):
                    continue
                if len(val_clean.split()) < 2 and not val_clean.startswith("Justice"):
                    continue
                    
            # --- STATUTE FILTERS ---
            elif label == "STATUTE":
                if val_clean.upper() in ["FACTS OF THE", "FACTS OF THE CASE", "WITNESSES AND EVIDENCE", "HELD", "JUDGMENT", "ISSUE BEFORE", "RESULT"]:
                    continue
                if len(val_clean) < 4:
                    continue
                if not any(x in val_clean.lower() for x in ["code", "act", "constitution", "ipc", "crpc", "cpc", "evidence", "penal", "regulation", "statute"]):
                    continue
                    
            # --- CITATION FILTERS ---
            elif label == "CITATION":
                if "witness" in val_clean.lower():
                    continue
                if not re.search(r'\d', val_clean):
                    continue
                if not any(x in val_clean.upper() for x in ["SCR", "SCC", "AIR", "SEC", "SCALE", "JT", "INSC", "CRIMINAL APPEAL"]):
                    continue
                    
            # --- DATE FILTERS ---
            elif label == "DATE":
                if "of" in val_clean.lower() and re.match(r'^\d+\s+of\s+\d+$', val_clean.lower().strip()):
                    continue
                months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
                if not any(m in val_clean.lower() for m in months) and not re.search(r'\d{4}', val_clean):
                    continue
                    
            # --- COURT FILTERS ---
            elif label == "COURT":
                if "court" not in val_clean.lower():
                    continue
                if any(x in val_clean.lower() for x in ["versus", "appellant", "respondent"]):
                    continue
                    
            # --- SECTION FILTERS ---
            elif label == "SECTION":
                if not any(x in val_clean.lower() for x in ["section", "sec", "u/s"]):
                    continue
                    
            # --- PETITIONER / RESPONDENT FILTERS ---
            elif label in ["PETITIONER", "RESPONDENT"]:
                val_clean = clean_party_name(val_clean)
                if not val_clean or len(val_clean) < 3:
                    continue
                if re.search(r'\d', val_clean):
                    continue
                # Filter out obvious verbs or stop words (like "State has")
                if any(f" {w} " in f" {val_clean.lower()} " for w in ["has", "is", "was", "appealed", "reversing", "finding", "conviction", "sentence", "trial", "prosecution", "defense", "court", "judge", "magistrate", "witness", "police"]):
                    continue
                # Check if it is the deceased or victim in the text
                if is_deceased_or_victim(val_clean, text):
                    continue
                # Check if it is a cited party and does not match the header parties
                is_main_party = False
                if header_pet_clean and val_clean.lower() in header_pet_clean.lower():
                    is_main_party = True
                if header_res_clean and val_clean.lower() in header_res_clean.lower():
                    is_main_party = True
                
                if val_clean.lower() in cited_parties and not is_main_party:
                    continue
                    
            cleaned[label].append(val_clean)
            
    # Remove generic duplicates (e.g. "High Court" if "Bombay High Court" exists)
    for lbl in cleaned:
        cleaned[lbl] = remove_generic_duplicates(cleaned[lbl])

    # Inject header court
    if header_court:
        for lbl in cleaned:
            cleaned[lbl] = [existing for existing in cleaned[lbl] if existing.lower() != header_court.lower()]
        cleaned["COURT"].insert(0, header_court)
        cleaned["COURT"] = remove_generic_duplicates(cleaned["COURT"])
            
    # Inject header parties
    if header_petitioner:
        header_pet_clean = clean_party_name(header_petitioner)
        for lbl in ["RESPONDENT", "COURT"]:
            cleaned[lbl] = [existing for existing in cleaned[lbl] if existing.lower() != header_pet_clean.lower()]
        if not any(header_pet_clean.lower() == existing.lower() for existing in cleaned["PETITIONER"]):
            cleaned["PETITIONER"].insert(0, header_pet_clean)
        cleaned["PETITIONER"] = remove_generic_duplicates(cleaned["PETITIONER"])
            
    if header_respondent:
        header_res_clean = clean_party_name(header_respondent)
        for lbl in ["PETITIONER", "COURT"]:
            cleaned[lbl] = [existing for existing in cleaned[lbl] if existing.lower() != header_res_clean.lower()]
        if not any(header_res_clean.lower() == existing.lower() for existing in cleaned["RESPONDENT"]):
            cleaned["RESPONDENT"].insert(0, header_res_clean)
        cleaned["RESPONDENT"] = remove_generic_duplicates(cleaned["RESPONDENT"])
            
    # Move other names from PETITIONER to RESPONDENT in state-prosecuted cases
    has_state_petitioner = any(any(s in p.lower() for s in ["state of", "union of", "republic of"]) for p in cleaned["PETITIONER"])
    if has_state_petitioner:
        for p in list(cleaned["PETITIONER"]):
            if not any(s in p.lower() for s in ["state of", "union of", "republic of"]):
                cleaned["PETITIONER"].remove(p)
                if not any(p.lower() == existing.lower() for existing in cleaned["RESPONDENT"]):
                    cleaned["RESPONDENT"].append(p)
                    
    # Clean up cross-contamination
    for p in list(cleaned["PETITIONER"]):
        cleaned["RESPONDENT"] = [existing for existing in cleaned["RESPONDENT"] if existing.lower() != p.lower()]

    # Run remove_generic_duplicates one final time
    for lbl in cleaned:
        cleaned[lbl] = remove_generic_duplicates(cleaned[lbl])

    cleaned.setdefault("VERDICT", [])
    cleaned.setdefault("CHARGE", [])
    return cleaned

if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "legal_ner")
    train_ner(output_dir=model_dir, n_iter=5)
