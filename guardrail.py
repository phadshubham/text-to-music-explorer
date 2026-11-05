"""
Unit 5: Named-Artist Guardrail & Intellectual Property Defense
Multi-layered defense:
1. Regex style-reference syntactic patterns.
2. Curated database (data/artists.txt, 300+ global, Indian, and classical artists).
3. RapidFuzz token matching with typo tolerance (threshold >= 85%).
4. Context-aware ambiguous common-word disambiguation (Queen, Prince, Seal, Pink, Future, etc.).
5. Automatic descriptive rewrite generator (genre + era + instrumentation).
"""

import os
import re
import json
from typing import Dict, Any, Optional, Tuple, List
from rapidfuzz import fuzz, process


# Ambiguous common words that are artist names but also valid English words or nouns
AMBIGUOUS_ARTISTS = {
    "queen", "prince", "seal", "pink", "future", "police", "cream", "kiss",
    "oasis", "blur", "air", "rush", "chic", "yes", "genesis", "low", "heart",
    "europe", "survivor", "wham", "america", "asia", "chicago", "boston", "king", "royal"
}

# Regex style-reference contextual triggers
STYLE_REFERENCE_PATTERNS = [
    r"\b(?:in\s+the\s+style\s+of|in\s+style\s+of)\b",
    r"\b(?:sounds?\s+like|sounding\s+like)\b",
    r"\b(?:similar\s+to|in\s+the\s+vein\s+of)\b",
    r"\b(?:inspired\s+by|influence(?:d\s+by)?)\b",
    r"\b(?:cover\s+of|remix\s+of)\b",
    r"\b(?:like\s+a\s+|like\s+)(?=[A-Za-z]+)",
    r"\b(?:in\s+[a-zA-Z'\s]+(?:'s|\s+own)\s+voice)\b",
    r"\b(?:[a-zA-Z]+-esque|[a-zA-Z]+-ish)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in STYLE_REFERENCE_PATTERNS]


class ArtistGuardrail:
    """Multi-tiered guardrail preventing imitation of named musical artists and composers."""

    def __init__(self, artists_file: Optional[str] = None):
        self.artists_file = artists_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "artists.txt"
        )
        self.artists: List[str] = []
        self.artists_lower_set = set()
        self._load_artists()

    def _load_artists(self):
        """Load and normalize curated artist database, indexing full names and prominent surnames."""
        if not os.path.exists(self.artists_file):
            print(f"[Guardrail Warning] Artists file not found at {self.artists_file}. Initializing empty.")
            return

        with open(self.artists_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.artists.append(line)
                    self.artists_lower_set.add(line.lower())

                    # Also index distinct surnames for classical/film composers and multi-word artists
                    parts = line.split()
                    if len(parts) >= 2:
                        surname = parts[-1]
                        if len(surname) >= 5 and surname.lower() not in AMBIGUOUS_ARTISTS:
                            common_words = {"music", "sound", "band", "group", "orchestra", "trio", "quartet", "junior", "senior"}
                            if surname.lower() not in common_words and surname not in self.artists:
                                self.artists.append(surname)
                                self.artists_lower_set.add(surname.lower())

        print(f"[Guardrail] Loaded {len(self.artists)} artists from {self.artists_file}")

    def has_style_reference_context(self, text: str) -> bool:
        """Determine if text contains stylistic imitation or attribution markers."""
        for pattern in COMPILED_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def check(self, text: str) -> Dict[str, Any]:
        """
        Evaluate text against artist guardrail.

        Returns:
            Dict containing:
            - is_safe: bool
            - detected_artist: Optional[str]
            - reason: Optional[str]
            - suggestion: Optional[str]
        """
        clean_text = text.strip()
        text_lower = clean_text.lower()
        has_context = self.has_style_reference_context(clean_text)

        # 1. Direct and suffix pattern checks (e.g. "Taylor Swift-ish", "Beethoven-esque")
        suffix_match = re.search(r"\b([A-Za-z]+)-(?:ish|esque)\b", clean_text, re.IGNORECASE)
        if suffix_match:
            potential_name = suffix_match.group(1).lower()
            for artist in self.artists:
                artist_lower = artist.lower()
                if potential_name in artist_lower or fuzz.ratio(potential_name, artist_lower) >= 80:
                    return self._block(
                        artist=artist,
                        reason=f"Request contains stylistic imitation suffix '-ish/-esque' targeting '{artist}'."
                    )

        # 2. Check for exact or high-confidence artist occurrences
        # Handle ambiguous common words specially
        for artist in self.artists:
            artist_lower = artist.lower()
            is_ambiguous = artist_lower in AMBIGUOUS_ARTISTS

            # Regex word boundary check
            pattern = r"\b" + re.escape(artist_lower) + r"\b"
            if re.search(pattern, text_lower):
                if is_ambiguous:
                    # Ambiguous words (e.g. Queen, Pink, Prince) are ONLY blocked in style-reference context
                    if has_context:
                        return self._block(
                            artist=artist,
                            reason=f"Stylistic reference to ambiguous artist name '{artist}' detected."
                        )
                else:
                    # Non-ambiguous named artists are blocked
                    return self._block(
                        artist=artist,
                        reason=f"Reference to named artist/composer '{artist}' detected."
                    )

        # 3. Fuzzy matching for typos in style-reference context or multi-word artists
        # Only run fuzzy matching if prompt looks like a style reference or contains multi-word capitalized names
        if has_context or any(c.isupper() for c in clean_text):
            # Extract n-grams of 1 to 3 words
            words = re.findall(r"\b\w+\b", clean_text)
            candidates = []
            for n in [1, 2, 3]:
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i : i + n])
                    candidates.append(ngram)

            for cand in candidates:
                cand_lower = cand.lower()
                if cand_lower in AMBIGUOUS_ARTISTS and not has_context:
                    continue

                best_match = process.extractOne(
                    cand,
                    self.artists,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=88,
                )
                if best_match:
                    matched_name, score, _ = best_match
                    matched_lower = matched_name.lower()
                    if matched_lower in AMBIGUOUS_ARTISTS and not has_context:
                        continue
                    # Ignore single short word matches that are low length
                    if len(cand) < 4:
                        continue
                    return self._block(
                        artist=matched_name,
                        reason=f"Fuzzy match detected named artist/composer '{matched_name}' (typo match confidence: {score:.1f}%)."
                    )

        return {
            "is_safe": True,
            "detected_artist": None,
            "reason": None,
            "suggestion": None,
        }

    def _block(self, artist: str, reason: str) -> Dict[str, Any]:
        """Construct descriptive block payload with constructive rewrite suggestion."""
        suggestion = self._generate_descriptive_rewrite(artist)
        return {
            "is_safe": False,
            "detected_artist": artist,
            "reason": reason,
            "suggestion": suggestion,
        }

    def _generate_descriptive_rewrite(self, artist: str) -> str:
        """Generate a compliant descriptive rewrite suggestion without named artists."""
        artist_lower = artist.lower()
        if "beethoven" in artist_lower or "mozart" in artist_lower or "bach" in artist_lower:
            return "Try describing the era and instrumentation instead: 'Classical symphonic orchestra with soaring strings, expressive woodwinds, and dynamic timpani crescendo, 120 BPM'."
        elif "rahman" in artist_lower or "singh" in artist_lower:
            return "Try describing musical elements: 'Indian cinematic fusion with evocative bansuri flute, emotive acoustic strings, subtle tabla groove, 85 BPM'."
        elif "swift" in artist_lower or "sheeran" in artist_lower:
            return "Try describing production and mood: 'Acoustic pop ballad with bright acoustic guitar strumming, heartfelt warm piano chords, crisp 4/4 drums, 90 BPM'."
        elif "daft punk" in artist_lower or "electronic" in artist_lower:
            return "Try describing genre and texture: 'French electro-house with funky slap bass, vocoder-filtered analog synths, driving four-on-the-floor kick, 124 BPM'."
        return "Try describing genre, era, instrumentation, and mood directly (e.g. '80s synth-pop with driving analog bass, bright synthesizer arpeggios, punchy drum machine, 120 BPM')."


# Optional BERT-family NER model (bert-base-NER) for Named Entity Recognition
_BERT_NER_PIPELINE = None


def get_bert_ner_pipeline():
    """Lazy loader for BERT-family Named Entity Recognition model (bert-base-NER)."""
    global _BERT_NER_PIPELINE
    if _BERT_NER_PIPELINE is None:
        try:
            from transformers import pipeline
            _BERT_NER_PIPELINE = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=-1,
            )
            print("[Guardrail] Initialized BERT-family Named Entity Recognizer (bert-base-NER)")
        except Exception as e:
            _BERT_NER_PIPELINE = False
    return _BERT_NER_PIPELINE if _BERT_NER_PIPELINE is not False else None


def extract_person_entities_bert(text: str) -> List[str]:
    """Extract PERSON entities using BERT-family token classification (bert-base-NER)."""
    ner = get_bert_ner_pipeline()
    if not ner:
        return []
    try:
        entities = ner(text)
        persons = [
            e["word"].strip() for e in entities
            if e.get("entity_group") == "PER" and len(e.get("word", "").strip()) >= 3
        ]
        return persons
    except Exception:
        return []


# Singleton instance
_GUARDRAIL_INSTANCE = None


def get_guardrail() -> ArtistGuardrail:
    """Return cached singleton instance of ArtistGuardrail."""
    global _GUARDRAIL_INSTANCE
    if _GUARDRAIL_INSTANCE is None:
        _GUARDRAIL_INSTANCE = ArtistGuardrail()
    return _GUARDRAIL_INSTANCE


def check_guardrail(prompt: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Convenience wrapper for singleton ArtistGuardrail.check()."""
    res = get_guardrail().check(prompt)
    return res["is_safe"], res.get("reason"), res.get("suggestion")

