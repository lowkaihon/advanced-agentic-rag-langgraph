"""
Document profiling module for analyzing document characteristics.

This module profiles individual documents to extract features that help inform
intelligent retrieval strategy selection.
"""

import re
from typing import Dict, List, Literal
from collections import Counter


class DocumentProfiler:
    """
    Profiles individual documents to extract characteristics for intelligent retrieval.

    Analyzes:
    - Document length and structure
    - Technical density (code, jargon, special terminology)
    - Content type (tutorial, API reference, conceptual, guide)
    - Reading level
    - Domain tags
    - Optimal retrieval strategy
    """

    # Technical indicators
    TECHNICAL_PATTERNS = {
        'code_block': r'```[\s\S]*?```|`[^`]+`',
        'function_call': r'\w+\([^)]*\)',
        'class_def': r'class\s+\w+',
        'import_stmt': r'import\s+\w+|from\s+\w+',
        'variable_assign': r'\w+\s*=\s*[^=]',
        'brackets': r'\{|\}|\[|\]',
        'special_chars': r'[<>@#$%^&*]',
    }

    # Domain keywords for categorization
    DOMAIN_KEYWORDS = {
        'machine_learning': ['model', 'training', 'neural', 'gradient', 'loss', 'accuracy', 'dataset'],
        'deep_learning': ['neural network', 'deep learning', 'cnn', 'rnn', 'transformer', 'backprop'],
        'nlp': ['language', 'text', 'tokenize', 'embedding', 'semantic', 'nlp'],
        'rag': ['retrieval', 'rag', 'vector', 'embedding', 'search', 'rerank'],
        'langgraph': ['graph', 'node', 'edge', 'state', 'workflow', 'langgraph'],
        'api': ['api', 'endpoint', 'request', 'response', 'parameter', 'return'],
        'framework': ['framework', 'library', 'package', 'module', 'install'],
    }

    # Question words indicating tutorial/guide content
    TUTORIAL_INDICATORS = ['how to', 'tutorial', 'guide', 'introduction', 'getting started', 'example', 'step']
    API_INDICATORS = ['reference', 'documentation', 'api', 'function', 'class', 'method', 'parameter']

    def profile_document(self, doc_text: str, doc_id: str = None) -> tuple[Dict, Dict]:
        """
        Profile a single document and return characteristics + metadata.

        Args:
            doc_text: The document content to analyze
            doc_id: Optional document identifier

        Returns:
            Tuple of (profile_dict, metadata_dict)
        """
        # Basic metrics
        words = doc_text.split()
        word_count = len(words)

        # Structural analysis
        has_code = self._detect_code_blocks(doc_text)
        has_lists = self._detect_lists(doc_text)
        has_headings = self._detect_headings(doc_text)
        has_math = self._detect_math_notation(doc_text)

        # Technical density
        technical_density = self._calculate_technical_density(doc_text)

        # Document type classification
        doc_type = self._classify_document_type(doc_text)

        # Reading level (simplified)
        reading_level = self._estimate_reading_level(doc_text, technical_density)

        # Domain tags
        domain_tags = self._extract_domain_tags(doc_text)

        # Structure quality
        structure = self._analyze_structure(has_headings, has_lists, has_code)

        # Build profile
        profile = {
            "doc_id": doc_id,
            "length": word_count,
            "type": doc_type,
            "technical_density": round(technical_density, 2),
            "has_code": has_code,
            "has_math": has_math,
            "has_lists": has_lists,
            "has_headings": has_headings,
            "structure": structure,
            "reading_level": reading_level,
            "domain_tags": domain_tags,
        }

        # Generate metadata for retrieval
        metadata = self._generate_metadata(profile)

        return profile, metadata

    def _detect_code_blocks(self, text: str) -> bool:
        """Detect if document contains code blocks or inline code."""
        pattern = self.TECHNICAL_PATTERNS['code_block']
        matches = re.findall(pattern, text)
        return len(matches) > 0

    def _detect_lists(self, text: str) -> bool:
        """Detect if document contains bulleted or numbered lists."""
        # Look for common list patterns
        list_patterns = [
            r'^\s*[-*•]\s',  # Bullet points
            r'^\s*\d+\.\s',  # Numbered lists
        ]
        for pattern in list_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _detect_headings(self, text: str) -> bool:
        """Detect if document contains headings (markdown-style or caps)."""
        # Look for markdown headings or ALL CAPS lines
        heading_patterns = [
            r'^#{1,6}\s+\w+',  # Markdown headings
            r'^[A-Z\s]{10,}$',  # ALL CAPS lines (likely headings)
        ]
        for pattern in heading_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _detect_math_notation(self, text: str) -> bool:
        """Detect mathematical notation."""
        math_indicators = [
            r'\$.*?\$',  # LaTeX math
            r'\\[a-z]+\{',  # LaTeX commands
            r'[∑∫∂∇αβγδε]',  # Math symbols
            r'\b(equation|formula|theorem|proof)\b',  # Math keywords
        ]
        for pattern in math_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _calculate_technical_density(self, text: str) -> float:
        """
        Calculate technical density (0.0 to 1.0).
        Higher values indicate more technical content.
        """
        total_indicators = 0
        found_indicators = 0

        for name, pattern in self.TECHNICAL_PATTERNS.items():
            total_indicators += 1
            if re.search(pattern, text):
                found_indicators += 1
                # Give extra weight to code blocks
                if name == 'code_block':
                    found_indicators += 1
                    total_indicators += 1

        # Also check for technical jargon (capitalized acronyms)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        if len(acronyms) > 3:
            found_indicators += 1
        total_indicators += 1

        return found_indicators / total_indicators if total_indicators > 0 else 0.0

    def _classify_document_type(self, text: str) -> Literal["tutorial", "api_reference", "guide", "conceptual"]:
        """
        Classify document type based on content patterns.
        """
        text_lower = text.lower()

        # Score each type
        scores = {
            "tutorial": 0,
            "api_reference": 0,
            "guide": 0,
            "conceptual": 0,
        }

        # Tutorial indicators
        for indicator in self.TUTORIAL_INDICATORS:
            if indicator in text_lower:
                scores["tutorial"] += 1

        # API reference indicators
        for indicator in self.API_INDICATORS:
            if indicator in text_lower:
                scores["api_reference"] += 1

        # Code presence suggests tutorial or API
        if self._detect_code_blocks(text):
            scores["tutorial"] += 1
            scores["api_reference"] += 1

        # Headings + lists suggest guide
        if self._detect_headings(text) and self._detect_lists(text):
            scores["guide"] += 1

        # Questions suggest tutorial/guide
        if re.search(r'\?', text):
            scores["tutorial"] += 1
            scores["guide"] += 1

        # Short, definition-style suggests conceptual
        word_count = len(text.split())
        if word_count < 100 and not self._detect_code_blocks(text):
            scores["conceptual"] += 2

        # Return highest scoring type
        max_score = max(scores.values())
        if max_score == 0:
            return "conceptual"  # Default

        return max(scores, key=scores.get)

    def _estimate_reading_level(self, text: str, technical_density: float) -> Literal["beginner", "intermediate", "advanced"]:
        """
        Estimate reading level based on complexity indicators.
        """
        # Start with technical density
        if technical_density > 0.7:
            return "advanced"
        elif technical_density > 0.4:
            return "intermediate"

        # Check for complex indicators
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        # Long words suggest higher level
        if avg_word_length > 6:
            return "intermediate" if technical_density > 0.3 else "advanced"

        return "beginner"

    def _extract_domain_tags(self, text: str) -> List[str]:
        """
        Extract domain tags based on keyword matching.
        """
        text_lower = text.lower()
        tags = []

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 1:  # At least 1 keyword match
                tags.append(domain)

        return tags if tags else ["general"]

    def _analyze_structure(self, has_headings: bool, has_lists: bool, has_code: bool) -> Literal["well-structured", "moderately-structured", "unstructured"]:
        """
        Analyze overall document structure.
        """
        structure_score = sum([has_headings, has_lists, has_code])

        if structure_score >= 2:
            return "well-structured"
        elif structure_score == 1:
            return "moderately-structured"
        else:
            return "unstructured"

    def _generate_metadata(self, profile: Dict) -> Dict:
        """
        Generate metadata for retrieval based on profile.
        """
        # Suggest best retrieval strategy
        best_strategy = self._suggest_retrieval_strategy(profile)

        metadata = {
            "content_type": profile["type"],
            "technical_level": profile["reading_level"],
            "has_code": profile["has_code"],
            "has_examples": profile["has_code"] or profile["has_lists"],  # Heuristic
            "domain": profile["domain_tags"][0] if profile["domain_tags"] else "general",
            "structure_quality": profile["structure"],
            "best_retrieval_strategy": best_strategy,
            "word_count": profile["length"],
        }

        return metadata

    def _suggest_retrieval_strategy(self, profile: Dict) -> Literal["semantic", "keyword", "hybrid"]:
        """
        Suggest optimal retrieval strategy for this document type.
        """
        doc_type = profile["type"]
        technical_density = profile["technical_density"]
        has_code = profile["has_code"]

        # API references: keyword-heavy (users search for specific function names)
        if doc_type == "api_reference" or (has_code and technical_density > 0.7):
            return "keyword"

        # Conceptual content: semantic search (meaning-based)
        if doc_type == "conceptual" and technical_density < 0.3:
            return "semantic"

        # Everything else: hybrid (best of both worlds)
        return "hybrid"

    def profile_corpus(self, documents: List[str]) -> Dict:
        """
        Profile entire corpus to extract aggregate statistics.

        Args:
            documents: List of document texts

        Returns:
            Corpus-level statistics
        """
        profiles = [self.profile_document(doc, f"doc_{i}")[0] for i, doc in enumerate(documents)]

        if not profiles:
            return {}

        # Aggregate statistics
        total_docs = len(profiles)
        avg_length = sum(p["length"] for p in profiles) / total_docs
        avg_technical_density = sum(p["technical_density"] for p in profiles) / total_docs

        # Count document types
        type_counts = Counter(p["type"] for p in profiles)

        # Count structure quality
        structure_counts = Counter(p["structure"] for p in profiles)

        # Collect all domain tags
        all_tags = []
        for p in profiles:
            all_tags.extend(p["domain_tags"])
        domain_distribution = Counter(all_tags)

        # Percentage with code/math
        pct_with_code = sum(1 for p in profiles if p["has_code"]) / total_docs * 100
        pct_with_math = sum(1 for p in profiles if p["has_math"]) / total_docs * 100

        corpus_stats = {
            "total_documents": total_docs,
            "avg_length": round(avg_length, 1),
            "avg_technical_density": round(avg_technical_density, 2),
            "document_types": dict(type_counts),
            "structure_quality": dict(structure_counts),
            "domain_distribution": dict(domain_distribution.most_common(5)),
            "pct_with_code": round(pct_with_code, 1),
            "pct_with_math": round(pct_with_math, 1),
        }

        return corpus_stats
