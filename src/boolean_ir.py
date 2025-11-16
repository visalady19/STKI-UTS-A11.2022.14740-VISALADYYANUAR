import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class BooleanRetrieval:
    def __init__(self, docs_data: List[Dict]):
        """
        Initialize Boolean Retrieval untuk medical search
        
        Args:
            docs_data: List of document dictionaries dari preprocess_corpus
        """
        self.docs_data = {doc['doc_id']: doc for doc in docs_data}
        self.docs_tokens = {doc['doc_id']: doc['tokens'] for doc in docs_data}
        self.doc_ids = sorted(self.docs_tokens.keys())
        self.vocabulary = self._build_vocabulary()
        self.inverted_index = self._build_inverted_index()
        self.incidence_matrix = self._build_incidence_matrix()
    
    def _build_vocabulary(self) -> List[str]:
        """Membangun vocabulary dari semua token unik"""
        vocab = set()
        for tokens in self.docs_tokens.values():
            vocab.update(tokens)
        return sorted(list(vocab))
    
    def _build_inverted_index(self) -> Dict[str, Set[str]]:
        """
        Membangun inverted index: term -> set of doc_ids
        """
        index = defaultdict(set)
        for doc_id, tokens in self.docs_tokens.items():
            unique_tokens = set(tokens)
            for token in unique_tokens:
                index[token].add(doc_id)
        return dict(index)
    
    def _build_incidence_matrix(self) -> np.ndarray:
        """
        Membangun incidence matrix (term-document matrix)
        Binary matrix [vocab_size x num_docs]
        """
        vocab_size = len(self.vocabulary)
        num_docs = len(self.doc_ids)
        
        matrix = np.zeros((vocab_size, num_docs), dtype=int)
        
        term_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}
        doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        
        for term, postings in self.inverted_index.items():
            term_idx = term_to_idx[term]
            for doc_id in postings:
                doc_idx = doc_to_idx[doc_id]
                matrix[term_idx][doc_idx] = 1
        
        return matrix
    
    def show_incidence_matrix(self, max_terms: int = 15, max_docs: int = 10):
        """Menampilkan incidence matrix"""
        print(f"\n{'='*100}")
        print("INCIDENCE MATRIX (Term-Document)")
        print(f"{'='*100}")
        
        terms_to_show = self.vocabulary[:max_terms]
        docs_to_show = self.doc_ids[:max_docs]
        
        # Header
        print(f"{'Term':<20}", end="")
        for doc_id in docs_to_show:
            print(f"{doc_id[:10]:<12}", end="")
        print()
        print("-" * 100)
        
        # Rows
        term_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}
        doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        
        for term in terms_to_show:
            term_idx = term_to_idx[term]
            print(f"{term:<20}", end="")
            for doc_id in docs_to_show:
                doc_idx = doc_to_idx[doc_id]
                val = self.incidence_matrix[term_idx][doc_idx]
                print(f"{val:<12}", end="")
            print()
        
        print(f"{'='*100}")
        print(f"Full size: {len(self.vocabulary)} terms × {len(self.doc_ids)} docs")
        sparsity = (1 - np.count_nonzero(self.incidence_matrix) / self.incidence_matrix.size) * 100
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"{'='*100}\n")
    
    def parse_query(self, query: str) -> List[str]:
        """Parse Boolean query ke postfix notation"""
        query = query.lower()
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        tokens = query.split()
        
        output = []
        ops = []
        precedence = {"not": 3, "and": 2, "or": 1}
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            if token in precedence:
                while (ops and ops[-1] != "(" and 
                       ops[-1] in precedence and 
                       precedence[ops[-1]] >= precedence[token]):
                    output.append(ops.pop())
                ops.append(token)
            elif token == "(":
                ops.append(token)
            elif token == ")":
                while ops and ops[-1] != "(":
                    output.append(ops.pop())
                if ops:
                    ops.pop()
            else:
                output.append(token)
        
        while ops:
            output.append(ops.pop())
        
        return output
    
    def eval_postfix(self, postfix: List[str], explain: bool = False) -> Tuple[Set[str], List[str]]:
        """Evaluasi postfix expression"""
        stack = []
        explanations = []
        all_docs = set(self.doc_ids)
        
        for token in postfix:
            if token not in ["and", "or", "not"]:
                result = self.inverted_index.get(token, set())
                stack.append(result)
                if explain:
                    explanations.append(f"'{token}' → {len(result)} diseases: {sorted(list(result)[:5])}")
            else:
                if token == "not":
                    if not stack:
                        continue
                    A = stack.pop()
                    result = all_docs - A
                    stack.append(result)
                    if explain:
                        explanations.append(f"NOT → {len(result)} diseases")
                else:
                    if len(stack) < 2:
                        continue
                    B = stack.pop()
                    A = stack.pop()
                    
                    if token == "and":
                        result = A & B
                        op_name = "AND (intersection)"
                    else:
                        result = A | B
                        op_name = "OR (union)"
                    
                    if explain:
                        explanations.append(f"{op_name} → {len(result)} diseases")
                    
                    stack.append(result)
        
        final_result = stack.pop() if stack else set()
        return final_result, explanations
    
    def search(self, query: str, explain: bool = True) -> Dict:
        """Search dengan Boolean query"""
        print(f"\n{'='*100}")
        print(f"🔍 BOOLEAN SEARCH: {query}")
        print(f"{'='*100}")
        
        postfix = self.parse_query(query)
        print(f"Postfix notation: {' '.join(postfix)}")
        
        result_docs, explanations = self.eval_postfix(postfix, explain=True)
        
        if explain and explanations:
            print(f"\nEvaluation Steps:")
            for i, step in enumerate(explanations, 1):
                print(f"  {i}. {step}")
        
        print(f"\n📊 Result: {len(result_docs)} disease(s) found")
        if result_docs:
            sorted_results = sorted(result_docs)
            for doc_id in sorted_results:
                disease_name = self.docs_data[doc_id]['disease_name']
                gejala = self.docs_data[doc_id]['gejala']
                print(f"  • {disease_name}: {gejala}")
        else:
            print("  No diseases match the query.")
        print(f"{'='*100}\n")
        
        return {
            'query': query,
            'postfix': postfix,
            'result': sorted(result_docs),
            'count': len(result_docs),
            'explanations': explanations
        }
    
    def search_by_symptoms(self, symptoms: List[str], operator: str = "OR", 
                          explain: bool = True) -> Dict:
        """
        Search penyakit berdasarkan daftar gejala
        
        Args:
            symptoms: List gejala, misal ['demam', 'batuk', 'lemas']
            operator: 'OR' (punya salah satu) atau 'AND' (semua gejala)
            explain: Show explanation
        
        Returns:
            Dictionary dengan hasil dan ranking
        """
        # Build query
        if operator.upper() == "OR":
            query = " OR ".join(symptoms)
        else:
            query = " AND ".join(symptoms)
        
        # Search
        result = self.search(query, explain=explain)
        
        # Ranking berdasarkan jumlah gejala yang match
        doc_scores = {}
        for doc_id in result['result']:
            doc_tokens = set(self.docs_tokens[doc_id])
            match_count = sum(1 for symptom in symptoms if symptom in doc_tokens)
            match_percentage = (match_count / len(symptoms)) * 100
            doc_scores[doc_id] = {
                'match_count': match_count,
                'match_percentage': match_percentage,
                'matched_symptoms': [s for s in symptoms if s in doc_tokens]
            }
        
        # Sort by match count descending
        ranked = sorted(doc_scores.items(), 
                       key=lambda x: x[1]['match_count'], 
                       reverse=True)
        
        print("📈 RANKED RESULTS (by symptom match):")
        print("-" * 100)
        for doc_id, score in ranked:
            disease_name = self.docs_data[doc_id]['disease_name']
            matched = score['matched_symptoms']
            print(f"  {disease_name}: {score['match_count']}/{len(symptoms)} symptoms ({score['match_percentage']:.1f}%)")
            print(f"    Matched: {', '.join(matched)}")
        print("=" * 100)
        
        return {
            'query': query,
            'symptoms': symptoms,
            'operator': operator,
            'result': result['result'],
            'ranked': ranked,
            'scores': doc_scores
        }
    
    def explain_match(self, doc_id: str, symptoms: List[str]):
        """Jelaskan kenapa penyakit ini match dengan gejala"""
        doc_data = self.docs_data[doc_id]
        doc_tokens = set(self.docs_tokens[doc_id])
        
        matched = [s for s in symptoms if s in doc_tokens]
        not_matched = [s for s in symptoms if s not in doc_tokens]
        
        print(f"\n{'='*100}")
        print(f"🏥 DISEASE MATCH EXPLANATION: {doc_data['disease_name']}")
        print(f"{'='*100}")
        print(f"Gejala penyakit: {doc_data['gejala']}")
        print(f"Deskripsi: {doc_data['deskripsi']}")
        print(f"\n✅ Matched symptoms ({len(matched)}/{len(symptoms)}):")
        for symptom in matched:
            print(f"  ✓ {symptom}")
        
        if not_matched:
            print(f"\n❌ Not matched symptoms:")
            for symptom in not_matched:
                print(f"  ✗ {symptom}")
        
        match_rate = (len(matched) / len(symptoms) * 100) if symptoms else 0
        print(f"\nMatch rate: {match_rate:.1f}%")
        print(f"{'='*100}\n")
    
    def evaluate(self, query: str, gold_relevant: Set[str]) -> Dict[str, float]:
        """
        Evaluasi dengan gold standard
        Menghitung Precision, Recall, F1
        """
        result = self.search(query, explain=False)
        retrieved = set(result['result'])
        relevant = gold_relevant
        
        tp = len(retrieved & relevant)
        fp = len(retrieved - relevant)
        fn = len(relevant - retrieved)
        
        precision = tp / len(retrieved) if len(retrieved) > 0 else 0.0
        recall = tp / len(relevant) if len(relevant) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n{'='*100}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*100}")
        print(f"Query: {query}")
        print(f"Retrieved: {sorted(retrieved)}")
        print(f"Relevant (gold): {sorted(relevant)}")
        print(f"\nMetrics:")
        print(f"  True Positives (TP): {tp}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"{'='*100}\n")
        
        return {
            'query': query,
            'retrieved': sorted(retrieved),
            'relevant': sorted(relevant),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def get_index_stats(self):
        """Statistik inverted index"""
        print(f"\n{'='*100}")
        print("INVERTED INDEX STATISTICS")
        print(f"{'='*100}")
        print(f"Vocabulary size: {len(self.vocabulary):,}")
        print(f"Number of diseases: {len(self.doc_ids):,}")
        
        posting_lengths = [len(postings) for postings in self.inverted_index.values()]
        if posting_lengths:
            avg_posting = sum(posting_lengths) / len(posting_lengths)
            print(f"Average posting list length: {avg_posting:.2f}")
            print(f"Max posting list length: {max(posting_lengths)}")
            print(f"Min posting list length: {min(posting_lengths)}")
        
        # Top symptoms (terms dengan posting list terpanjang)
        top_terms = sorted(self.inverted_index.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)[:10]
        
        print(f"\nTop 10 most common symptoms/terms:")
        for term, postings in top_terms:
            print(f"  {term}: appears in {len(postings)} disease(s)")
        
        print(f"{'='*100}\n")


if __name__ == "__main__":
    # Demo akan dijalankan dari main script
    print("Boolean Retrieval Module for Medical Search Engine")
    print("Import this module and use with preprocessed medical documents.")