import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

class VSMRetrieval:
    def __init__(self, docs_data: List[Dict], weighting_scheme: str = "tfidf"):
        """
        Initialize Vector Space Model untuk medical search
        
        Args:
            docs_data: List of document dictionaries dari preprocess_corpus
            weighting_scheme: 'tfidf', 'tfidf_sublinear', or 'bm25'
        """
        self.docs_data = {doc['doc_id']: doc for doc in docs_data}
        self.docs_tokens = {doc['doc_id']: doc['tokens'] for doc in docs_data}
        self.doc_ids = sorted(self.docs_tokens.keys())
        self.weighting_scheme = weighting_scheme
        
        # Build vocabulary and statistics
        self.vocab = self._build_vocab()
        self.term_to_idx = {term: idx for idx, term in enumerate(self.vocab)}
        self.df = self._compute_df()
        self.idf = self._compute_idf()
        
        # Compute document vectors
        self.doc_vectors = self._compute_doc_vectors()
        self.doc_lengths = self._compute_doc_lengths()
        
        print(f"✅ VSM initialized with {len(self.doc_ids)} documents")
        print(f"   Vocabulary size: {len(self.vocab)}")
        print(f"   Weighting scheme: {weighting_scheme}")
    
    def _build_vocab(self) -> List[str]:
        """Build vocabulary dari semua dokumen"""
        vocab = set()
        for tokens in self.docs_tokens.values():
            vocab.update(tokens)
        return sorted(list(vocab))
    
    def _compute_df(self) -> Dict[str, int]:
        """
        Compute Document Frequency (DF) untuk setiap term
        DF = jumlah dokumen yang mengandung term
        """
        df = defaultdict(int)
        for tokens in self.docs_tokens.values():
            unique_tokens = set(tokens)
            for term in unique_tokens:
                df[term] += 1
        return dict(df)
    
    def _compute_idf(self) -> Dict[str, float]:
        """
        Compute Inverse Document Frequency (IDF)
        IDF(t) = log((N + 1) / (df(t) + 1)) + 1
        """
        N = len(self.docs_tokens)
        idf = {}
        
        for term in self.vocab:
            df_t = self.df.get(term, 0)
            # Smoothed IDF
            idf[term] = math.log((N + 1) / (df_t + 1)) + 1
        
        return idf
    
    def _compute_tf(self, tokens: List[str], sublinear: bool = False) -> Dict[str, float]:
        """
        Compute Term Frequency (TF)
        
        Args:
            tokens: List of tokens in document
            sublinear: If True, use 1 + log(tf) instead of raw tf
        
        Returns:
            Dict of {term: tf_value}
        """
        tf = Counter(tokens)
        
        if sublinear:
            # Sublinear TF scaling: 1 + log(tf) if tf > 0, else 0
            tf = {term: (1 + math.log(count)) for term, count in tf.items()}
        else:
            # Raw TF
            tf = dict(tf)
        
        return tf
    
    def _compute_tfidf_vector(self, tokens: List[str], sublinear: bool = False) -> np.ndarray:
        """
        Compute TF-IDF vector untuk dokumen atau query
        
        Returns:
            Dense numpy array of size len(vocab)
        """
        tf = self._compute_tf(tokens, sublinear=sublinear)
        vector = np.zeros(len(self.vocab))
        
        for term, tf_value in tf.items():
            if term in self.term_to_idx:
                idx = self.term_to_idx[term]
                idf_value = self.idf.get(term, 0)
                vector[idx] = tf_value * idf_value
        
        return vector
    
    def _compute_bm25_vector(self, tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """
        Compute BM25 vector (optional bonus untuk Soal 05)
        
        BM25(q,d) = Σ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1*(1-b+b*|d|/avgdl))
        
        Args:
            tokens: Document/query tokens
            k1: BM25 parameter (term frequency saturation)
            b: BM25 parameter (length normalization)
        
        Returns:
            BM25 vector
        """
        tf = Counter(tokens)
        doc_len = len(tokens)
        avg_doc_len = np.mean([len(t) for t in self.docs_tokens.values()])
        
        vector = np.zeros(len(self.vocab))
        
        for term, freq in tf.items():
            if term in self.term_to_idx:
                idx = self.term_to_idx[term]
                idf_value = self.idf.get(term, 0)
                
                # BM25 score
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
                bm25_score = idf_value * (numerator / denominator)
                
                vector[idx] = bm25_score
        
        return vector
    
    def _compute_doc_vectors(self) -> Dict[str, np.ndarray]:
        """Compute vectors untuk semua dokumen"""
        vectors = {}
        
        for doc_id, tokens in self.docs_tokens.items():
            if self.weighting_scheme == "tfidf":
                vectors[doc_id] = self._compute_tfidf_vector(tokens, sublinear=False)
            elif self.weighting_scheme == "tfidf_sublinear":
                vectors[doc_id] = self._compute_tfidf_vector(tokens, sublinear=True)
            elif self.weighting_scheme == "bm25":
                vectors[doc_id] = self._compute_bm25_vector(tokens)
            else:
                raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")
        
        return vectors
    
    def _compute_doc_lengths(self) -> Dict[str, int]:
        """Compute document lengths (for analysis)"""
        return {doc_id: len(tokens) for doc_id, tokens in self.docs_tokens.items()}
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity antara dua vector
        
        cosine(v1, v2) = (v1 · v2) / (||v1|| * ||v2||)
        """
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query_tokens: List[str], top_k: int = 10, 
               explain: bool = True) -> List[Tuple[str, float]]:
        """
        Search dokumen menggunakan VSM
        
        Args:
            query_tokens: Preprocessed query tokens
            top_k: Number of top results to return
            explain: Show explanation
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        # Compute query vector dengan scheme yang sama
        if self.weighting_scheme == "tfidf":
            query_vector = self._compute_tfidf_vector(query_tokens, sublinear=False)
        elif self.weighting_scheme == "tfidf_sublinear":
            query_vector = self._compute_tfidf_vector(query_tokens, sublinear=True)
        elif self.weighting_scheme == "bm25":
            query_vector = self._compute_bm25_vector(query_tokens)
        else:
            query_vector = self._compute_tfidf_vector(query_tokens)
        
        # Compute cosine similarity dengan semua dokumen
        scores = []
        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            scores.append((doc_id, similarity))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        if explain:
            self._display_search_results(query_tokens, scores[:top_k])
        
        return scores[:top_k]
    
    def _display_search_results(self, query_tokens: List[str], 
                               results: List[Tuple[str, float]]):
        """Display search results dengan formatting"""
        print(f"\n{'='*100}")
        print(f"🔍 VSM SEARCH: {' '.join(query_tokens)}")
        print(f"{'='*100}")
        print(f"Weighting scheme: {self.weighting_scheme}")
        print(f"Top {len(results)} results:\n")
        
        if not results or results[0][1] == 0:
            print("❌ No relevant documents found.")
            print("="*100 + "\n")
            return
        
        for rank, (doc_id, score) in enumerate(results, 1):
            doc_data = self.docs_data[doc_id]
            disease_name = doc_data['disease_name']
            gejala = doc_data['gejala'][:80] + "..." if len(doc_data['gejala']) > 80 else doc_data['gejala']
            
            print(f"{rank}. {disease_name} (score: {score:.4f})")
            print(f"   Gejala: {gejala}")
            
            # Show matching terms
            doc_tokens_set = set(self.docs_tokens[doc_id])
            query_tokens_set = set(query_tokens)
            matched_terms = doc_tokens_set & query_tokens_set
            
            if matched_terms:
                print(f"   Matching terms: {', '.join(sorted(matched_terms)[:10])}")
            print()
        
        print("="*100 + "\n")
    
    def explain_document_score(self, query_tokens: List[str], doc_id: str):
        """
        Jelaskan kenapa dokumen ini dapat score tertentu
        Useful untuk debugging dan understanding
        """
        if doc_id not in self.doc_vectors:
            print(f"Document {doc_id} not found.")
            return
        
        # Compute query vector
        if self.weighting_scheme.startswith("tfidf"):
            sublinear = "sublinear" in self.weighting_scheme
            query_vector = self._compute_tfidf_vector(query_tokens, sublinear=sublinear)
        else:
            query_vector = self._compute_bm25_vector(query_tokens)
        
        doc_vector = self.doc_vectors[doc_id]
        doc_data = self.docs_data[doc_id]
        
        print(f"\n{'='*100}")
        print(f"📊 SCORE EXPLANATION: {doc_data['disease_name']}")
        print(f"{'='*100}")
        
        # Overall similarity
        similarity = self._cosine_similarity(query_vector, doc_vector)
        print(f"Cosine Similarity: {similarity:.4f}\n")
        
        # Term-by-term breakdown
        print("Term-by-term contribution:")
        print(f"{'Term':<15} {'Query Weight':<15} {'Doc Weight':<15} {'Contribution':<15}")
        print("-" * 100)
        
        contributions = []
        for term in query_tokens:
            if term in self.term_to_idx:
                idx = self.term_to_idx[term]
                q_weight = query_vector[idx]
                d_weight = doc_vector[idx]
                contribution = q_weight * d_weight
                
                contributions.append((term, q_weight, d_weight, contribution))
        
        # Sort by contribution
        contributions.sort(key=lambda x: x[3], reverse=True)
        
        for term, q_w, d_w, contrib in contributions[:10]:
            print(f"{term:<15} {q_w:<15.4f} {d_w:<15.4f} {contrib:<15.4f}")
        
        print("="*100 + "\n")
    
    def get_term_statistics(self):
        """Display term statistics (TF, DF, IDF)"""
        print(f"\n{'='*100}")
        print("📊 TERM STATISTICS")
        print(f"{'='*100}")
        
        # Top terms by IDF (discriminative terms)
        top_idf = sorted(self.idf.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 terms by IDF (most discriminative):")
        for term, idf_val in top_idf:
            df_val = self.df.get(term, 0)
            print(f"  {term:<20} IDF: {idf_val:.4f}, DF: {df_val}")
        
        # Top terms by DF (most common)
        top_df = sorted(self.df.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 terms by DF (most common):")
        for term, df_val in top_df:
            idf_val = self.idf.get(term, 0)
            print(f"  {term:<20} DF: {df_val}, IDF: {idf_val:.4f}")
        
        print("="*100 + "\n")
    
    def compare_weighting_schemes(self, query_tokens: List[str], 
                                 relevant_docs: Set[str],
                                 k: int = 10) -> Dict:
        """
        Bandingkan hasil dari berbagai weighting schemes
        Untuk Soal 05: perbandingan skema bobot istilah
        
        Args:
            query_tokens: Query yang sudah dipreprocess
            relevant_docs: Ground truth relevant documents
            k: Top-k untuk evaluasi
        
        Returns:
            Dictionary berisi hasil dari setiap scheme
        """
        from eval import precision_at_k, recall_at_k, f1_at_k, average_precision
        
        schemes = ["tfidf", "tfidf_sublinear"]
        
        # Tambahkan BM25 jika mau bonus
        try:
            schemes.append("bm25")
        except:
            pass
        
        results = {}
        
        print(f"\n{'='*100}")
        print(f"🔬 COMPARING WEIGHTING SCHEMES")
        print(f"{'='*100}")
        print(f"Query: {' '.join(query_tokens)}")
        print(f"Relevant docs (gold): {sorted(relevant_docs)}\n")
        
        for scheme in schemes:
            # Create temporary VSM with this scheme
            temp_vsm = VSMRetrieval([doc for doc in self.docs_data.values()], 
                                   weighting_scheme=scheme)
            
            # Search
            ranked_results = temp_vsm.search(query_tokens, top_k=k, explain=False)
            
            # Evaluate
            p_at_k = precision_at_k(ranked_results, relevant_docs, k)
            r_at_k = recall_at_k(ranked_results, relevant_docs, k)
            f1 = f1_at_k(ranked_results, relevant_docs, k)
            ap = average_precision(ranked_results, relevant_docs)
            
            results[scheme] = {
                'ranked_results': ranked_results,
                'precision@k': p_at_k,
                'recall@k': r_at_k,
                'f1@k': f1,
                'average_precision': ap
            }
            
            print(f"{scheme.upper()}:")
            print(f"  Top results: {[doc_id for doc_id, _ in ranked_results[:5]]}")
            print(f"  P@{k}: {p_at_k:.4f}, R@{k}: {r_at_k:.4f}, F1@{k}: {f1:.4f}, AP: {ap:.4f}\n")
        
        # Determine best scheme
        best_scheme = max(results.items(), key=lambda x: x[1]['f1@k'])
        print(f"🏆 Best scheme (by F1@{k}): {best_scheme[0].upper()}")
        print("="*100 + "\n")
        
        return results
    
    def get_vector_stats(self):
        """Display vector statistics"""
        print(f"\n{'='*100}")
        print("📊 VECTOR STATISTICS")
        print(f"{'='*100}")
        
        # Compute sparsity
        total_elements = len(self.doc_vectors) * len(self.vocab)
        non_zero = sum(np.count_nonzero(vec) for vec in self.doc_vectors.values())
        sparsity = (1 - non_zero / total_elements) * 100
        
        print(f"Vector dimension: {len(self.vocab)}")
        print(f"Number of vectors: {len(self.doc_vectors)}")
        print(f"Sparsity: {sparsity:.2f}%")
        
        # Average non-zero elements per vector
        avg_nonzero = non_zero / len(self.doc_vectors)
        print(f"Average non-zero elements per vector: {avg_nonzero:.2f}")
        
        print("="*100 + "\n")


if __name__ == "__main__":
    print("Vector Space Model Module for Medical Search Engine")
    print("Import this module and use with preprocessed medical documents.")