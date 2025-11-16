import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

# 🔥 FIX: pastikan modul preprocess milik Anda yang dipakai,
# bukan preprocess library bawaan Python
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preprocess import preprocess_corpus, preprocess
from boolean_ir import BooleanRetrieval
from vsm_ir import VSMRetrieval
from eval import evaluate_ranking, compare_systems, print_evaluation_results


class MedicalSearchEngine:
    """
    Medical Search Engine Supporting Boolean + VSM Retrieval
    """

    def __init__(self, data_dir: str = "data", weighting_scheme: str = "tfidf"):
        """
        Initialize Search Engine
        
        Args:
            data_dir: folder berisi dokumen penyakit (.txt)
            weighting_scheme: tfidf / tfidf_sublinear / bm25
        """

        self.data_dir = data_dir
        self.weighting_scheme = weighting_scheme

        print("\n" + "=" * 100)
        print("🏥 INITIALIZING MEDICAL SEARCH ENGINE")
        print("=" * 100)
        print(f"Data folder       : {data_dir}")
        print(f"Weighting scheme  : {weighting_scheme}")

        # Load + preprocess dokumen penyakit
        self.documents = self._load_and_preprocess()

        # Build retrieval system
        self.boolean = BooleanRetrieval(self.documents)
        self.vsm = VSMRetrieval(self.documents, weighting_scheme=weighting_scheme)

        print("\n✅ Search Engine Ready!")
        print(f"Total documents: {len(self.documents)}")
        print("=" * 100 + "\n")

    # =========================================================================
    # LOADING & PREPROCESSING
    # =========================================================================
    def _load_and_preprocess(self) -> List[Dict]:
        """Load dan preprocess seluruh dokumen di folder data/"""
        docs = preprocess_corpus(self.data_dir, use_stemming=True)

        if not docs:
            raise ValueError(f"❌ Tidak ditemukan dokumen di folder {self.data_dir}")

        return docs

    # =========================================================================
    # BOOLEAN PREPROCESSING (khusus Boolean Retrieval)
    # =========================================================================
    def preprocess_boolean_query(self, query: str) -> str:
        """
        Preprocess query Boolean TANPA merusak operator AND/OR/NOT.
        
        Contoh:
            "demam AND virus"
            -> "demam AND virus"
        
        Query kata di-stemming, operator tetap.
        """

        tokens = query.split()
        processed_tokens = []

        for tok in tokens:
            upper_tok = tok.upper()

            # Jika operator, pertahankan bentuknya
            if upper_tok in ["AND", "OR", "NOT"]:
                processed_tokens.append(upper_tok)

            else:
                # Preprocess kata
                stemmed = preprocess(tok, use_stemming=True)
                if stemmed:
                    processed_tokens.append(stemmed[0])

        return " ".join(processed_tokens)

    # =========================================================================
    # MAIN SEARCH FUNCTION
    # =========================================================================
    def search(self, query: str, model: str = "vsm", top_k: int = 10,
               explain: bool = False) -> List[Tuple[str, float]]:
        """
        Melakukan pencarian menggunakan Boolean atau VSM.
        
        Returns:
            list of (doc_id, score)
        """

        # ---------------------- BOOLEAN RETRIEVAL ----------------------
        if model.lower() == "boolean":

            processed_query = self.preprocess_boolean_query(query)

            result_docs = self.boolean.search(processed_query, explain=explain)

            # Convert ke format (doc_id, score)
            return [(doc_id, 1.0) for doc_id in result_docs["result"]]

        # ---------------------- VSM RETRIEVAL --------------------------
        elif model.lower() == "vsm":

            tokens = preprocess(query, use_stemming=True)
            return self.vsm.search(tokens, top_k=top_k, explain=explain)

        else:
            print(f"❌ Unknown model: {model}")
            return []

    # =========================================================================
    # SEARCH BY SYMPTOMS (ORIGINAL)
    # =========================================================================
    def search_by_symptoms(self, symptoms: List[str], operator: str = "OR",
                          model: str = "vsm", top_k: int = 10) -> Dict:

        symptoms_preprocessed = [preprocess(s, use_stemming=True) for s in symptoms]
        symptoms_tokens = [t for tokens in symptoms_preprocessed for t in tokens]

        if model.lower() == "boolean":
            return self.boolean.search_by_symptoms(symptoms_tokens, operator=operator)

        elif model.lower() == "vsm":
            ranked = self.vsm.search(symptoms_tokens, top_k=top_k)
            return {
                "query": " ".join(symptoms),
                "symptoms": symptoms_tokens,
                "results": ranked
            }

    # =========================================================================
    # EXPLAIN RESULTS
    # =========================================================================
    def explain_result(self, query: str, doc_id: str, model: str = "vsm"):
        query_tokens = preprocess(query, use_stemming=True)

        if model.lower() == "boolean":
            processed = self.preprocess_boolean_query(query)
            self.boolean.explain_match(doc_id, processed.split())

        elif model.lower() == "vsm":
            self.vsm.explain_document_score(query_tokens, doc_id)

    # =========================================================================
    # EVALUATION METHODS
    # =========================================================================
    def evaluate(self, test_queries: List[Dict], model: str = "vsm", k: int = 10):

        print("\n" + "=" * 100)
        print(f"📊 Evaluating model: {model.upper()}")
        print("=" * 100)

        all_results = []

        for i, test_case in enumerate(test_queries, 1):

            query = test_case["query"]
            relevant = test_case["relevant"]

            results = self.search(query, model=model, top_k=k, explain=False)

            eval_result = evaluate_ranking(results, relevant, k_values=[3, 5, k])
            print_evaluation_results(eval_result, f"Query {i}")

            all_results.append((results, relevant))

        from eval import map_at_k

        map_score = map_at_k(all_results, k=k)

        print("\n" + "=" * 100)
        print(f"📊 OVERALL MAP@{k}: {map_score:.4f}")
        print("=" * 100)

        return {
            "model": model,
            "k": k,
            "num_queries": len(test_queries),
            "map@k": map_score
        }

    # =========================================================================
    # COMPARE BOOLEAN vs VSM
    # =========================================================================
    def compare_models(self, test_queries: List[Dict], k: int = 10):

        boolean_results = []
        vsm_results = []

        for test_case in test_queries:

            query = test_case["query"]
            relevant = test_case["relevant"]

            # Boolean
            boolean_r = self.search(query, model="boolean", top_k=k)
            boolean_results.append((boolean_r, relevant))

            # VSM
            vsm_r = self.search(query, model="vsm", top_k=k)
            vsm_results.append((vsm_r, relevant))

        system_results = {
            "Boolean": boolean_results,
            "VSM": vsm_results
        }

        comparison = compare_systems(system_results, k=k)

        from eval import print_comparison_table
        print_comparison_table(comparison)

        return comparison

    # =========================================================================
    # STATISTICS
    # =========================================================================
    def get_statistics(self):
        print("\n" + "=" * 100)
        print("📊 SEARCH ENGINE STATISTICS")
        print("=" * 100)

        print("\n🔹 BOOLEAN RETRIEVAL:")
        self.boolean.get_index_stats()

        print("\n🔹 VECTOR SPACE MODEL:")
        self.vsm.get_term_statistics()
        self.vsm.get_vector_stats()

    # =========================================================================
    # INTERACTIVE CLI MODE (ORIGINAL)
    # =========================================================================
    def interactive_search(self):
        print("\n=== MEDICAL SEARCH ENGINE INTERACTIVE MODE ===")

        last_query = ""
        last_model = "vsm"

        while True:
            try:
                cmd = input("🔍 Enter command: ").strip()

                if cmd.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                parts = cmd.split(maxsplit=1)
                if not parts:
                    continue

                command = parts[0].lower()

                if command == "search":
                    q = parts[1]
                    self.search(q, model="vsm")
                    last_query = q
                    last_model = "vsm"

                elif command == "boolean":
                    q = parts[1]
                    self.search(q, model="boolean")
                    last_query = q
                    last_model = "boolean"

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


# =========================================================================
# MAIN CLI ENTRY
# =========================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--model", type=str, choices=["vsm", "boolean"], default="vsm")
    parser.add_argument("--query", type=str)
    parser.add.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    engine = MedicalSearchEngine(args.data)

    if args.query:
        engine.search(args.query, model=args.model, top_k=args.k)

    else:
        engine.interactive_search()


if __name__ == "__main__":
    main()
