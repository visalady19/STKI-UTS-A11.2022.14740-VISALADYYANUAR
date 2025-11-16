import numpy as np
from typing import List, Set, Dict, Tuple
import matplotlib.pyplot as plt

def precision_at_k(ranked_results: List[Tuple[str, float]], 
                   relevant_docs: Set[str], 
                   k: int = 10) -> float:
    """
    Precision@k: proporsi dokumen relevan di top-k hasil
    
    Args:
        ranked_results: List of (doc_id, score) tuples, sorted by score desc
        relevant_docs: Set of doc_ids yang relevan (gold standard)
        k: Number of top results to consider
    
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if not ranked_results or k <= 0:
        return 0.0
    
    ranked = [doc for doc, _ in ranked_results[:k]]
    hits = sum(1 for doc in ranked if doc in relevant_docs)
    return hits / k

def recall_at_k(ranked_results: List[Tuple[str, float]], 
                relevant_docs: Set[str], 
                k: int = 10) -> float:
    """
    Recall@k: proporsi dokumen relevan yang ditemukan di top-k
    
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant_docs or not ranked_results or k <= 0:
        return 0.0
    
    ranked = [doc for doc, _ in ranked_results[:k]]
    hits = sum(1 for doc in ranked if doc in relevant_docs)
    return hits / len(relevant_docs)

def f1_at_k(ranked_results: List[Tuple[str, float]], 
            relevant_docs: Set[str], 
            k: int = 10) -> float:
    """
    F1@k: harmonic mean of Precision@k and Recall@k
    
    Returns:
        F1@k score (0.0 to 1.0)
    """
    precision = precision_at_k(ranked_results, relevant_docs, k)
    recall = recall_at_k(ranked_results, relevant_docs, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def average_precision(ranked_results: List[Tuple[str, float]], 
                     relevant_docs: Set[str]) -> float:
    """
    Average Precision (AP): mean precision at each relevant document position
    
    Returns:
        AP score (0.0 to 1.0)
    """
    if not relevant_docs or not ranked_results:
        return 0.0
    
    hits = 0
    score = 0.0
    
    for i, (doc_id, _) in enumerate(ranked_results, start=1):
        if doc_id in relevant_docs:
            hits += 1
            precision_at_i = hits / i
            score += precision_at_i
    
    return score / len(relevant_docs)

def map_at_k(queries_results: List[Tuple[List[Tuple[str, float]], Set[str]]], 
             k: int = 10) -> float:
    """
    Mean Average Precision at k (MAP@k)
    
    Args:
        queries_results: List of (ranked_results, relevant_docs) tuples
        k: Truncate rankings at k
    
    Returns:
        MAP@k score (0.0 to 1.0)
    """
    if not queries_results:
        return 0.0
    
    ap_scores = []
    for ranked_results, relevant_docs in queries_results:
        # Truncate to k
        truncated = ranked_results[:k]
        ap = average_precision(truncated, relevant_docs)
        ap_scores.append(ap)
    
    return sum(ap_scores) / len(ap_scores)

def dcg_at_k(ranked_results: List[Tuple[str, float]], 
             relevance_scores: Dict[str, int], 
             k: int = 10) -> float:
    """
    Discounted Cumulative Gain at k (DCG@k)
    
    Args:
        ranked_results: List of (doc_id, score) tuples
        relevance_scores: Dict of {doc_id: relevance_grade} (e.g., 0-3 scale)
        k: Number of top results
    
    Returns:
        DCG@k score
    """
    dcg = 0.0
    for i, (doc_id, _) in enumerate(ranked_results[:k], start=1):
        rel = relevance_scores.get(doc_id, 0)
        dcg += (2**rel - 1) / np.log2(i + 1)
    
    return dcg

def ndcg_at_k(ranked_results: List[Tuple[str, float]], 
              relevance_scores: Dict[str, int], 
              k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at k (nDCG@k)
    
    Args:
        ranked_results: List of (doc_id, score) tuples
        relevance_scores: Dict of {doc_id: relevance_grade}
        k: Number of top results
    
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    dcg = dcg_at_k(ranked_results, relevance_scores, k)
    
    # Ideal DCG: sort by relevance scores descending
    ideal_ranking = sorted(relevance_scores.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:k]
    
    idcg = 0.0
    for i, (doc_id, rel) in enumerate(ideal_ranking, start=1):
        idcg += (2**rel - 1) / np.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def evaluate_ranking(ranked_results: List[Tuple[str, float]], 
                    relevant_docs: Set[str],
                    relevance_scores: Dict[str, int] = None,
                    k_values: List[int] = [3, 5, 10]) -> Dict:
    """
    Comprehensive evaluation untuk satu query
    
    Args:
        ranked_results: Ranked list of (doc_id, score)
        relevant_docs: Binary relevance (set of doc_ids)
        relevance_scores: Graded relevance (optional, for nDCG)
        k_values: List of k values to evaluate at
    
    Returns:
        Dictionary of metrics
    """
    results = {
        'total_retrieved': len(ranked_results),
        'total_relevant': len(relevant_docs),
        'metrics': {}
    }
    
    for k in k_values:
        results['metrics'][f'P@{k}'] = precision_at_k(ranked_results, relevant_docs, k)
        results['metrics'][f'R@{k}'] = recall_at_k(ranked_results, relevant_docs, k)
        results['metrics'][f'F1@{k}'] = f1_at_k(ranked_results, relevant_docs, k)
        
        if relevance_scores:
            results['metrics'][f'nDCG@{k}'] = ndcg_at_k(ranked_results, relevance_scores, k)
    
    # Overall metrics (no k limit)
    results['metrics']['AP'] = average_precision(ranked_results, relevant_docs)
    
    return results

def compare_systems(system_results: Dict[str, List[Tuple[List, Set]]], 
                   k: int = 10) -> Dict:
    """
    Bandingkan beberapa sistem IR (misal: TF-IDF vs BM25)
    
    Args:
        system_results: Dict of {system_name: [(ranked_results, relevant_docs), ...]}
        k: k value for evaluation
    
    Returns:
        Comparison dictionary dengan metrics untuk setiap sistem
    """
    comparison = {}
    
    for system_name, queries_results in system_results.items():
        # Calculate metrics untuk sistem ini
        all_ap = []
        all_p_at_k = []
        all_r_at_k = []
        all_f1_at_k = []
        
        for ranked_results, relevant_docs in queries_results:
            all_ap.append(average_precision(ranked_results[:k], relevant_docs))
            all_p_at_k.append(precision_at_k(ranked_results, relevant_docs, k))
            all_r_at_k.append(recall_at_k(ranked_results, relevant_docs, k))
            all_f1_at_k.append(f1_at_k(ranked_results, relevant_docs, k))
        
        comparison[system_name] = {
            f'MAP@{k}': np.mean(all_ap),
            f'P@{k}': np.mean(all_p_at_k),
            f'R@{k}': np.mean(all_r_at_k),
            f'F1@{k}': np.mean(all_f1_at_k),
            'num_queries': len(queries_results)
        }
    
    return comparison

def print_evaluation_results(results: Dict, query_name: str = "Query"):
    """Pretty print evaluation results"""
    print(f"\n{'='*100}")
    print(f"📊 EVALUATION RESULTS: {query_name}")
    print(f"{'='*100}")
    print(f"Total retrieved: {results['total_retrieved']}")
    print(f"Total relevant: {results['total_relevant']}")
    print(f"\nMetrics:")
    
    for metric_name, value in results['metrics'].items():
        print(f"  {metric_name:<12}: {value:.4f}")
    
    print(f"{'='*100}\n")

def print_comparison_table(comparison: Dict):
    """Pretty print system comparison"""
    print(f"\n{'='*100}")
    print(f"📊 SYSTEM COMPARISON")
    print(f"{'='*100}")
    
    # Header
    systems = list(comparison.keys())
    metrics = list(comparison[systems[0]].keys())
    metrics.remove('num_queries')
    
    print(f"{'System':<20}", end="")
    for metric in metrics:
        print(f"{metric:<15}", end="")
    print()
    print("-" * 100)
    
    # Rows
    for system_name in systems:
        print(f"{system_name:<20}", end="")
        for metric in metrics:
            value = comparison[system_name][metric]
            print(f"{value:<15.4f}", end="")
        print()
    
    print(f"{'='*100}\n")

def plot_precision_recall_curve(ranked_results: List[Tuple[str, float]], 
                                relevant_docs: Set[str],
                                title: str = "Precision-Recall Curve"):
    """
    Plot Precision-Recall curve
    """
    precisions = []
    recalls = []
    
    for k in range(1, len(ranked_results) + 1):
        p = precision_at_k(ranked_results, relevant_docs, k)
        r = recall_at_k(ranked_results, relevant_docs, k)
        precisions.append(p)
        recalls.append(r)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    return plt

def plot_metrics_comparison(comparison: Dict, 
                           metrics: List[str] = None,
                           title: str = "System Comparison"):
    """
    Bar plot untuk membandingkan metrics antar sistem
    """
    if metrics is None:
        # Ambil semua metrics kecuali num_queries
        first_system = list(comparison.keys())[0]
        metrics = [m for m in comparison[first_system].keys() if m != 'num_queries']
    
    systems = list(comparison.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(systems)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, system in enumerate(systems):
        values = [comparison[system][metric] for metric in metrics]
        offset = (i - len(systems)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=system)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return plt

def create_confusion_matrix(retrieved: Set[str], 
                           relevant: Set[str], 
                           all_docs: Set[str]) -> Dict:
    """
    Create confusion matrix untuk binary classification
    
    Returns:
        Dict dengan TP, FP, TN, FN
    """
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    tn = len(all_docs - retrieved - relevant)
    
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Accuracy': (tp + tn) / len(all_docs) if len(all_docs) > 0 else 0,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }

def print_confusion_matrix(cm: Dict):
    """Pretty print confusion matrix"""
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"                Predicted")
    print(f"                Relevant    Not Relevant")
    print(f"Actual Relevant    {cm['TP']:<8}    {cm['FN']:<8}")
    print(f"       Not Rel     {cm['FP']:<8}    {cm['TN']:<8}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {cm['Accuracy']:.4f}")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"{'='*60}\n")


# ==================== DEMO ====================
if __name__ == "__main__":
    print("Evaluation Metrics Module for Medical Search Engine")
    print("\nAvailable functions:")
    print("  - precision_at_k, recall_at_k, f1_at_k")
    print("  - average_precision, map_at_k")
    print("  - ndcg_at_k (for graded relevance)")
    print("  - evaluate_ranking (comprehensive)")
    print("  - compare_systems (compare multiple IR systems)")
    print("  - plot_precision_recall_curve, plot_metrics_comparison")
    
    # Demo dengan data dummy
    print("\n" + "="*60)
    print("DEMO EVALUATION")
    print("="*60)
    
    # Dummy ranked results
    ranked_results = [
        ('Anemia', 0.95),
        ('Diabetes2', 0.87),
        ('Flu', 0.76),
        ('Demam_Berdarah', 0.65),
        ('Asma', 0.54)
    ]
    
    # Dummy relevance
    relevant_docs = {'Anemia', 'Demam_Berdarah', 'Malaria'}
    
    # Evaluate
    results = evaluate_ranking(ranked_results, relevant_docs, k_values=[3, 5])
    print_evaluation_results(results, "demam AND lemas")
    
    # Comparison demo
    system_results = {
        'TF-IDF': [(ranked_results, relevant_docs)],
        'BM25': [(ranked_results[::-1], relevant_docs)]  # reverse order untuk demo
    }
    
    comparison = compare_systems(system_results, k=5)
    print_comparison_table(comparison)