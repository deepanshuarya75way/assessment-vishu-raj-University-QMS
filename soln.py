from pydantic import BaseModel

class DocumentResult(BaseModel):
  document_id: str
  metadata: Dict[str, Any]
  score: float
from typing import List, Dict, Any, Optional

class HybridSearchService:
  def __init__(self,rrf_k: int = 60):
    self.rrf_k = rrf_k

  def hybrid_search(
    self,
    dense_results: List[DocumentResult],
    sparse_results: List[DocumentResult],
    top_k: int
  )-> List[DocumentResult]:
  """it will perform hybrid search by combining dense and sparse results using rrf"""

    rrf_scores: Dict[str,float] = {}
    document_metadata_map: Dict[str, Dict[str, Any]] = {}

    #1. process dense results and assign rrf

    for rank, doc in enumerate(dense_results, start = 1):
      document_metadata_map.setdefault(doc.document_id, doc.metadata)
      rrf_score = 1.0 / (self.rrf_k + rank)
      rrf_scores[doc.document_id] = rrf_scores.get(doc.document_id, 0.0) + rrf_score

    # 2. process sparse results and assign rrf score
    for rank, doc in enumerate(sparse_results, start = 1):
      document_metadata_map.setdefault(doc.document_id, doc.metadata)
      rrf_score = 1.0 / (self.rrf_k + rank)
      rrf_scores[doc.document_id] = rrf_scores.get(doc.document_id, 0.0) + rrf_score

    #3. sort doc by theire combined rrf scores in desc
    sorted_docs = sorted(rrf_scores.items(), key = lambda x: x[1],reverse = True) 

    #4. building final result list

    final_results: List[DocumentResult] = []
    for doc_id, final_score in sorted_docs[:top_k]:
      final_results.append(
        DocumentResult(
          document_id = doc_id,
          metadata = document_metadata_map[doc_id],
          score = final_score
        )
      )
    return final_results