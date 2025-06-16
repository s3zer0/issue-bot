"""
키워드 클러스터링 및 확장 모듈.
"""

from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
from loguru import logger

from .base import KeywordItem, KeywordImportance


@dataclass
class KeywordCluster:
    """키워드 클러스터."""
    cluster_id: int
    keywords: List[KeywordItem]
    centroid_embedding: Optional[np.ndarray] = None
    expansion_keyword: Optional[str] = None
    expansion_confidence: float = 0.0
    theme: Optional[str] = None


@dataclass
class ClusteringResult:
    """클러스터링 결과."""
    clusters: List[KeywordCluster]
    expansion_keywords: List[KeywordItem]
    noise_keywords: List[KeywordItem]  # 클러스터에 속하지 않는 키워드
    clustering_time: float
    model_name: str
    metadata: Dict[str, any] = field(default_factory=dict)


class KeywordClusterer:
    """키워드 의미적 클러스터링 및 확장 키워드 생성기."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        eps: float = 0.3,
        min_samples: int = 2,
        expansion_confidence_threshold: float = 0.6
    ):
        """
        키워드 클러스터러 초기화.

        Args:
            model_name: 사용할 sentence transformer 모델
            eps: DBSCAN eps 파라미터 (클러스터링 거리 임계값)
            min_samples: DBSCAN min_samples 파라미터
            expansion_confidence_threshold: 확장 키워드 생성 최소 신뢰도
        """
        self.model_name = model_name
        self.eps = eps
        self.min_samples = min_samples
        self.expansion_confidence_threshold = expansion_confidence_threshold
        
        # Sentence Transformer 모델 로드
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"키워드 클러스터러 초기화 완료 (모델: {model_name})")
        except Exception as e:
            logger.error(f"Sentence Transformer 모델 로드 실패: {e}")
            # 폴백: 더 작은 모델 사용
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.warning("폴백 모델 사용: all-MiniLM-L6-v2")

        # 클러스터링 알고리즘
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')

    def cluster_and_expand(
        self,
        keywords: List[KeywordItem],
        topic: str,
        generate_expansion: bool = True
    ) -> ClusteringResult:
        """
        키워드를 클러스터링하고 확장 키워드를 생성합니다.

        Args:
            keywords: 클러스터링할 키워드 리스트
            topic: 원본 주제 (확장 키워드 생성에 사용)
            generate_expansion: 확장 키워드 생성 여부

        Returns:
            ClusteringResult: 클러스터링 및 확장 결과
        """
        start_time = time.time()
        logger.info(f"키워드 클러스터링 시작: {len(keywords)}개 키워드")

        if not keywords:
            return ClusteringResult(
                clusters=[],
                expansion_keywords=[],
                noise_keywords=[],
                clustering_time=0.0,
                model_name=self.model_name
            )

        # 1. 키워드 임베딩 생성
        keyword_texts = [kw.keyword for kw in keywords]
        try:
            embeddings = self.embedding_model.encode(keyword_texts, convert_to_tensor=False)
            embeddings = np.array(embeddings)
        except Exception as e:
            logger.error(f"키워드 임베딩 생성 실패: {e}")
            return ClusteringResult(
                clusters=[],
                expansion_keywords=[],
                noise_keywords=keywords,
                clustering_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={'error': str(e)}
            )

        # 2. 클러스터링 수행
        cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # 3. 클러스터 생성
        clusters = self._create_clusters(keywords, embeddings, cluster_labels)
        
        # 4. 노이즈 키워드 (클러스터에 속하지 않는 키워드) 분리
        noise_keywords = [
            keywords[i] for i, label in enumerate(cluster_labels) if label == -1
        ]

        # 5. 확장 키워드 생성
        expansion_keywords = []
        if generate_expansion:
            expansion_keywords = self._generate_expansion_keywords(clusters, topic)

        result = ClusteringResult(
            clusters=clusters,
            expansion_keywords=expansion_keywords,
            noise_keywords=noise_keywords,
            clustering_time=time.time() - start_time,
            model_name=self.model_name,
            metadata={
                'topic': topic,
                'total_keywords': len(keywords),
                'num_clusters': len(clusters),
                'num_noise': len(noise_keywords),
                'eps': self.eps,
                'min_samples': self.min_samples
            }
        )

        logger.success(
            f"키워드 클러스터링 완료: "
            f"{len(clusters)}개 클러스터, {len(expansion_keywords)}개 확장 키워드, "
            f"{len(noise_keywords)}개 노이즈 키워드"
        )

        return result

    def _create_clusters(
        self,
        keywords: List[KeywordItem],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> List[KeywordCluster]:
        """클러스터 객체 생성."""
        clusters = []
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # 노이즈 제거

        for cluster_id in unique_labels:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_keywords = [keywords[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            # 클러스터 중심점 계산
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # 클러스터 테마 분석
            theme = self._analyze_cluster_theme(cluster_keywords)
            
            cluster = KeywordCluster(
                cluster_id=int(cluster_id),
                keywords=cluster_keywords,
                centroid_embedding=centroid,
                theme=theme
            )
            clusters.append(cluster)

        # 클러스터를 크기 순으로 정렬 (큰 클러스터 우선)
        clusters.sort(key=lambda x: len(x.keywords), reverse=True)
        
        return clusters

    def _analyze_cluster_theme(self, keywords: List[KeywordItem]) -> str:
        """클러스터의 주제를 분석합니다."""
        # 키워드의 공통 패턴이나 주제를 찾기
        keyword_texts = [kw.keyword.lower() for kw in keywords]
        
        # 간단한 휴리스틱: 가장 빈번한 단어나 패턴 찾기
        all_words = []
        for text in keyword_texts:
            words = text.split()
            all_words.extend(words)
        
        # 단어 빈도 계산
        word_freq = {}
        for word in all_words:
            if len(word) > 2:  # 짧은 단어 제외
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            # 가장 빈번한 단어를 테마로 사용
            most_common_word = max(word_freq, key=word_freq.get)
            if word_freq[most_common_word] >= 2:  # 최소 2번 이상 등장
                return most_common_word
        
        # 폴백: 첫 번째 키워드의 첫 단어
        if keywords:
            first_word = keywords[0].keyword.split()[0].lower()
            return first_word
        
        return "unknown"

    def _generate_expansion_keywords(
        self,
        clusters: List[KeywordCluster],
        topic: str
    ) -> List[KeywordItem]:
        """클러스터 기반 확장 키워드 생성."""
        expansion_keywords = []
        
        for cluster in clusters:
            if len(cluster.keywords) < 2:  # 너무 작은 클러스터는 스킵
                continue

            expansion_keyword = self._create_expansion_keyword(cluster, topic)
            if expansion_keyword:
                expansion_keywords.append(expansion_keyword)

        return expansion_keywords

    def _create_expansion_keyword(
        self,
        cluster: KeywordCluster,
        topic: str
    ) -> Optional[KeywordItem]:
        """단일 클러스터에 대한 확장 키워드 생성."""
        keywords_in_cluster = [kw.keyword for kw in cluster.keywords]
        
        # 클러스터의 특성에 따라 확장 키워드 생성 규칙 적용
        expansion_text = None
        confidence = 0.5

        # 규칙 1: 기술 관련 클러스터
        if any(tech_term in cluster.theme.lower() for tech_term in 
               ['api', 'sdk', 'framework', 'library', 'tool', 'platform']):
            expansion_text = f"{topic} {cluster.theme} ecosystem"
            confidence = 0.8

        # 규칙 2: 버전 관련 클러스터
        elif any(version_term in cluster.theme.lower() for version_term in 
                ['version', 'update', 'release', 'new', 'latest']):
            expansion_text = f"{topic} latest developments"
            confidence = 0.7

        # 규칙 3: 플랫폼/OS 관련 클러스터
        elif any(os_term in cluster.theme.lower() for os_term in 
                ['ios', 'android', 'windows', 'mac', 'linux']):
            expansion_text = f"{topic} {cluster.theme} platform features"
            confidence = 0.75

        # 규칙 4: 일반적인 확장 (폴백)
        else:
            # 클러스터 내 가장 중요한 키워드를 기반으로 확장
            high_importance_keywords = [
                kw for kw in cluster.keywords 
                if kw.importance == KeywordImportance.HIGH
            ]
            
            if high_importance_keywords:
                base_keyword = high_importance_keywords[0].keyword
                expansion_text = f"{topic} {base_keyword} related topics"
                confidence = 0.6
            else:
                expansion_text = f"{topic} {cluster.theme} comprehensive guide"
                confidence = 0.5

        # 확장 키워드 생성
        if expansion_text and confidence >= self.expansion_confidence_threshold:
            # 클러스터의 평균 신뢰도를 확장 키워드에 반영
            avg_confidence = sum(kw.confidence for kw in cluster.keywords) / len(cluster.keywords)
            final_confidence = min(confidence, avg_confidence + 0.1)  # 약간의 보너스
            
            expansion_keyword = KeywordItem(
                keyword=expansion_text,
                sources=['clusterer'],
                importance=KeywordImportance.NORMAL,
                confidence=final_confidence,
                category='expansion'
            )
            
            # 클러스터에 확장 키워드 정보 저장
            cluster.expansion_keyword = expansion_text
            cluster.expansion_confidence = final_confidence
            
            return expansion_keyword

        return None

    def get_cluster_summary(self, result: ClusteringResult) -> Dict[str, any]:
        """클러스터링 결과 요약 생성."""
        summary = {
            'total_clusters': len(result.clusters),
            'total_expansion_keywords': len(result.expansion_keywords),
            'total_noise_keywords': len(result.noise_keywords),
            'clustering_time': result.clustering_time,
            'model_used': result.model_name,
            'clusters_detail': []
        }

        for cluster in result.clusters:
            cluster_detail = {
                'cluster_id': cluster.cluster_id,
                'size': len(cluster.keywords),
                'theme': cluster.theme,
                'expansion_keyword': cluster.expansion_keyword,
                'keywords': [kw.keyword for kw in cluster.keywords[:5]]  # 상위 5개만
            }
            summary['clusters_detail'].append(cluster_detail)

        return summary

    def merge_with_original_keywords(
        self,
        original_keywords: List[KeywordItem],
        clustering_result: ClusteringResult
    ) -> List[KeywordItem]:
        """
        원본 키워드와 확장 키워드를 병합합니다.
        
        Args:
            original_keywords: 원본 키워드 리스트
            clustering_result: 클러스터링 결과
            
        Returns:
            List[KeywordItem]: 병합된 키워드 리스트
        """
        # 원본 키워드 + 확장 키워드
        merged_keywords = original_keywords.copy()
        merged_keywords.extend(clustering_result.expansion_keywords)
        
        # 중복 제거 (키워드 텍스트 기준)
        seen_keywords = set()
        unique_keywords = []
        
        for kw in merged_keywords:
            kw_lower = kw.keyword.lower()
            if kw_lower not in seen_keywords:
                seen_keywords.add(kw_lower)
                unique_keywords.append(kw)
        
        # 중요도 기준으로 정렬
        importance_order = {
            KeywordImportance.HIGH: 0,
            KeywordImportance.NORMAL: 1,
            KeywordImportance.LOW: 2
        }
        unique_keywords.sort(key=lambda x: (importance_order[x.importance], -x.confidence))
        
        return unique_keywords