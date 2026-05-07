import re
import numpy as np
import gc
from kiwipiepy import Kiwi
from fastembed import TextEmbedding

class RecordAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()
        # 1. 512MB의 구원자, FastEmbed (384차원 모델 사용)
        # 이 모델은 한국어 성능이 매우 뛰어나며, 메모리를 아주 적게 먹습니다.
        self.model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미", "네트워킹 & 관계", "감정 & 일기 회고"]
        
        # 2. 라벨 임베딩 미리 계산 (384 -> 768 패딩)
        raw_labels = list(self.model.embed(self.candidate_labels))
        self.label_embeddings = np.array([self._pad(emb) for emb in raw_labels])
        gc.collect()

    def _pad(self, vec):
        """384차원 뒤에 0을 채워 768차원 규격을 맞춥니다."""
        return np.pad(vec, (0, 384), 'constant')

    def _cosine_sim(self, a, b):
        """직접 계산하는 코사인 유사도 (메모리 절약)"""
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        return dot / (norm_a * norm_b.T + 1e-9)

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768
            
        # 3. 분석 수행
        # embed 함수는 iterator를 반환하므로 list로 변환해 사용합니다.
        raw_embedding = list(self.model.embed([cleaned_text]))[0]
        padded_embedding = self._pad(raw_embedding)
        
        # 유사도 계산
        similarities = self._cosine_sim(padded_embedding.reshape(1, -1), self.label_embeddings)[0]
        best_idx = np.argmax(similarities)
        topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
        
        # 키워드 로직
        analysis_result = self.kiwi.analyze(cleaned_text)
        candidates = list(set([t.form for t in analysis_result[0][0] if t.tag in ['NNG', 'NNP', 'SL']]))
        
        final_keywords = []
        if candidates:
            word_raws = list(self.model.embed(candidates))
            word_paddeds = np.array([self._pad(w) for w in word_raws])
            word_sims = self._cosine_sim(padded_embedding.reshape(1, -1), word_paddeds)[0]
            scored = sorted(zip(candidates, word_sims), key=lambda x: x[1], reverse=True)
            
            for word, score in scored:
                if score >= 0.3 or (score >= threshold and len(final_keywords) < 3):
                    final_keywords.append(word)
        
        gc.collect()
        return topic, final_keywords, padded_embedding.tolist()