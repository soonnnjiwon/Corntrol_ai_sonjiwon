import re
import numpy as np
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer

class RecordAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()
        
        # 768차원 유지를 위해 로컬 모델 직접 로드
        print("Loading Local AI Model (768-dim)...")
        self.model = SentenceTransformer('jhgan/ko-sbert-multitask')
        print("Local Model Loaded (Dimensions: 768)!")

        self.candidate_labels = [
            "아이디어 & 프로젝트", 
            "업무 & 학업", 
            "자기계발 & 루틴", 
            "경제 & 자산", 
            "일상 & 취미", 
            "네트워킹 & 관계", 
            "감정 & 일기 회고"
        ]
        
        # 카테고리 임베딩 미리 계산
        self.label_embeddings = self.model.encode(self.candidate_labels)

    def _cosine_similarity(self, a, b):
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1) + 1e-9)

    def analyze_all(self, text, threshold=0.15):
        # 0. 텍스트 전처리
        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text).strip()
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768

        # 1. 키워드 추출
        tokens = self.kiwi.tokenize(cleaned_text)
        keywords = [t.form for t in tokens if t.tag in ['NNG', 'NNP', 'SL'] and (len(t.form) > 1 or t.tag in ['NNP', 'SL'])]
        final_keywords = list(dict.fromkeys(keywords))[:3]

        # 2. 로컬 임베딩 계산 (768차원)
        sentence_embedding = self.model.encode(cleaned_text)
        
        # 3. 주제 분류
        similarities = self._cosine_similarity(sentence_embedding, self.label_embeddings)
        best_idx = np.argmax(similarities)
        
        topic = "새로운 줄기"
        if float(similarities[best_idx]) >= threshold:
            topic = self.candidate_labels[best_idx]
        
        return topic, final_keywords, sentence_embedding.tolist()