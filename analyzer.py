import re
import numpy as np
import torch
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecordAnalyzer:
    def __init__(self):
        # [메모리 사수 대작전] 512MB를 넘지 않기 위한 설정
        torch.set_num_threads(1) 
        
        self.kiwi = Kiwi()
        
        # 1. 768차원을 유지하는 한국어 특화 모델 (snunlp)
        # jhgan 모델보다 초기 로드 메모리 점유가 미세하게 낮아 시도해볼 가치가 있습니다.
        self.model_name = 'snunlp/KR-SBERT-V40K-klueNLI-aug'
        self.model = SentenceTransformer(self.model_name)
        
        # 2. 규격 유지: 768차원 라벨 임베딩
        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미","네트워킹 & 관계","감정 & 일기 회고"]
        self.label_embeddings = self.model.encode(self.candidate_labels)

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            # 768차원 유지!
            return "새로운 줄기", [], [0.0] * 768 
            
        # 지원님의 로직 100% 그대로 유지
        sentence_embedding = self.model.encode([cleaned_text])
        sentence_embedding_list = sentence_embedding[0].tolist()
        
        similarities = cosine_similarity(sentence_embedding, self.label_embeddings)[0]
        best_idx = np.argmax(similarities)
        topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
        
        analysis_result = self.kiwi.analyze(cleaned_text)
        candidates = list(set([t.form for t in analysis_result[0][0] if t.tag in ['NNG', 'NNP', 'SL']]))
        
        final_keywords = []
        if candidates:
            word_embeddings = self.model.encode(candidates)
            word_sims = cosine_similarity(sentence_embedding, word_embeddings)[0]
            scored_candidates = sorted(zip(candidates, word_sims), key=lambda x: x[1], reverse=True)
            
            for word, score in scored_candidates:
                if score >= 0.3 or (score >= threshold and len(final_keywords) < 3):
                    final_keywords.append(word)
                    
        return topic, final_keywords, sentence_embedding_list