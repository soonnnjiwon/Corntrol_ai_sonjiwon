import re
import numpy as np
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecordAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # [최적화] 라벨 임베딩은 서버 뜰 때 딱 한 번만 미리 구워둡니다.
        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미","네트워킹 & 관계","감정 & 일기 회고"]
        self.label_embeddings = self.model.encode(self.candidate_labels)

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    # [통합 매직 메서드] 문장 인코딩을 딱 1번만 수행합니다.
    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768 # 예외 처리
            
        # 1. 문장 임베딩 단 1회 생성!
        sentence_embedding = self.model.encode([cleaned_text])
        sentence_embedding_list = sentence_embedding[0].tolist()
        
        # 2. 주제 분류 로직 (미리 계산된 라벨 임베딩 활용)
        similarities = cosine_similarity(sentence_embedding, self.label_embeddings)[0]
        best_idx = np.argmax(similarities)
        topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
        
        # 3. 키워드 추출 로직
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