import re
import numpy as np
import torch
import gc
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecordAnalyzer:
    def __init__(self):
        # 1. 메모리 폭주 방지
        torch.set_num_threads(1)
        self.kiwi = Kiwi()
        
        # 2. [필살기] 768차원이면서 몸집은 초경량인 ALBERT 모델 사용
        # 이 모델은 768차원 벡터를 생성하며, 메모리를 아주 적게 먹습니다.
        self.model_name = 'bongsoo/albert-small-kor-sbert-v1'
        self.model = SentenceTransformer(self.model_name)
        
        # 메모리 청소 (로딩 직후 불필요한 찌꺼기 제거)
        gc.collect()

        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미","네트워킹 & 관계","감정 & 일기 회고"]
        # 라벨 임베딩 (당연히 768차원입니다!)
        self.label_embeddings = self.model.encode(self.candidate_labels)

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768 # 규격 유지
            
        with torch.no_grad():
            # 3. 768차원 임베딩 생성 (지원님의 로직 그대로!)
            sentence_embedding = self.model.encode([cleaned_text])
            sentence_embedding_list = sentence_embedding[0].tolist()
            
            similarities = cosine_similarity(sentence_embedding, self.label_embeddings)[0]
            best_idx = np.argmax(similarities)
            topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
            
            # 키워드 추출 로직 유지
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