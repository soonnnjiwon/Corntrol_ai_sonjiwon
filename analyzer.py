import re
import numpy as np
import torch
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecordAnalyzer:
    def __init__(self):
        torch.set_num_threads(1)
        self.kiwi = Kiwi()
        
        # 1. 512MB 무료 서버에서 가장 안정적인 모델 (메모리 사용량 약 150MB)
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.model = SentenceTransformer(self.model_name)
        
        # 주제 라벨 (768차원 규격을 맞추기 위해 미리 패딩 처리)
        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미","네트워킹 & 관계","감정 & 일기 회고"]
        raw_label_embeddings = self.model.encode(self.candidate_labels)
        self.label_embeddings = self._pad_to_768(raw_label_embeddings)

    def _pad_to_768(self, vector):
        """384차원 결과 뒤에 0을 채워 강제로 768차원 규격을 맞춥니다."""
        v = np.array(vector)
        if v.ndim == 1:
            return np.pad(v, (0, 384), 'constant').tolist()
        else:
            return np.pad(v, ((0, 0), (0, 384)), 'constant')

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768 # 768차원 규격 준수
            
        with torch.no_grad():
            # 분석은 정밀하게 384로, 결과는 768로 포장!
            raw_embedding = self.model.encode([cleaned_text])
            padded_embedding = self._pad_to_768(raw_embedding)
            sentence_embedding_list = padded_embedding[0].tolist()
            
            # 유사도 계산
            similarities = cosine_similarity(padded_embedding, self.label_embeddings)[0]
            best_idx = np.argmax(similarities)
            topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
            
            # 키워드 로직 (기존 유지)
            analysis_result = self.kiwi.analyze(cleaned_text)
            candidates = list(set([t.form for t in analysis_result[0][0] if t.tag in ['NNG', 'NNP', 'SL']]))
            
            final_keywords = []
            if candidates:
                word_raw_embeddings = self.model.encode(candidates)
                word_padded_embeddings = self._pad_to_768(word_raw_embeddings)
                word_sims = cosine_similarity(padded_embedding, word_padded_embeddings)[0]
                scored_candidates = sorted(zip(candidates, word_sims), key=lambda x: x[1], reverse=True)
                
                for word, score in scored_candidates:
                    if score >= 0.3 or (score >= threshold and len(final_keywords) < 3):
                        final_keywords.append(word)
                        
            return topic, final_keywords, sentence_embedding_list