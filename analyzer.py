import re
import numpy as np
import torch
import gc
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer

class RecordAnalyzer:
    def __init__(self):
        # 1. 쓰레드 제한 및 메모리 비우기 설정
        torch.set_num_threads(1)
        self.kiwi = Kiwi()
        
        # 2. 512MB에서 절대 안 터지는 384차원 모델 (약 150MB 사용)
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.model = SentenceTransformer(self.model_name)
        
        # 3. 주제 라벨 (768차원 규격을 위해 미리 0으로 패딩)
        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미", "네트워킹 & 관계", "감정 & 일기 회고"]
        raw_label_embeddings = self.model.encode(self.candidate_labels)
        self.label_embeddings = self._pad_to_768(raw_label_embeddings)
        gc.collect()

    def _pad_to_768(self, vector):
        """384차원 뒤에 0을 채워 768차원 규격을 맞춤"""
        v = np.array(vector)
        if v.ndim == 1:
            return np.pad(v, (0, 384), 'constant')
        return np.pad(v, ((0, 0), (0, 384)), 'constant')

    def _cosine_sim(self, a, b):
        """Scikit-learn 없이 NumPy로 계산하여 메모리 절약"""
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        return dot / (norm_a * norm_b.T)

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768
            
        with torch.no_grad():
            # 분석은 정밀하게 384로, 규격은 768로!
            raw_embedding = self.model.encode([cleaned_text])
            padded_embedding = self._pad_to_768(raw_embedding)
            
            # 유사도 계산
            similarities = self._cosine_sim(padded_embedding, self.label_embeddings)[0]
            best_idx = np.argmax(similarities)
            topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
            
            # 키워드 로직
            analysis_result = self.kiwi.analyze(cleaned_text)
            candidates = list(set([t.form for t in analysis_result[0][0] if t.tag in ['NNG', 'NNP', 'SL']]))
            
            final_keywords = []
            if candidates:
                word_raw = self.model.encode(candidates)
                word_padded = self._pad_to_768(word_raw)
                word_sims = self._cosine_sim(padded_embedding, word_padded)[0]
                scored = sorted(zip(candidates, word_sims), key=lambda x: x[1], reverse=True)
                
                for word, score in scored:
                    if score >= 0.3 or (score >= threshold and len(final_keywords) < 3):
                        final_keywords.append(word)
            
            gc.collect()
            return topic, final_keywords, padded_embedding[0].tolist()