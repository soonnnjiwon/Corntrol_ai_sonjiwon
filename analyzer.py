import re
import numpy as np
import torch
import gc  # 메모리 청소용
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecordAnalyzer:
    def __init__(self):
        # 1. 쓰레드 제한으로 메모리 폭주 방지
        torch.set_num_threads(1)
        
        self.kiwi = Kiwi()
        
        # 2. 768차원 모델 로드 (jhgan 모델보다 약간 더 최적화된 SBERT 사용)
        self.model_name = 'jhgan/ko-sbert-multitask'
        base_model = SentenceTransformer(self.model_name)
        
        # [핵심 최적화] 모델을 INT8 타입으로 양자화하여 메모리 사용량을 50% 이상 절감합니다.
        # 이 과정에서 768차원은 그대로 유지됩니다.
        self.model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # 사용하지 않는 베이스 모델은 즉시 메모리에서 제거
        del base_model
        gc.collect()

        # 3. 라벨 임베딩 (768차원 유지)
        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미","네트워킹 & 관계","감정 & 일기 회고"]
        self.label_embeddings = self.model.encode(self.candidate_labels)

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768 # 768차원 유지!
            
        # 추론 시에도 메모리 절약 모드 활성화
        with torch.no_grad():
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