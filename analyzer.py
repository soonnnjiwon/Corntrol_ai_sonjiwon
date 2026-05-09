import re
import os
import requests
import numpy as np
from kiwipiepy import Kiwi

class RecordAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()
        self.hf_token = os.environ.get("HF_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/models/jhgan/ko-sbert-multitask"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

        self.candidate_labels = ["아이디어 & 프로젝트", "업무 & 학업", "자기계발 & 루틴", "경제 & 자산", "일상 & 취미", "네트워킹 & 관계", "감정 & 일기 회고"]
        
        # 서버 시작 시 라벨 임베딩을 한 번 받아옵니다. 
        # (만약 여기서 에러가 나면 아래 analyze_all에서 처리하도록 예외 처리를 추가했습니다)
        try:
            res = self.query({"inputs": self.candidate_labels})
            if isinstance(res, list):
                self.label_embeddings = np.array(res)
            else:
                self.label_embeddings = None
        except:
            self.label_embeddings = None

    def query(self, payload):
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=10)
            return response.json()
        except Exception as e:
            print(f"API Query Error: {e}")
            return None

    def _cosine_similarity(self, a, b):
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1) + 1e-9)

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768

        # 1. 문장 임베딩 생성
        res = self.query({"inputs": cleaned_text})
        if not isinstance(res, list):
            return "새로운 줄기", [], [0.0] * 768
            
        sentence_embedding = res
        
        # 2. 주제 분류
        if self.label_embeddings is not None:
            similarities = self._cosine_similarity(np.array(sentence_embedding), self.label_embeddings)
            best_idx = np.argmax(similarities)
            topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
        else:
            topic = "새로운 줄기"

        # 3. 키워드 추출
        analysis_result = self.kiwi.analyze(cleaned_text)
        keywords = list(set([t.form for t in analysis_result[0][0] if t.tag in ['NNG', 'NNP', 'SL']]))
        
        return topic, keywords[:3], sentence_embedding