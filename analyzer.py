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
        
        self.label_embeddings = np.array(self.query({"inputs": self.candidate_labels}))

    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()

    def _cosine_similarity(self, a, b):
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

    def _preprocess(self, text):
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return ' '.join(cleaned.split())

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768

        # 1. 외부 API를 통해 입력 문장의 768차원 임베딩 딱 한 번만 생성
        sentence_embedding = self.query({"inputs": cleaned_text})
        
        # API 응답 에러 처리 (에러 발생 시 0으로 채운 리스트 반환)
        if not isinstance(sentence_embedding, list):
            return "새로운 줄기", [], [0.0] * 768

        # 2. 로컬(Numpy)에서 주제 분류 계산 
        similarities = self._cosine_similarity(np.array(sentence_embedding), self.label_embeddings)
        best_idx = np.argmax(similarities)
        
        # 가장 높은 유사도가 문턱값(threshold)보다 높으면 해당 라벨 선택
        topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"

        # 3. 키워드 추출 
        analysis_result = self.kiwi.analyze(cleaned_text)
        keywords = list(set([t.form for t in analysis_result[0][0] if t.tag in ['NNG', 'NNP', 'SL']]))
        
        # 서영님과 약속한 규격: 주제, 키워드(최대 3개), 768차원 임베딩
        return topic, keywords[:3], sentence_embedding