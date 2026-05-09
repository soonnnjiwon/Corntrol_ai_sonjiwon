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
        self.label_embeddings = None
        self._load_label_embeddings()

    def _load_label_embeddings(self):
        """서버 시작 시 카테고리 벡터를 미리 로드합니다."""
        res = self.query({"inputs": self.candidate_labels})
        if isinstance(res, list):
            self.label_embeddings = np.array(res)

    def query(self, payload):
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=15)
            return response.json()
        except:
            return None

    def _cosine_similarity(self, a, b):
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1) + 1e-9)

    def analyze_all(self, text, threshold=0.05):
        # 0. 전처리
        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text).strip()
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768

        # 1. 키워드 추출 (API와 상관없이 로컬에서 즉시 수행)
        # NNG(일반명사), NNP(고유명사), SL(외래어/영어) 추출
        tokens = self.kiwi.tokenize(cleaned_text)
        keywords = []
        for t in tokens:
            if t.tag in ['NNG', 'NNP', 'SL']:
                # '것', '수', '등' 같은 의미 없는 1글자 명사 제외 (영어/숫자는 유지)
                if len(t.form) > 1 or t.tag in ['NNP', 'SL']:
                    keywords.append(t.form)
        
        # 중복 제거 및 최대 3개 선택
        final_keywords = list(dict.fromkeys(keywords))[:3]

        # 2. 문장 임베딩 생성 (허깅페이스 API)
        res = self.query({"inputs": cleaned_text})
        
        # API 응답 실패 시 처리
        if not isinstance(res, list):
            return "새로운 줄기", final_keywords, [0.0] * 768
            
        sentence_embedding = res
        
        # 3. 주제 분류
        if self.label_embeddings is not None:
            similarities = self._cosine_similarity(np.array(sentence_embedding), self.label_embeddings)
            best_idx = np.argmax(similarities)
            topic = self.candidate_labels[best_idx] if similarities[best_idx] >= threshold else "새로운 줄기"
        else:
            topic = "새로운 줄기"
        
        return topic, final_keywords, sentence_embedding