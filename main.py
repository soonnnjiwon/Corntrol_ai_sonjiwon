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
        
        # 서버 시작 시 라벨 임베딩 로드 시도
        self._load_label_embeddings()

    def _load_label_embeddings(self):
        print("--- [시스템] 라벨 임베딩 로드 시도 중... ---")
        res = self.query({"inputs": self.candidate_labels})
        if isinstance(res, list) and len(res) > 0:
            self.label_embeddings = np.array(res)
            print(f"--- [성공] 라벨 임베딩 로드 완료 (Shape: {self.label_embeddings.shape}) ---")
        else:
            print(f"--- [실패] 라벨 임베딩 로드 실패. 응답: {res} ---")

    def query(self, payload):
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=20)
            return response.json()
        except Exception as e:
            print(f"--- [에러] API 호출 중 예외 발생: {e} ---")
            return None

    def _cosine_similarity(self, a, b):
        # 분모가 0이 되는 것을 방지하기 위해 1e-9 더함
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1) + 1e-9)

    def analyze_all(self, text, threshold=0.01): # 테스트를 위해 threshold를 더 낮춤
        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text).strip()
        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768

        # 1. 문장 임베딩 생성
        res = self.query({"inputs": cleaned_text})
        
        # 진단 로그 출력
        if not isinstance(res, list):
            print(f"--- [경고] 문장 임베딩 실패. API 응답: {res} ---")
            return "새로운 줄기", [], [0.0] * 768
            
        sentence_embedding = res
        
        # 2. 주제 분류
        if self.label_embeddings is None:
            print("--- [경고] 라벨 임베딩이 없어 다시 로드를 시도합니다... ---")
            self._load_label_embeddings()
            
        if self.label_embeddings is not None:
            similarities = self._cosine_similarity(np.array(sentence_embedding), self.label_embeddings)
            best_idx = np.argmax(similarities)
            max_score = similarities[best_idx]
            
            # 🔥 핵심 진단 로그: 점수가 얼마인지 로그에 찍힙니다!
            print(f"--- [분석 결과] 문장: '{cleaned_text[:10]}...' / 최고 점수: {max_score:.4f} / 카테고리: {self.candidate_labels[best_idx]} ---")
            
            topic = self.candidate_labels[best_idx] if max_score >= threshold else "새로운 줄기"
        else:
            topic = "새로운 줄기"

        # 3. 키워드 추출
        analysis_result = self.kiwi.analyze(cleaned_text)
        keywords = list(set([t.form for t in analysis_result[0][0] if t.tag in ['NNG', 'NNP', 'SL']]))
        
        return topic, keywords[:3], sentence_embedding