import re
import numpy as np
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer


class RecordAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()

        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        self.candidate_labels = [
            "아이디어 & 프로젝트",
            "업무 & 학업",
            "자기계발 & 루틴",
            "경제 & 자산",
            "일상 & 취미",
            "네트워킹 & 관계",
            "감정 & 일기 회고"
        ]

        self.label_embeddings = self.model.encode(
            self.candidate_labels,
            normalize_embeddings=True
        )

    def _cosine_similarity(self, a, b):
        a = np.asarray(a, dtype=float).reshape(1, -1)
        b = np.asarray(b, dtype=float)
        return (a @ b.T)[0]

    def analyze_all(self, text, threshold=0.15):
        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text).strip()

        if not cleaned_text:
            return "새로운 줄기", [], [0.0] * 768

        tokens = self.kiwi.tokenize(cleaned_text)

        keywords = [
            t.form for t in tokens
            if t.tag in ["NNG", "NNP", "SL"]
            and (len(t.form) > 1 or t.tag in ["NNP", "SL"])
        ]

        final_keywords = list(dict.fromkeys(keywords))[:3]

        sentence_embedding = self.model.encode(
            cleaned_text,
            normalize_embeddings=True
        )

        similarities = self._cosine_similarity(
            sentence_embedding,
            self.label_embeddings
        )

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        topic = "새로운 줄기"

        if best_score >= threshold:
            topic = self.candidate_labels[best_idx]

        return topic, final_keywords, sentence_embedding.tolist()