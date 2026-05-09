from fastapi import FastAPI
from pydantic import BaseModel
from analyzer import RecordAnalyzer

app = FastAPI()

analyzer = RecordAnalyzer()


class AnalysisRequest(BaseModel):
    recordId: int
    userId: str
    content: str


@app.post("/analysis")
def create_analysis(request: AnalysisRequest):
    topic, keywords, embedding = analyzer.analyze_all(request.content)

    return {
        "recordId": request.recordId,
        "userId": request.userId,
        "content": request.content,
        "topic": topic,
        "keywords": keywords,
        "embedding": embedding
    }

