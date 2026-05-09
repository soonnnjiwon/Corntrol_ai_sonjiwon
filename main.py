from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from analyzer import RecordAnalyzer

# 1. FastAPI 앱 인스턴스 생성 (Render가 찾는 'app' 변수)
app = FastAPI()

# 2. 분석기 초기화
analyzer = RecordAnalyzer()

# 3. 요청 데이터 규격 정의 (서영님 백엔드와 맞춘 규격)
class RecordRequest(BaseModel):
    recordId: int      # 숫자형 필수
    userId: str        # 문자열 필수
    content: str       # 분석할 내용

@app.get("/")
def read_root():
    return {"message": "Corn-trol AI Server is Running!"}

@app.post("/analysis")
async def analyze_record(request: RecordRequest):
    try:
        # analyzer.py의 analyze_all 함수 호출
        # 카테고리 분류가 더 잘 되도록 threshold를 0.05로 설정했습니다.
        topic, keywords, embedding = analyzer.analyze_all(request.content, threshold=0.05)
        
        # 분석 결과 반환
        return {
            "recordId": request.recordId,
            "topic": topic,
            "keywords": keywords,
            "sentence_embedding": embedding
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))