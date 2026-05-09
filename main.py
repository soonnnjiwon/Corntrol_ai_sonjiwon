from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
from analyzer import RecordAnalyzer

# 1. FastAPI 앱 생성
app = FastAPI()

# 2. 분석기 초기화
analyzer = RecordAnalyzer()

# 3. 데이터 규격 정의 (입력)
class RecordRequest(BaseModel):
    recordId: int
    userId: str
    content: str

@app.get("/")
def read_root():
    return {"status": "running", "format": "snake_case"}

@app.post("/analysis")
async def analyze_record(request: RecordRequest):
    try:
        # 분석 실행
        topic, keywords, embedding = await run_in_threadpool(
            analyzer.analyze_all, request.content
        )
        
        return {
            "recordId": request.recordId,
            "userId": request.userId,
            "content": request.content,
            "topic": topic,
            "keywords": keywords,
            "embedding": embedding
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))