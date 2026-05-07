import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

from analyzer import RecordAnalyzer  # 기존에 튜닝한 엔진 클래스

app = FastAPI(title="Corn-trol AI Analysis API")

# --- 외부 접속 및 메인 백엔드 서버와의 통신을 허용하는 CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI 엔진 초기화 (모델 로드 및 라벨 인코딩)
analyzer = RecordAnalyzer()


# 1. 사용자로부터 받는 요청 모델
class AnalyzeRequest(BaseModel):
    recordId: int            
    userId: Union[str, int]  
    content: str             

# 2. 분석 후 반환되는 응답 모델 
class AnalyzeResponse(BaseModel):
    recordId: int            
    userId: Union[str, int]  
    content: str
    topic: str                
    keywords: List[str]       
    embedding: List[float]    

# [POST] 기록 분석 요청 및 결과 반환
@app.post(
    "/analysis",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_analysis(request: AnalyzeRequest):
    try:
        # AI 모델을 통해 토픽, 키워드, 임베딩 '계산'
        topic, keywords, embedding = analyzer.analyze_all(request.content)

        # 즉시 백엔드로 리턴합니다.
        return {
            "recordId": request.recordId,
            "userId": request.userId,
            "content": request.content,
            "topic": topic,
            "keywords": keywords,
            "embedding": embedding,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI 분석 중 서버 에러 발생: {str(e)}",
        )

# --- 실행 설정 ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)