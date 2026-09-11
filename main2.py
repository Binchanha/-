from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from tavily import TavilyClient
import os

# ==========================================
# 1. API 키 설정 (환경 변수에서 불러오기)
# ==========================================
import os # 이 줄이 파일 맨 위에 없다면 추가해 주세요.

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# 클라이언트 초기화
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ==========================================
# 2. FastAPI 앱 초기화
# ==========================================
app = FastAPI(title="정보 검증 API (Tavily 버전)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델 정의
class VerifyRequest(BaseModel):
    query: str

class VerifyResponse(BaseModel):
    status: str
    result_text: str

# ==========================================
# 3. 내부 검색 로직 (Tavily 사용)
# ==========================================
def search_web_with_tavily(query: str) -> str:
    """Tavily API를 사용해 웹을 검색하고 결과를 텍스트로 반환"""
    try:
        # Tavily 검색 실행 (최신 정보 포함, AI 요약용 설정)
        response = tavily_client.search(
        query, 
        search_depth="advanced", # basic을 advanced로 변경 (더 깊게 탐색)
        max_results=10,          # 5개에서 10개로 증가 (더 많은 문맥 확보)
        include_raw_content=True # (선택사항) 필요시 원본 텍스트를 더 많이 가져옴
    )
        
        results = response.get('results', [])
        if not results:
            return None
            
        context = ""
        for item in results:
            context += f"제목: {item.get('title')}\n내용: {item.get('content')}\n출처: {item.get('url')}\n\n"
            
        return context
    except Exception as e:
        print(f"Tavily 검색 오류 발생: {e}")
        return None

# ==========================================
# 4. API 엔드포인트
# ==========================================
@app.post("/api/verify", response_model=VerifyResponse)
async def verify_endpoint(request: VerifyRequest):
    try:
        # 1. Tavily 웹 검색
        print(f"검색 시작: {request.query}")
        search_context = search_web_with_tavily(request.query)
        
        if not search_context:
            return VerifyResponse(
                status="not_found", 
                result_text="현재 인터넷상에서 신뢰할 만한 관련 검색 결과를 찾을 수 없습니다."
            )

        # 2. 제미나이 검증 프롬프트 작성
        prompt = f"""
        당신은 최고 수준의 팩트 체크 및 정보 검증 전문가입니다. 
        제공된 [검색 결과]를 바탕으로 [사용자 질문]에 대한 답이 존재하는지 철저히 분석하세요.

        [사용자 질문]
        {request.query}

        [검색 결과]
        {search_context}

        **[분석 지침 - 반드시 지킬 것]**
        1. 단어가 완벽히 일치하지 않더라도, 문맥상 같은 의미를 내포하고 있다면 정보가 존재하는 것으로 간주합니다.
        2. 영어 논문 제목이나 외신 기사가 한국어로 번역되어 질문된 경우, 의미가 통하면 동일한 것으로 판단하세요.
        3. 검색 결과가 불충분하더라도 파편적인 정보들을 조합하여 답변의 실마리가 있다면 그 내용을 최대한 추출하여 요약하세요.

        다음 형식에 맞춰 명확하게 답변해주세요:
        - 정보 존재 여부: (예: 관련 정보가 확인됩니다 / 신뢰할 수 있는 정보를 찾을 수 없습니다)
        - 핵심 요약: (확인된 내용을 구체적이고 상세하게 3~4줄 요약)
        - 주요 출처: (확인된 정보의 링크 제공)
        """
        
        print("제미나이 분석 중...")
        # 3. 제미나이 호출
        ai_response = gemini_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        
        print("분석 완료!")
        return VerifyResponse(
            status="success",
            result_text=str(ai_response.text)
        )
        
    except Exception as e:
        print(f"서버 내부 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))