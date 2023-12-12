from dto import ChatbotRequest
import time
import logging
import vector_repo
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
import os
import requests

os.environ["OPENAI_API_KEY"] = ""

logger = logging.getLogger("Callback")

chat = ChatOpenAI(temperature=0)
db = vector_repo.init_db("project_data_kakao_sync.txt")
system_message = "assistant는 카카오 서비스 제공자입니다. user의 내용을 참고하여 안내하라."

def callback_handler(request: ChatbotRequest) -> dict:

    docs = vector_repo.get_relevant_documents(db, 2, request.userRequest.utterance)
    doc_contents = []
    for doc in docs:
        doc_contents.append(doc.page_content)
    system_message_prompt = SystemMessage(content=system_message)

    human_template = ("가이드 문서: {doc_contents}\n"
                      "위 정보를 참조해서 아래 질문에 대한 답변을 만들어줘\n"
                      "---질문\n"
                      "{user}"
                      "---"
                      )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    llm_response = chain.run(doc_contents=doc_contents, user=request.userRequest.utterance)

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": llm_response
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        requests.post(url=url, json=payload, verify=False)