import os
import logging
import requests
from chroma import vector_repo
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.chat_models import ChatOpenAI

from history import load_conversation_history, get_chat_history, log_user_message, log_bot_message
from model.template.template_api import read_prompt_template
from model.chat_request import ChatbotRequest

os.environ["OPENAI_API_KEY"] = ""

logger = logging.getLogger("Callback")

keyword_extractor_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=ChatPromptTemplate.from_template(
        template=read_prompt_template('./model/template/keyword_extractor_template.txt')
    ),
    output_key="keywords",
    verbose=True,
)

ask_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=ChatPromptTemplate.from_template(
        template=read_prompt_template('./model/template/kakao_service_template.txt')
    ),
    output_key="answer",
    verbose=True,
)

def callback_handler(request: ChatbotRequest) -> dict:
    conversation_id = request.userRequest.user.id
    history_file = load_conversation_history(conversation_id)

    history = get_chat_history(conversation_id)
    question = request.userRequest.utterance
    keywords = keyword_extractor_chain.run(dict(question=question, chat_histories=history))

    docs = vector_repo.search(keywords, 2)

    # If no relevant reference document is found, return "I don't know."
    if not docs:
        answer = "죄송합니다, 그 질문에 대해선 답변해드릴 수 없습니다."
    else:
        answer = ask_chain.run(dict(documents=docs,
                                    user=question,
                                    chat_histories=history
                                    ))

    log_user_message(history_file, question)
    log_bot_message(history_file, answer)

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }

    url = request.userRequest.callbackUrl
    if url:
        requests.post(url=url, json=payload, verify=False)