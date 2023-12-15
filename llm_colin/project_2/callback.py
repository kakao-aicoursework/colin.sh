import os
import json
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
ask_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=ChatPromptTemplate.from_template(
        template=read_prompt_template('./model/template/kakao_service_template.txt')
    ),
    output_key="answer",
    verbose=True,
)

def document_to_json(doc):
    return json.dumps(doc.__dict__).encode('utf8')

def callback_handler(request: ChatbotRequest) -> dict:
    conversation_id = request.userRequest.user.id
    history_file = load_conversation_history(conversation_id)

    question = request.userRequest.utterance

    docs = vector_repo.search(question, 2)
    doc_contents = []
    for doc in docs:
        doc_contents.append(document_to_json(doc))

    history = get_chat_history(conversation_id)
    answer = ask_chain.run(dict(documents=doc_contents,
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