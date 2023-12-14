import os
import logging
import requests
from chroma import vector_repo
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from model.template.template_api import read_prompt_template
from model.chat_request import ChatbotRequest

os.environ["OPENAI_API_KEY"] = ""
vector_repo.init_db("chroma/kakao_reference/")

logger = logging.getLogger("Callback")
ask_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=ChatPromptTemplate.from_template(
        template=read_prompt_template('./model/template/kakao_service_template.txt')
    ),
    output_key="answer",
    verbose=True,
)

def callback_handler(request: ChatbotRequest) -> dict:
    question = request.userRequest.utterance

    docs = vector_repo.search(question, 2)
    doc_contents = []
    for doc in docs:
        doc_contents.append(doc.page_content)

    answer = ask_chain.run(dict(documents=docs,
                                user=question,
                                # chat_history=get_chat_history()
                                ))

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