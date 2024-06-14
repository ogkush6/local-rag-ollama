import os
import sys
import ollama
import argparse
from tqdm import tqdm
from typing import List
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    PyMuPDFLoader
)
from langchain_experimental.text_splitter import SemanticChunker 
from langchain.globals import set_verbose
from loadingModule import (loading_document_and_add_to_db, update_retriever)
#set_verbose(True)
import time
import chainlit as cl 

from langchain_community.vectorstores import FAISS

from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
                  # RANKING:

from chainlit.input_widget import Slider
from ollamaModule import check_if_model_is_available
from ollama import chat


PROMPT_TEMPLATE = """
    ### Instuction: 
    - You help us generate the response of our application. The main point of our application is to answer the question using only materials provided by the user. You cannot generate text that is not in the provided user documents. We need your generated text to be found in the documents.
    - The user's request may be in a different language, or may not be written correctly or be complete. You need to clearly understand the meaning of the request.
    - If the user data does not contain a clear answer to the user's request, then you need to answer that according to the user documents, our application cannot provide an answer and not generate a response based not on user documents. Therefore, you need to offer the user, when downloading which PDF documents, can help when downloading into the chat, which will make these materials accessible, and answer the question.
    - If you can give a clear answer to the user's request, based on these documents, without using third-party sources, provide this answer to the request. The answer must contain clear references to the names of the documents on the basis of which you made the answer, so that the user can see which documents may contain answers to his questions.

    ### User documents: 
    {context}

    ### Question:
    {question}


"""



sad_prompt = """
We want to model your behavior. We need the following from you. Inform the user that according to his request, no documents were found in our application to answer his question. Therefore, invite him to upload his document in order to answer the request. Tell the user what documents he could use to make sure our application responded to his request. Our application only allows upload PDF documents. But don’t try to help the user yourself, this is incorrect behavior.
# Request: """


prompt_for_halucinacion = """
Hello, you are a very helpfull AI assistent. The question may not be asked properly or correctly, or in a different language, try to determine for yourself the main meaning of the question, so could you, without taking this into account, try to formulate answer in English.

Question: """


    
prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
model = "mistral"
print(">> scanning for pdf documents")
print(">> embedding pdfs")
try:
    check_if_model_is_available(model)
except Exception as e:
    print(e)
    sys.exit()

embedding = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
            )
        
print(">>> initializing llm " + model)
llm = Ollama(
        model=model,
    )



@cl.on_chat_start
async def on_chat_start(): 
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="full_db", value="full-db", label="🎒 Full-db"),
            cl.Action(name="sincak_db", value="sincak-db", label="⏰Prof. Sincak DB"),
            cl.Action(name="empty_db", value="empty-db", label="Empty db"),
        ],
    ).send()

    if res.get("value") == "empty-db":
        faissdb = FAISS.from_texts(["FAISS is an important library", "LangChain supports FAISS"], embedding)
    else:
        faissdb = FAISS.load_local(res.get("value"), embedding, allow_dangerous_deserialization=True)
        
        

        await cl.Message(
            content="You select "+res.get("value")+", enjoy",
        ).send()
    settings = await cl.ChatSettings(
        [
            Slider(
                id="retriever_k", 
                label="Number of find pieces of Document", 
                initial=5, 
                min=1, 
                max=20, 
                step=1
                ),
            Slider(
                id="retriever_score", 
                label="How similar should the pieces be?", 
                initial=0.77, 
                min=0.2, 
                max=0.8, 
                step=0.01
                ),
        ]
            ).send() 
    faiss_retriever = faissdb.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 5, 'score_threshold':0.77})
    qa_chain_faiss = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever = faiss_retriever,
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
        )
    cl.user_session.set("qa_chain", qa_chain_faiss)
    cl.user_session.set("faiss_retriever", faiss_retriever)
    cl.user_session.set("faissdb", faissdb)
    cl.user_session.set("settings", settings)
    cl.user_session.set("last_number_retriever", (settings["retriever_k"], settings["retriever_score"]))
    cl.user_session.set("messages_history", [])
    cl.user_session.set("stop_message", False)

    

@cl.on_message
async def on_message(msg: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    faiss_retriever = cl.user_session.get("faiss_retriever")
        
    
    settings = cl.user_session.get("settings")    
    faissdb = cl.user_session.get("faissdb")
    
    if len(msg.elements) != 0: 
        for item in msg.elements: 
            if type(item) == cl.File:
                faiss_db = loading_document_and_add_to_db(item.path, embedding, faissdb, item.name)
                faissdb = faiss_db
        faiss_retriever = faissdb.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': int(settings["retriever_k"]), 'score_threshold':settings["retriever_score"]})
        qa_chain = update_retriever(llm, faiss_retriever, prompt)
        cl.user_session.set("qa_chain", qa_chain)
        cl.user_session.set("faiss_retrieval", faiss_retriever)


    elif (settings["retriever_k"], settings["retriever_score"]) != cl.user_session.get("last_number_retriever"): 
        faiss_retriever = faissdb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":int(settings["retriever_k"]), "score_threshold":settings["retriever_score"]})
        qa_chain = update_retriever(llm, faiss_retriever, prompt)
        cl.user_session.set("qa_chain", qa_chain)
        cl.user_session.set("faiss_retrieval", faiss_retriever)
        



        
    messages = cl.user_session.get("messages_history")
    print(messages)
    message = {'role': 'assistant', 'content': ''}
    
    
    docs = faiss_retriever.invoke(msg.content)
    have_relevent = len(docs) != 0
    if have_relevent:
        messages.append({'role': 'user', 'content': ">>> " + msg.content})
        async with cl.Step(name="Retriver documents", root=True) as step:
            for idx, i in enumerate(docs):
                async with cl.Step(name="Document {}".format(idx+1)) as child_step:
                    child_step.output= i.page_content
        result = await cl.make_async(qa_chain.invoke)({"query": msg.content})
        result = result['result']
        print("result from qachain")
    else:
        messages.append({'role': 'user', 'content': prompt_for_halucinacion + msg.content})
        async with cl.Step(name="Retriver documents", root=True) as step:
            pass
        result = await cl.make_async(chat)(model='mistral', messages=messages)
        halucinacion = result['message']['content']
        docs = faiss_retriever.invoke(halucinacion)
        print("haluciantion")
        print(halucinacion)
        have_relevent = len(docs) != 0
        if have_relevent:
            del messages[-1]
            messages.append({'role': 'user', 'content': ">>> " + msg.content})
            async with cl.Step(name="Retriver documents", root=True) as step:
                for idx, i in enumerate(docs):
                    async with cl.Step(name="Document {}".format(idx+1)) as child_step:
                        child_step.output= i.page_content
            result = await cl.make_async(qa_chain.invoke)({"query": msg.content})
            result = result['result']
        else: 
            #del messages[-1]
            #messages.append({'role': 'user', 'content': sell + msg.content})
            print("sad answer")
            messages[-1]["content"] = sad_prompt + msg.content
            result = await cl.make_async(chat)(model='mistral', messages=[messages[-1]])
            result = result["message"]["content"]
        print("result from ollama")
                
                
                
    message['content'] += result
    messages.append(message)
    msge = cl.Message(content="")
    print(result)
    for i in result: 
        if cl.user_session.get("stop_message"): 
            cl.user_session.set("stop_message", False)
            break
            
        await msge.stream_token(i)
    await msge.send()



# @cl.on_stop
# def on_stop():
#     print("stopped")
#     cl.user_session.set("stop_message", True)

