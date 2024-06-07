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
from loadingModule import (loading_documents_and_create_db, loading_document_and_add_to_db, update_retriever)
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
    - You are one of the best assistant who carries out only the instructions given below, and nothing else. 
    - The question may be posed incorrectly, or be in a different language, try to understand what the user wants from you and what you would answer using his documents, so you can only use user materials.
    - You shouldn’t invent information or somehow try to connect the pieces of these paragraphs with each other, so if you can’t answer from the documents, then there is no answer to the question. You cannot use your internal knowledge to generate the answer and also quote them, since in our application we must generate the answer only based on user documents. Therefore, if you cannot provide a correct answer on user materials, offer the user the name of PDF files, after loading which into the database, we will be able to process user data.
    - If you can still answer the question using user materials, then make a short answer, just be sure to indicate on the basis of which user documents you made your answer. It is very important for us that the answer contains the resources on the basis of which you made the answer. Since this allows the user to quickly find a fragment that can help him with the answer.
    - User documents resources will be indicated in the form.

    ### User documents: 
    {context}

    ### Question:
    {question}


"""


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


PROMPT_TEMPLATE = """
    ### Instuction: 
    - Hello, you are an assistant in our application. The main goal of our application is to provide an answer based only on user materials. Our main challenge is to make sure that language models can not use their knowledge on which they were trained, but use only user documents. Our application is based on answering user request according to user submissions. Essentially, text is loaded into our application, and then we give you only those parts of the documents that we think will help answer the user’s question. It may also be that the documents that we provided to you may not contain an answer to the user request.
    - The user's request may be in a different language, or may not be written correctly or be complete. You need to clearly understand the meaning of the request.
    - If you cannot give a clear answer to the user's request, you need to inform the user that our application cannot process the user's request In no case do not use your internal knowledge, so you need to inform the user that the answer to his request cannot be generated and offer him options for documents when sent to the chat which may change.
    - If you still think that user documents can provide an answer to the user's request, then you need to generate a good answer using only user materials. It is also necessary to indicate the name of the resources that served as the impetus for this answer.

    ### User documents: 
    {context}

    ### Question:
    {question}


"""


PROMPT_TEMPLATE = """

QUESTION:
Enter the role of an assistant who has only the documents provided in front of you and try not to use your internal documents, but only to provide me with an clear and short answer using the text of the documents. Give me short answer! If you say that there is no answer in the documents and there really isn’t one, then that’s correct. If you provide me with an answer based on data that is not in the document, the answer is considered not relevant and I will be very sad that you did not clearly understand me. If you provide me with an answer to a question and there is no answer in the documents, this is considered not relevant answer, so tell me the name of the documents so that I can check it.
{question}


DOCUMENTS:
{context}

"""

# PROMPT_TEMPLATE = """
#     ### Instuction: 
#     - You are one of the best assistant who carries out only the instructions given below, and nothing else. 
#     - We provide a response to the user request according to the documents that were provided. It is very important for us that the answer is generated only on the basis of the materials provided. The request may be in another language, or composed incorrectly or incompletely. Try to understand what the user wants. The question may be posed incorrectly, or be in a *different language*, try to understand what the user wants from you and what you would answer using his documents, so you can only use user materials.
#     - You shouldn’t invent information or somehow try to connect the pieces of these paragraphs with each other, so if you can’t answer from the documents, then there is no answer to the question. You cannot use your internal knowledge to generate the answer and also quote them, since in our application we must generate the answer only based on user documents. Therefore, if you cannot provide a correct answer on user materials, offer the user the name of PDF files, after loading which into the database, we will be able to process user data.
#     - If you can still answer the question using user materials, please create a correct, clear and focused answer based on the files that the user provided that can actually help answer the question. When generating a question, indicate the resources of the files from which you took the material. Since the user is interested in which files were taken in order to answer the question. The user always needs to indicate resources, consider this as his request.
#     - User documents resources will be indicated in the form.

#     ### User documents: 
#     {context}

#     ### Question:
#     {question}


# """

# PROMPT_TEMPLATE = """
#     ### Instuction: 
#     - You are our most important part of the application in the generation part, so be responsible and strictly follow the instructions. 
#     - We provide a response to the user request according to the documents that were provided. It is very important for us that the answer is generated only on the basis of the materials provided. The request may be in another language, or composed incorrectly or incompletely. Try to understand what the user wants.
#     - If using documents it is not possible to compose a correct answer, then advise the user what documents he should download so that our application can give the correct answer. You cannot use your internal knowledge to generate the answer and also quote them, since in our application we must generate the answer only based on user documents. Therefore, if you cannot provide a correct answer on user materials, tell the user what PDF materials he can upload to chat so that you can correctly process his request.
#     - Using the material that was provided to you, try to give a complete, correct and clear answer to the request. It is very important for us that the answer contains the resources on the basis of which you made the answer. Since this allows the user to quickly find a fragment that can help him with the answer.
#     - User documents sources will be indicated in the form format like: Source - > file: [file_name], page: [page_number]

#     ### User documents: 
#     {context}

#     ### Question:
#     {question}


# """

sad_prompt = """
In general, analyze the question and decide whether it is just a welcome message, valid question or instructions related to chat history. If this is a welcome question, say hello to the user. If this message is a valid question, then the user entered an inappropriate question, tell the user that we cannot provide an answer to his question based on the documents we have. If this is an instruction that is related to the chat history, then try to use it to fulfill the user’s request.

Question: """

sad_prompt = """
We want to model your behavior. We need the following from you. Inform the user that according to his request, no documents were found in our application to answer his question. Therefore, invite him to upload his document in order to answer the request. Tell the user what documents he could use to make sure our application responded to his request. Our application only allows upload PDF documents. But don’t try to help the user yourself, this is incorrect behavior.
# Request: """


prompt_for_halucinacion = """
Hello, you are a very helpfull AI assistent. The question may not be asked properly or correctly, or in a different language, try to determine for yourself the main meaning of the question, so could you, without taking this into account, try to formulate two or three questions in English that would help you answer.

Question: """


prompt_for_halucinacion = """
Hello, in general, the user may have asked the question incorrectly, or you can process the request in another language and give an answer to it. 

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
    #faissdb = FAISS.load_local("sincak-db", embedding, allow_dangerous_deserialization=True) 
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

