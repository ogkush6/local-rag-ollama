import os
import sys
import ollama
import argparse
from tqdm import tqdm
from typing import List
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
from loadingModule import (loading_documents_and_create_db, loading_document_and_add_to_db)
#set_verbose(True)
import time
import chainlit as cl 

from langchain_community.vectorstores import FAISS

from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
                  # RANKING:
MODEL = "mistral" # mistral, gemma, llama2 | command-r 20GB? je rovnaky ako mistral, no pomaly, nema zmysel..

from chainlit.input_widget import Slider
from ollamaModule import check_if_model_is_available


PROMPT_TEMPLATE = """
    ### Instuction: 
    - Give an answer based on the attached documents, if the documents do not contain an answer, just answer I don’t know
    - You shouldn’t invent information or somehow try to connect the pieces of these paragraphs with each other, so if you can’t answer from the documents, then there is no answer to the question.
    - In Research section the sources will be indicated in the form format like: Source - > file: [file_name], page: [page_number]

    ### User documents: 
    {context}

    ### Question:
    {question}


    """






    
PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
model = "ragmistral"
print(">> scanning for pdf documents")
print(">> embedding pdfs")
try:
    check_if_model_is_available(model)
except Exception as e:
    print(e)
    sys.exit()

# try:
embedding = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
            )
faissdb = FAISS.load_local("faiss-index", embedding, allow_dangerous_deserialization=True)
#     qddb, faissdb = loading_documents_and_create_db("../../../pdfs/slovakConst", "")
# except FileNotFoundError as e:
#     print(e)
#     sys.exit()
        
print(">> initializing LLM " + MODEL)
llm = Ollama(
        model=model,
#         callbacks=[StreamingStdOutCallbackHandler()],
#         streaming=True
    )

# from langchain.schema.runnable.config import RunnableConfig
# cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
# config = RunnableConfig(callbacks=[cb])
# result = agent.invoke(input, config=config)

    
# doc_type = 5
# qddb_retriever = qddb.as_retriever(search_type="mmr", search_kwargs={'k':doc_type, 'score_threshold':0.81})
# faiss_retriever = faissdb.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': doc_type, 'score_threshold':0.77})
# qa_chain_faiss = RetrievalQA.from_chain_type(
#         llm,
#         chain_type="stuff",
#         retriever = faiss_retriever,
#         chain_type_kwargs={"prompt": PROMPT},
#         verbose=True,
        
# #         callbacks=[cl.LangchainCallbackHandler()],
# #         output_parser=StrOutputParser()
#         )


@cl.on_chat_start
async def on_chat_start(): 
    

#     files = None

    # Wait for the user to upload a file
#     while files == None:
#         files = await cl.AskFileMessage(
#             content="Please upload a text file to begin!", accept=["application/pdf"]
#         ).send()

#     text_file = files[0]
    
#     faiss_db = loading_document_and_add_to_db(text_file.path, embedding, faissdb)
    faiss_retriever = faissdb.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 5, 'score_threshold':0.77})
    qa_chain_faiss = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever = faiss_retriever,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True,
        
#         callbacks=[cl.LangchainCallbackHandler()],
#         output_parser=StrOutputParser()
        )
    # Let the user know that the system is ready
    cl.user_session.set("qa_chain", qa_chain_faiss)
    cl.user_session.set("faiss", faiss_retriever)
#     settings = await cl.ChatSettings(
#         [
#             Slider(
#                 id="Temperature",
#                 label="OpenAI - Temperature",
#                 initial=1,
#                 min=0,
#                 max=2,
#                 step=0.1,
#             ),
#         ]
#     ).send()
#     value = settings["Temperature"]


    

@cl.on_message
async def on_message(msg: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    faiss = cl.user_session.get("faiss")
    async with cl.Step(name="Retriver documents", root=True) as step:
#         stOut = ""
        for idx, i in enumerate(faiss.invoke(msg.content)):
            async with cl.Step(name="Document {}".format(idx+1)) as child_step:
#             some_step = cl.Step()
#             some_step.output = i.page_content
                child_step.output= i.page_content
#             stOut.append(some_step)
#         step.output = stOut
   
#     async with cl.Step(name="Parent step") as parent_step:
#         parent_step.input = "Parent step input"

#         async with cl.Step(name="Child step") as child_step:
#             child_step.input = "Child step input"
#             child_step.output = "Child step output"
#         async with cl.Step(name="Child step") as child_step:
#             child_step.input = "Child step input"
#             child_step.output = "Child step output"
#         parent_step.output = "Parent step output"
    result = await cl.make_async(qa_chain.invoke)({"query": msg.content})
    
#     await cl.sleep(100)
    #                                                   config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()]))
#                                                  config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]))
#     result = await cl.make_async(qa_chain.stream)(msg.content)
    
        # Step is sent as soon as the context manager is entered
        
        
    print(msg.elements)
    msge = cl.Message(content="")
    print(result['result'])
    for i in result['result']: 
        await msge.stream_token(i)
    await msge.send()
#     await cl.Message(content=result).send()
#     msg = cl.Message(content="")

#     async for chunk in qa_chain.astream(
#         {"query": message.content},
#         config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()]),
#     ):
#         print(chunk)
#         await msg.stream_token(chunk['result'])

#     await msg.send()
