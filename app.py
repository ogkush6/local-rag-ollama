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
from langchain_community.embeddings import OllamaEmbeddings
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
We want to model your behavior. We need the following from you. Inform the user that according to his request, no documents were found in our application to answer his question. Therefore, invite him to upload his document in order to answer the request. Tell the user what documents he could use to make sure our application responded to his request. Our application only allows upload PDF documents. But don't try to help the user yourself, this is incorrect behavior.
# Request: """


prompt_for_halucinacion = """
Hello, you are a very helpfull AI assistent. The question may not be asked properly or correctly, or in a different language, try to determine for yourself the main meaning of the question, so could you, without taking this into account, try to formulate answer in English.

Question: """


    
prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
model = "deepseek-r1:7b"
print(">> scanning for pdf documents")
print(">> embedding pdfs")
try:
    check_if_model_is_available(model)
except Exception as e:
    print(e)
    sys.exit()

embedding = OllamaEmbeddings(
    model=model,
)
        
print(">>> initializing llm " + model)
llm = Ollama(
    model=model,
)



@cl.on_chat_start
async def on_chat_start(): 
    try:
        # Present options to the user
        res = await cl.AskActionMessage(
            content="Pick an action!",
            actions=[
                cl.Action(name="full_db", value="full-db", label="🎒 Full-db"),
                cl.Action(name="empty_db", value="empty-db", label="Empty db"),
            ],
        ).send()

        # Initialize with fallback values
        selected_option = "empty-db"
        if res is not None and res.get("value") is not None:
            selected_option = res.get("value")
            print(f"User selected: {selected_option}")
        else:
            print("No selection received, defaulting to empty-db")
            
        # Check for model availability
        try:
            check_if_model_is_available(model)
            print(f"Model {model} is available")
        except Exception as e:
            error_msg = f"Error checking model availability: {str(e)}"
            print(error_msg)
            await cl.Message(
                content=f"⚠️ {error_msg}. Using default model settings.",
            ).send()

        # Create and test embedding model
        try:
            test_embedding = embedding.embed_query("Test embedding for dimension check")
            embedding_dim = len(test_embedding)
            print(f"Embedding dimension: {embedding_dim}")
        except Exception as e:
            error_msg = f"Error testing embedding model: {str(e)}"
            print(error_msg)
            await cl.Message(
                content=f"⚠️ {error_msg}. Using default embeddings.",
            ).send()

        # Create a new empty database with the current embedding model
        print("Creating a new FAISS database...")
        try:
            faissdb = FAISS.from_texts(["FAISS initialization test", "LangChain supports FAISS"], embedding)
            print("FAISS database created successfully")
        except Exception as e:
            error_msg = f"Error creating FAISS database: {str(e)}"
            print(error_msg)
            await cl.Message(
                content=f"⚠️ {error_msg}. Please try refreshing.",
            ).send()
            raise
        
        # Inform the user about the database selection
        if selected_option != "empty-db":
            await cl.Message(
                content=f"You selected {selected_option}, but we're using a fresh database for compatibility.",
            ).send()
        else:
            await cl.Message(
                content="You selected empty-db.",
            ).send()
            
        # Ask for user settings
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
                    initial=0.1,  # Lower initial value for OllamaEmbeddings
                    min=0.01,     # Much lower minimum to handle negative scores
                    max=0.8, 
                    step=0.01
                    ),
            ]
        ).send() 
                
        # Use a lower score threshold to accommodate potential negative scores from OllamaEmbeddings
        score_threshold = 0.1  # Much lower threshold to handle negative scores
        retriever_k = 5
        
        # Get settings from user if available
        if settings is not None:
            if "retriever_k" in settings:
                retriever_k = int(settings["retriever_k"])
            if "retriever_score" in settings:
                score_threshold = float(settings["retriever_score"])
        
        print(f"Using retriever settings: k={retriever_k}, score_threshold={score_threshold}")
        
        # Create retriever with the settings
        try:
            faiss_retriever = faissdb.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={'k': retriever_k, 'score_threshold': score_threshold}
            )
            print("FAISS retriever created successfully")
        except Exception as e:
            error_msg = f"Error creating FAISS retriever: {str(e)}"
            print(error_msg)
            await cl.Message(
                content=f"⚠️ {error_msg}. Using default retriever settings.",
            ).send()
            # Fallback to simpler retriever
            faiss_retriever = faissdb.as_retriever(search_kwargs={"k": retriever_k})
        
        # Create QA chain
        try:
            qa_chain_faiss = RetrievalQA.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=faiss_retriever,
                chain_type_kwargs={"prompt": prompt},
                verbose=True,
            )
            print("QA chain created successfully")
        except Exception as e:
            error_msg = f"Error creating QA chain: {str(e)}"
            print(error_msg)
            await cl.Message(
                content=f"⚠️ {error_msg}. Please try refreshing.",
            ).send()
            raise
            
        # Store session variables
        cl.user_session.set("qa_chain", qa_chain_faiss)
        cl.user_session.set("faiss_retriever", faiss_retriever)
        cl.user_session.set("faissdb", faissdb)
        cl.user_session.set("settings", settings)
        cl.user_session.set("last_number_retriever", (retriever_k, score_threshold))
        cl.user_session.set("messages_history", [])
        cl.user_session.set("stop_message", False)
        
        print("Chat session initialized successfully")
        
    except Exception as e:
        error_msg = f"Error in on_chat_start: {str(e)}"
        print(error_msg)
        await cl.Message(
            content=f"An error occurred while starting the chat: {str(e)}. Please try refreshing the page.",
        ).send()



@cl.on_message
async def on_message(msg: cl.Message):
    try:
        qa_chain = cl.user_session.get("qa_chain")
        faiss_retriever = cl.user_session.get("faiss_retriever")
        settings = cl.user_session.get("settings")    
        faissdb = cl.user_session.get("faissdb")
        
        if None in [qa_chain, faiss_retriever, faissdb]:
            await cl.Message(
                content="Session data is missing. Please refresh the page and try again.",
            ).send()
            return
        
        # Handle file uploads
        if len(msg.elements) != 0: 
            try:
                for item in msg.elements: 
                    if type(item) == cl.File:
                        # Install pymupdf if needed
                        try:
                            faiss_db = loading_document_and_add_to_db(item.path, embedding, faissdb, item.name)
                            faissdb = faiss_db
                        except ImportError:
                            await cl.Message(
                                content="Error: PyMuPDF is not installed. Please install it with 'pip install pymupdf'.",
                            ).send()
                            return
                
                # Get updated settings
                retriever_k = 5
                score_threshold = 0.1
                
                if settings is not None:
                    if "retriever_k" in settings:
                        retriever_k = int(settings["retriever_k"])
                    if "retriever_score" in settings:
                        score_threshold = settings["retriever_score"]
                
                faiss_retriever = faissdb.as_retriever(
                    search_type="similarity_score_threshold", 
                    search_kwargs={'k': retriever_k, 'score_threshold': score_threshold}
                )
                qa_chain = update_retriever(llm, faiss_retriever, prompt)
                cl.user_session.set("qa_chain", qa_chain)
                cl.user_session.set("faiss_retriever", faiss_retriever)
            except Exception as e:
                await cl.Message(
                    content=f"Error processing your document: {str(e)}",
                ).send()
                return

        # Handle settings changes
        elif settings is not None and "retriever_k" in settings and "retriever_score" in settings:
            last_number_retriever = cl.user_session.get("last_number_retriever")
            current_settings = (int(settings["retriever_k"]), settings["retriever_score"])
            
            if last_number_retriever != current_settings:
                try:
                    faiss_retriever = faissdb.as_retriever(
                        search_type="similarity_score_threshold", 
                        search_kwargs={'k': current_settings[0], 'score_threshold': current_settings[1]}
                    )
                    qa_chain = update_retriever(llm, faiss_retriever, prompt)
                    cl.user_session.set("qa_chain", qa_chain)
                    cl.user_session.set("faiss_retriever", faiss_retriever)
                    cl.user_session.set("last_number_retriever", current_settings)
                except Exception as e:
                    await cl.Message(
                        content=f"Error updating settings: {str(e)}",
                    ).send()
                    return

        # Process the message and generate a response
        messages = cl.user_session.get("messages_history") or []
        message = {'role': 'assistant', 'content': ''}
        
        try:
            docs = faiss_retriever.invoke(msg.content)
            have_relevant = len(docs) > 0
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            have_relevant = False
            docs = []
        
        if have_relevant:
            messages.append({'role': 'user', 'content': ">>> " + msg.content})
            try:
                async with cl.Step(name="Retrieved documents", root=True) as step:
                    for idx, i in enumerate(docs):
                        async with cl.Step(name=f"Document {idx+1}") as child_step:
                            child_step.output = i.page_content
                result = await cl.make_async(qa_chain.invoke)({"query": msg.content})
                result = result['result']
                print("Result from qachain")
            except Exception as e:
                await cl.Message(
                    content=f"Error processing your query: {str(e)}",
                ).send()
                return
        else:
            try:
                messages.append({'role': 'user', 'content': prompt_for_halucinacion + msg.content})
                async with cl.Step(name="Retrieved documents", root=True) as step:
                    pass
                result = await cl.make_async(chat)(model=model, messages=messages)
                if result and "message" in result and "content" in result["message"]:
                    hallucination = result["message"]["content"]
                    print("Hallucination text:")
                    print(hallucination)
                    
                    # Try searching with hallucination text
                    try:
                        docs = faiss_retriever.invoke(hallucination)
                        have_relevant = len(docs) > 0
                    except Exception as e:
                        print(f"Error retrieving documents with hallucination: {str(e)}")
                        have_relevant = False
                    
                    if have_relevant:
                        del messages[-1]
                        messages.append({'role': 'user', 'content': ">>> " + msg.content})
                        async with cl.Step(name="Retrieved documents", root=True) as step:
                            for idx, i in enumerate(docs):
                                async with cl.Step(name=f"Document {idx+1}") as child_step:
                                    child_step.output = i.page_content
                        result = await cl.make_async(qa_chain.invoke)({"query": msg.content})
                        result = result['result']
                    else: 
                        print("No relevant documents found, using sad answer")
                        messages[-1]["content"] = sad_prompt + msg.content
                        result = await cl.make_async(chat)(model=model, messages=[messages[-1]])
                        if result and "message" in result and "content" in result["message"]:
                            result = result["message"]["content"]
                        else:
                            result = "I'm sorry, I couldn't find any relevant documents to answer your question. Please upload PDF documents that might contain the information you're looking for."
                else:
                    result = "I'm sorry, I couldn't process your request. Please try again."
            except Exception as e:
                await cl.Message(
                    content=f"Error processing your query: {str(e)}",
                ).send()
                return
                
        # Send the response to the user
        message['content'] += result
        messages.append(message)
        cl.user_session.set("messages_history", messages)
        
        msge = cl.Message(content="")
        print(f"Final result: {result}")
        
        try:
            for i in result: 
                if cl.user_session.get("stop_message"): 
                    cl.user_session.set("stop_message", False)
                    break
                await msge.stream_token(i)
            await msge.send()
        except Exception as e:
            # Fallback in case streaming fails
            await cl.Message(
                content=result,
            ).send()
            
    except Exception as e:
        print(f"Unexpected error in on_message: {str(e)}")
        await cl.Message(
            content=f"An unexpected error occurred: {str(e)}. Please try again or refresh the page.",
        ).send()



# @cl.on_stop
# def on_stop():
#     print("stopped")
#     cl.user_session.set("stop_message", True)

