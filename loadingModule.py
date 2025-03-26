from langchain_community.document_loaders import AsyncChromiumLoader 
from langchain_community.document_transformers import BeautifulSoupTransformer 
from langchain_core.documents import Document
from bs4 import BeautifulSoup

from langchain.chains import RetrievalQA
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker 
from tqdm import tqdm
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    PyMuPDFLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant




def loading_documents_and_save_db(file_path, db_name, model_name): 
    absolute_path = os.path.abspath(file_path)
    files = os.listdir(absolute_path)
    
    if model_name == "":
        model_name = "deepseek-r1:7b"
    
    embedding = OllamaEmbeddings(model=model_name)
    
    text_splitter = SemanticChunker(embedding, buffer_size=7, breakpoint_threshold_type="interquartile") 
    documents = []
    for i in tqdm(range(len(files)), desc="Load pdfs", unit="file"):
        loader = PyMuPDFLoader(absolute_path + "/" + files[i])
        data = loader.load_and_split(text_splitter)
        for j in data: 
            if(len(j.page_content) > 12): 
                documents.append(j)
    docs = []
    for i in range(len(documents)):
        docs.append(Document(page_content="[[metadata: file: {}, page: {}]\n[Text: ".format(documents[i].metadata['source'].split('/')[-1], str(int(documents[i].metadata['page'])+1))+documents[i].page_content.replace('\n', ' ').replace('\\', ' ').replace('\t', ' ').replace('  ', ' ')+"]]", metadata=documents[i].metadata))    
    
    urls = [
            "https://en.wikipedia.org/wiki/Neural_network",
            "https://en.wikipedia.org/wiki/Neuron",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Ethics_of_artificial_intelligence",
            "https://en.wikipedia.org/wiki/History_of_artificial_intelligence",
            "https://en.wikipedia.org/wiki/Outline_of_artificial_intelligence",
            "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
            "https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare",
            "https://en.wikipedia.org/wiki/Regulation_of_artificial_intelligence",
            "https://en.wikipedia.org/wiki/Human%E2%80%93artificial_intelligence_collaboration",
            "https://en.wikipedia.org/wiki/Applications_of_artificial_intelligence",
            "https://en.wikipedia.org/wiki/Computing_Machinery_and_Intelligence",
            "https://en.wikipedia.org/wiki/Artificial_Intelligence:_A_Guide_for_Thinking_Humans",
            "https://en.wikipedia.org/wiki/Allen_Institute_for_AI",
            "https://en.wikipedia.org/wiki/Artificial_Intelligence_Act",
            "https://en.wikipedia.org/wiki/Artificial_intelligence_in_industry",
            "https://en.wikipedia.org/wiki/Computational_intelligence",
            "https://en.wikipedia.org/wiki/Computer_science",
            "https://en.wikipedia.org/wiki/Outline_of_computer_science",
            "https://en.wikipedia.org/wiki/Adaptive_learning",
            "https://en.wikipedia.org/wiki/Human_intelligence",
            "https://en.wikipedia.org/wiki/Geoffrey_Hinton",
            "https://en.wikipedia.org/wiki/Computer_science",
            "https://en.wikipedia.org/wiki/Outline_of_computer_science",
            "https://en.wikipedia.org/wiki/Adaptive_learning",
            "https://en.wikipedia.org/wiki/Human_intelligence",
            "https://en.wikipedia.org/wiki/Geoffrey_Hinton"
            ]
    wiki_docs = web_scrapping_wiki(urls)
    wiki_docs = text_splitter.transform_documents(wiki_docs)
    new_docs = []
    for i in range(len(wiki_docs)):
        docs.append(Document(page_content="[[metadata: {}]\n[Text: ".format(wiki_docs[i].metadata['source'])+wiki_docs[i].page_content.replace('\n', ' ').replace('\\', ' ').replace('\t', ' ').replace('  ', ' ')+" ]]", metadata=wiki_docs[i].metadata))    
    docs = new_docs+docs 

    faissdb = FAISS.from_documents(docs, embedding)
    faissdb.save_local(db_name)
    


def loading_document_and_add_to_db(file_path, embedding, faissdb, file_name): 
    try:
        # Check for PyMuPDF import
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except ImportError:
            raise ImportError(
                "Unable to import PyMuPDFLoader. Please install with 'pip install pymupdf'"
            )
        
        text_splitter = SemanticChunker(embedding, buffer_size=7, breakpoint_threshold_type="interquartile")
        documents = []
        print(f"Loading document: {file_name}")
        
        # Load the PDF file
        loader = PyMuPDFLoader(file_path)
        data = loader.load_and_split(text_splitter)
        
        # Filter out short content
        for j in data: 
            if(len(j.page_content) > 12): 
                documents.append(j)
                
        # Check if any documents were extracted
        if not documents:
            print(f"Warning: No usable content found in {file_name}")
        
        # Create documents with formatted metadata
        docs = []
        for i in range(len(documents)):
            try:
                # Clean up the text content to avoid encoding issues
                clean_content = (documents[i].page_content
                                .replace('\n', ' ')
                                .replace('\\', ' ')
                                .replace('\t', ' ')
                                .replace('  ', ' '))
                
                # Format the document with metadata
                doc = Document(
                    page_content=f"[[metadata: file: {documents[i].metadata['source'].split('/')[-1]}, page: {str(int(documents[i].metadata['page'])+1)}]\n[Text: {clean_content}]]", 
                    metadata=documents[i].metadata
                )
                docs.append(doc)
            except Exception as e:
                print(f"Error processing document page {i}: {str(e)}")
        
        # Create a new FAISS index with the documents and merge with existing
        if docs:
            print(f"Adding {len(docs)} segments from {file_name} to vector database")
            faissdb.merge_from(FAISS.from_documents(docs, embedding))
            print(f"Successfully added {file_name} to vector database")
        else:
            print(f"No documents to add from {file_name}")
            
        return faissdb
        
    except ImportError as e:
        print(f"Import error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error loading document {file_name}: {str(e)}")
        raise





def web_scrapping_wiki(urls): 
    loader = AsyncChromiumLoader(urls)
    html = loader.load()
    new_html = []
    for i in html: 
        page_content = i.page_content
        soup = BeautifulSoup(page_content, 'html.parser')
        classes_to_remove = [
                "vector-column-start", "mw-jump-link", "vector-sticky-pinned-container", 
                "vector-page-toolbar", "mw-references-wrap", "vector-settings", 
                "mw-hidden-catlinks", "refbegin"
                ]
        tags_to_remove = [
                "script", "footer", "header", "head",
                "cite"
                ]
        for cls in classes_to_remove: 
            for elem in soup.find_all(class_=cls): 
                elem.decompose()
        for tag in tags_to_remove: 
            for elem in soup.find_all(tag): 
                elem.decompose()
        page_content = str(soup)
        new_html.append(Document(page_content=page_content, metadata=i.metadata))
    html = new_html
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=("p", "li", "a"))
    return docs_transformed



    
def update_retriever(llm, retriever, prompt): 
    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever = retriever,
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
        )
