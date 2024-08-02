# Local RAG(Not all files are in the repository)
---

The project implements the deployment of the language model locally, with the implementation of the response only using upload documents. The project includes loading with an internal database (possibly several timatic ones), as well as loading documents from the user and adding them to the general vector database. The project is already implementing one processed vector databases. And also the ability to start without them (from an empty database). You can also create your own database.

---
1. [Setting](#setting)
2. [Launch](#launch)
3. [Examples](#examples)
4. [Some Fun](#some-fun)
---


### Setting
1. Download and setup [Ollama](https://ollama.ai/download) for your os. For check ollama open comand line and type `ollama help`, you should see ollama help message.
2. Download mistral in ollama. For this use `ollama run mistral` and wait for loading.
3. Download the git repo.
4. When you in git repo type `python -m venv env`
5. Activate virtual environment `source env/bin/activate` or for Windows
`.\env\Scripts\activate`.
6. Install pip package `pip install -r ollama-rag-pip.txt`

### Launch
1. In another terminal, run ollama `ollama serve`
2. The experiments were carried out on a powerful university server. To run on the server, enter the following command to run on a specific port. `python -m chainlit run app.py -h --port [your_port]`
To run locally on your computer, you can enter a simplified command.
`python -m chainlit run app.py`
Your chatbot UI should now be accessible at http://localhost:8000.




### Examples
1. ![Exmaple of ML answer clue](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/MLDoc.JPG)![ML answer](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/MLAnswer.JPG)
2. ![Linear Regression in python answer](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/linearRegression.JPG)![Linear Regression in python answer clue](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/reggressionLinear.JPG)
3. ![Neural wiki answer clue](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/neuralWikiDoc.png)![Neural wiki answer](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/neuralWikiAnswer.png)



### Some Fun😛
In general, due to the fact that the model is only in English, and the database is searched by vectors, it cannot search for relevant parts of documents in the database, so we decided to experiment. In general, we can use the power of the language model to understand other languages. The point is that in order to generate some answer to a question (hallucination), we don’t care if it’s correct or not, the most important thing for us is that it will end up in the vector space next to the possible answers to the question. And already using the search for this answer, we can find relevant pieces of documents and generate an answer.

Experiment with the Slovak question
![Query sheme](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/queryScheme.jpg)
![Slovak question example](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/querySchemeExample.jpg)
