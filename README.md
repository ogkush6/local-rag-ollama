# Local RAG(Not all files are in the repository)
---

The project implements the deployment of the language model locally, with the implementation of the response only using upload documents. The project includes loading with an internal database (possibly several timatic ones), as well as loading documents from the user and adding them to the general vector database. The project is already implementing two processed vector databases. And also the ability to start without them (from an empty database). You can also create your own database. The project was done as part of a university assignment, so a lot of it was done in haste.


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



