<!-- Improved compatibility of back to top link -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

</div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">🧠🤖 ChatBot Creation Using LLM</h3>

  <p align="center">
    A LangChain-powered chatbot that provides answers from CSV-based FAQs using vector search and LLMs.
    <br />
    <a href="https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM"><strong>Explore the repo »</strong></a>
    <br />
    <br />
    <a href="#usage">View Usage</a>
    ·
    <a href="https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM/issues">Report Bug</a>
    ·
    <a href="https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## 🔍 About The Project

This project is a lightweight, retrieval-based **Q&A chatbot** built using **LangChain**, **GooglePalm**, and **FAISS**. It loads FAQ-style documents from a CSV file and creates a semantic index (vector database) of the content using `HuggingFaceInstructEmbeddings`. When a user inputs a query, the chatbot retrieves the most relevant context and generates a context-aware answer using Google Palm LLM.

### ✨ Key Features:
- Extracts context-rich responses using **RetrievalQA chain**
- Uses **semantic embeddings** to understand query intent
- Employs a **custom prompt template** to reduce hallucination
- Powered by **GooglePalm** LLM and **FAISS vector store**
- User interface built using **Streamlit**

### 💡 Technical Workflow:
1. Loads a CSV of FAQs with `CSVLoader`
2. Embeds text using Hugging Face Instructor embeddings
3. Stores embeddings using `FAISS` vector DB
4. Retrieves relevant context chunks using similarity search
5. Generates answers via LangChain's `RetrievalQA` and a structured prompt

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- BUILT WITH -->
### Built With

* [LangChain](https://python.langchain.com/)
* [Google Palm](https://developers.generativeai.google/products/palm)
* [Hugging Face Transformers](https://huggingface.co/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io/)
* [Python-dotenv](https://pypi.org/project/python-dotenv/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## 🚀 Getting Started

### Prerequisites
Install the required packages:
```bash
pip install langchain streamlit openai faiss-cpu python-dotenv
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM.git
cd ChatBot_Creation_Using_LLM
```
2. Add your `GOOGLE_API_KEY` to a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key
```
3. Ensure `codebasics_faqs.csv` is present in the root directory.
4. Run the Streamlit app:
```bash
streamlit run main.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## 📊 Usage

- Click **"Create Knowledgebase"** to initialize the vector store
- Type your question in the input box (e.g., *"Do you offer internships?"*)
- The chatbot responds with answers sourced directly from the FAQ CSV file
- If no relevant context is found, it says: *"I don't know"*

This setup is ideal for building **internal documentation bots**, **customer support assistants**, or **educational chat interfaces** based on CSV/FAQ data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## 📆 Roadmap

- [x] Load CSV-based knowledgebase
- [x] Create semantic vector DB with FAISS
- [x] Integrate GooglePalm via LangChain
- [x] Build Streamlit frontend
- [ ] Add support for PDF and TXT ingestion
- [ ] Enhance response formatting with Markdown
- [ ] Add memory support to allow multi-turn conversation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## 👥 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## 📧 Contact

**Bhavani Malisetty**  
[GitHub](https://github.com/MalisettyBhavani)  
[LinkedIn](https://linkedin.com/in/malisettybhavani)

Project Link: [ChatBot Creation Using LLM](https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## 📖 Acknowledgments

* [LangChain Documentation](https://docs.langchain.com/)
* [Google Generative AI API](https://developers.generativeai.google/)
* [Streamlit Docs](https://docs.streamlit.io/)
* [HuggingFace Instruct Embeddings](https://huggingface.co/hkunlp/instructor-large)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/MalisettyBhavani/ChatBot_Creation_Using_LLM.svg?style=for-the-badge
[contributors-url]: https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MalisettyBhavani/ChatBot_Creation_Using_LLM.svg?style=for-the-badge
[forks-url]: https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM/network/members
[stars-shield]: https://img.shields.io/github/stars/MalisettyBhavani/ChatBot_Creation_Using_LLM.svg?style=for-the-badge
[stars-url]: https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM/stargazers
[issues-shield]: https://img.shields.io/github/issues/MalisettyBhavani/ChatBot_Creation_Using_LLM.svg?style=for-the-badge
[issues-url]: https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM/issues
[license-shield]: https://img.shields.io/github/license/MalisettyBhavani/ChatBot_Creation_Using_LLM.svg?style=for-the-badge
[license-url]: https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/malisettybhavani
