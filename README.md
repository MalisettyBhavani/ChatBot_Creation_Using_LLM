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
<!-- ABOUT THE PROJECT -->
## 🔍 About The Project

This project delivers an intelligent, retrieval-augmented Q&A chatbot leveraging advanced language models and vector similarity search to enable real-time, context-aware responses from structured FAQ data.

By integrating **LangChain**, **GooglePalm**, and **FAISS**, this chatbot transforms a CSV-based FAQ dataset into a high-performance semantic search system. Utilizing **HuggingFace Instruct Embeddings**, the chatbot encodes each question and answer pair into a dense vector space, facilitating precise similarity-based retrieval. Upon receiving a user query, the system efficiently retrieves semantically related FAQ entries and generates accurate, human-like answers via the Google Palm LLM.

### ✨ Key Features:
- Semantic search using **vector embeddings** and **FAISS**
- Query understanding and response generation powered by **GooglePalm LLM**
- Contextual filtering and answer synthesis via **LangChain's RetrievalQA**
- Custom prompt templating to mitigate hallucination and enforce relevance
- Lightweight and interactive **Streamlit-based UI** for rapid deployment

### 💡 Why This Project Matters:
This chatbot is ideal for automating **internal knowledge access**, **customer service interactions**, and **educational tools** where FAQ-based data can be reused. It significantly reduces manual support costs and improves user satisfaction by delivering:
- Low-latency, relevant answers based on indexed prior knowledge
- Scalable architecture with modular components for ingestion and inference
- Domain adaptability for enterprise, education, healthcare, and beyond

### 💡 Technical Workflow:
1. Load structured FAQ data using `CSVLoader`
2. Generate semantic vector representations with `HuggingFaceInstructEmbeddings`
3. Store and index vectors in a FAISS database
4. Retrieve top-k relevant chunks using cosine similarity
5. Feed results into LangChain’s `RetrievalQA` chain
6. Generate concise, accurate answers with GooglePalm LLM

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- BUILT WITH -->
### Built With

* [![LangChain][LangChain-badge]][LangChain-url]
* [![Google Palm][Palm-badge]][Palm-url]
* [![Hugging Face][HuggingFace-badge]][HuggingFace-url]
* [![FAISS][FAISS-badge]][FAISS-url]
* [![Streamlit][Streamlit-badge]][Streamlit-url]
* [![python-dotenv][Dotenv-badge]][Dotenv-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

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
## Usage

- Click **"Create Knowledgebase"** to initialize the vector store
- Type your question in the input box (e.g., *"Do you offer internships?"*)
- The chatbot responds with answers sourced directly from the FAQ CSV file
- If no relevant context is found, it says: *"I don't know"*

This setup is ideal for building **internal documentation bots**, **customer support assistants**, or **educational chat interfaces** based on CSV/FAQ data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Load CSV-based knowledgebase
- [x] Create semantic vector DB with FAISS
- [x] Integrate GooglePalm via LangChain
- [x] Build Streamlit frontend
- [ ] Add support for PDF and TXT ingestion
- [ ] Enhance response formatting with Markdown
- [ ] Add memory support to allow multi-turn conversation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

**Bhavani Malisetty**  
**Email: bmalisetty@unomaha.edu**  
[GitHub](https://github.com/MalisettyBhavani)  
[LinkedIn](https://www.linkedin.com/in/bhavani-malisetty/)

Project Link: [ChatBot Creation Using LLM](https://github.com/MalisettyBhavani/ChatBot_Creation_Using_LLM)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

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
[LangChain-badge]: https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white
[LangChain-url]: https://python.langchain.com/
[Palm-badge]: https://img.shields.io/badge/Google%20Palm-4285F4?style=for-the-badge&logo=google&logoColor=white
[Palm-url]: https://developers.generativeai.google/products/palm
[HuggingFace-badge]: https://img.shields.io/badge/Hugging%20Face-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black
[HuggingFace-url]: https://huggingface.co/
[FAISS-badge]: https://img.shields.io/badge/FAISS-005571?style=for-the-badge
[FAISS-url]: https://github.com/facebookresearch/faiss
[Streamlit-badge]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
[Dotenv-badge]: https://img.shields.io/badge/python--dotenv-367cfe?style=for-the-badge
[Dotenv-url]: https://pypi.org/project/python-dotenv/
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/bhavani-malisetty/
