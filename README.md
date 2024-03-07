## Webby: Chatbot for Website Interaction with Streamlit GUI
Welcome to the GitHub repository for Webby, a chatbot designed for interacting with websites through a Streamlit GUI. Webby integrates LangChain technology to extract information from various websites and provide user-friendly responses.

### Key Features
- Website Interaction: Webby leverages LangChain to interact with websites and extract relevant information.
- Advanced Language Models: Compatibility with various language models such as GPT-4, Mistral, Llama2, and ollama.
- Intuitive Streamlit GUI: The user interface is built with Streamlit, ensuring an easy and accessible experience for users.
- Python-based: Entirely coded in Python for flexibility and ease of use.
- Entirely langchain bassed development. Easy integration of multiple new features possible.
- Webby utilizes a Retrieval-Augmented Generation (RAG) approach. This involves augmenting the knowledge of the language model with additional information passed in the prompt. Text is vectorized to find -the most similar content to the prompt, which is then passed to the language model as a prefix.

### Installation
1. Clone this repository to your local machine.

```
git clone [repository-link]
cd [repository-directory]
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Create a .env file with your OpenAI API key:

```
OPENAI_API_KEY=[your-openai-api-key]
```

4. Usage
To run the Webby application:

```
streamlit run app.py
```
## Contributing
This repository is primarily for tutorial purposes. Therefore, contributions are limited to bug fixes or typo corrections.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Note: This project is for educational and research purposes. Ensure to comply with the terms of use and guidelines of the utilized APIs and services.

I hope you find Webby helpful in your exploration of AI and chatbot development.

Happy Coding! 🚀👨‍💻🤖

Don't forget to star this repo if you find it useful!

