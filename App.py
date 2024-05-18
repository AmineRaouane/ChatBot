from apikey import APIKEY as key

import os  

import streamlit as st 
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ["HUGGINGFACE_API_TOKEN"] = key

if "Data" not in st.session_state:
    st.session_state.Data = {}

# App framework
st.title('ChatBot 🤖')
prompt = st.text_input('Hi there how can i help you ?') 

# Prompt templates
script_template = PromptTemplate(
    input_variables = ['prompt', 'wikipedia_research'], 
    template='{prompt} (while leveraging this wikipedia reserch:{wikipedia_research} and write you answer in the format Answer: '
)

# Memory 
script_memory = ConversationBufferMemory(input_key='prompt', memory_key='chat_history')


# Llms
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=key,
    model_kwargs={"temperature": 0.001, "max_length": 500,"max_new_tokens": 250}
)

script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper(wiki_client=None)

# Show stuff to the screen if there's a prompt
if prompt: 
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(prompt=prompt, wikipedia_research=wiki_research)

    st.write(script.split("Answer:")[1]) 
    st.session_state.Data[prompt] = (script_memory.buffer,wiki_research)


with st.sidebar :
    for prompt, (script_history, wiki_research) in st.session_state.Data.items():
        with st.expander(f"{prompt[:30]}..."):
            with st.popover('Your input'): 
                st.write(prompt)
            with st.popover('AI Answer'): 
                st.info(script_memory.buffer)
            with st.popover('Wikipedia Research'): 
                st.info(wiki_research)