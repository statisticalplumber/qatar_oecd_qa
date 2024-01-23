import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly_express as px 
from typing import Any, Optional, Union
from uuid import UUID
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
import streamlit as st 
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.llamacpp import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


st.set_page_config(layout='wide')

st.markdown('''<style>
[data-testid="baseButton-header"] {
    display: none;
}
</style>''', unsafe_allow_html=True)


key = st.sidebar.text_input('Enter OpenAI Key to get detailed insights', type='password')
with st.sidebar.container():
    gif_col = st.columns([2,2])
    gif_col[0].image('https://raw.githubusercontent.com/tylerjrichards/GPT3-Dataset-Generator-V2/main/Gifs/blue_grey_arrow.gif', width=40)
    gif_col[1].caption(
            "No OpenAI API key? Get yours [here!](https://openai.com/blog/api-no-waitlist/)"
        )

col_header = st.columns([1,1.5,10, 5])
col_header[1].image('https://wallpapercave.com/wp/wp4214269.png', width=80)
col_header[2].header(':blue[Qatar Performance in 2022 PISA]')
col_header[2].markdown('''*PISA is the OECD's Programme for International Student Assessment.
                       PISA measures 15-year-olds’ ability to use their reading, mathematics and science knowledge
                       and skills to meet real-life challenges.*''')

csv_path = 'data/cleaned_qatar_oecd.csv'


@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    return df 

df = load_data()
df['Gender'] = df['ST004D01T']
df['overall_score'] =  df[['Reading', 'Math', 'Science']].mean(1)

col_mark = st.columns([9,2,9])
col_mark[0].markdown('----------------')
col_mark[1].markdown('#### :violet[*Overview*]')
col_mark[2].markdown('----------------')

with st.container(height=100):
    col_subheader = st.columns([1, 3,3,3,1])
    with col_subheader[1]: 
        st.markdown(f'#### Number of Students: :green[**{df.shape[0]}**]')
    with col_subheader[2]: 
        st.markdown(f'#### Number of Schools: :green[**{df.CNTSCHID.nunique()}**]')
    with col_subheader[3]: 
        st.markdown(f'#### PISA Score: :green[**{df.overall_score.mean().round(2)}**]')


# Performance by city and gender
def bar_performance_subject():
    df_out = df[['Reading', 'Math', 'Science']].mean().to_frame('Score').reset_index(names='Subject').round(2)
    fig = px.bar(df_out, x = 'Subject', y='Score', color = 'Subject', text_auto=True, width=400, height=300)
    return fig, df_out.to_csv() 

def bar_performance_school():
    df_out = df.groupby('School_Type')[['Reading', 'Math', 'Science']
                                       ].mean().mean(1).to_frame('Score').reset_index().round(2)
    fig = px.bar(df_out, x = 'Score', y='School_Type', color = 'School_Type', text_auto=True, width=400, height=300)
    return fig, df_out.to_csv() 

def bar_performance_gender():
    df_out = df.groupby('Gender')[['Reading', 'Math', 'Science']
                                       ].mean().mean(1).to_frame('Score').reset_index().round(2)
    fig = px.bar(df_out, x = 'Score', y='Gender', color = 'Gender', text_auto=True, width=400, height=300)
    return fig, df_out.to_csv()

def student_cnt_school_type():
    df_out = df.groupby('School_Type').size().to_frame('Number of students').reset_index()
    fig = px.pie(df_out, values='Number of students', names='School_Type', height=400, width=500)
    return fig, df_out.to_csv()  

def gender_pyramid_chart():
    df['Score_Bin'] = pd.cut(df.overall_score, bins=[0, 300, 350, 400, 450, 500, 550, 600, 700, 800])
    out = df.groupby(['Score_Bin', 'Gender']).size().unstack().reset_index()

    # Create a dataframe with the data
    out_df = pd.DataFrame(dict(
        Score=out.Score_Bin.astype(str).tolist(),
        male=out.Male.tolist(),
        female=(out.Female*-1).tolist()
    ))


    # Create a pyramid chart using px.bar
    fig = px.bar(out_df, x=['male', 'female'], y='Score', orientation='h', barmode='overlay',
                labels=dict(male='Number', female='Number', Score='Score Bin'),
                range_x=[-1200, 1200],
                color_discrete_map={'male': 'powderblue', 'female': 'seagreen'},
                text_auto=True, height=400, width=600)

    # Customize the x-axis ticks and title
    fig.update_xaxes(tickvals=[-1000, -700, -300, 0, 300, 700, 1000],
                    ticktext=[1000, 700, 300, 0, 300, 700, 1000],
                    title_text='Number')

    # Show the figure
    return fig

#run plot 
fig_subject, out_subject = bar_performance_subject()
fig_school, out_school = bar_performance_school()
fig_gender, out_gender  = bar_performance_gender()
fig_pyramid = gender_pyramid_chart()
fig_pie, out_pie = student_cnt_school_type()

col_mark = st.columns([9,2,9])
col_mark[0].markdown('----------------')
col_mark[1].markdown('#### :violet[*Performance*]')
col_mark[2].markdown('----------------')

with st.container(height=400):
    cols = st.columns([3,3,3])
    cols[0].markdown("#### Performance By Subjects")
    cols[0].plotly_chart(fig_subject)
    cols[1].markdown("#### Performance By School Types")
    cols[1].plotly_chart(fig_school)
    cols[2].markdown("#### Performance By Gender")
    cols[2].plotly_chart(fig_gender)

col_mark = st.columns([9,2,9])
col_mark[0].markdown('----------------')
col_mark[1].markdown('#### :violet[*Distribution*]')
col_mark[2].markdown('----------------')


with st.container(height=500):
    col_mark = st.columns([10,10])
    col_mark[0].markdown('#### Distribution of Student By Score and Gender')
    col_mark[0].plotly_chart(fig_pyramid)
    col_mark[1].markdown('#### Distribution of Student By School Type')
    col_mark[1].plotly_chart(fig_pie)

class StreamCallback(BaseCallbackHandler):
    def __init__(self, container) -> None:
        self.container = container
        self.token = ''
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.token +=token
        self.container.write(self.token)

@st.cache_resource
def get_llm():
    n_gpu_layers = 40
    n_batch = 512
    # model = LlamaCpp(
    #     model_path="../mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    #     n_gpu_layers=n_gpu_layers,
    #     n_batch=n_batch,
    #     n_ctx = 4000,
    #     max_tokens = 2000, 
    #     temperature=0.7,
    #     verbose=True,  # Verbose is required to pass to the callback manager
    # )
    model=ChatOpenAI(openai_api_key=key, 
                          model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    return model 

def get_llm_chain():
    # template = '''
    # <|system|>You are smart assitant</s>
    # <|user|>{prompt}</s>
    # <|assistant|>
    # '''
    template = f'''
    [INS]You are smart assitant, you have given access to these Qatar PISA 2022 Performance data, based on user question you will select relevant data to answer[INS]</s>
    [INS] If question is generic then use your own knowledge to answer in very short. [INS]
    [INS]Data: Performance of students by Gender: {out_gender}, Performance by Subject: {out_subject},
          Performance by school_type: {out_school}, Distribution of students by School Type: {out_pie} [/INS]
    question: {{prompt}}</s>
    Response: 
    '''
    st.info(template.split(' ').__len__())

    prompt_template = PromptTemplate.from_template(template)
    model_chain = LLMChain(prompt=prompt_template, llm=get_llm(), verbose=True)
    return model_chain

st.sidebar.markdown('## :rainbow: Conversation History')
st.sidebar.markdown('------------')


prompt: str = st.chat_input('Enter your query to get insights about Qatar performance in OECD 2022', key = 'prompt', disabled= not key)

context = out_school

if prompt:
    with st.sidebar:
        st.chat_message('user').write(prompt)
        llm = get_llm_chain()
        with st.chat_message('assistant'):
            st_callback = StreamCallback(st.empty())
            result = llm({'prompt':prompt}, callbacks = [st_callback])['text']

