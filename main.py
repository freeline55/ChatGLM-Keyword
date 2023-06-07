import gradio as gr
import os
import shutil
from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
from langchain.llms.base import LLM
from models.chatglm_llm import ChatGLM
from utils import torch_gc
import streamlit as st
import numpy as np
import pandas as np
from pyecharts.charts import WordCloud
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
import streamlit.components.v1 as components

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

@st.cache_resource(ttl=10800)
def load_chatglm():
    llm = ChatGLM()
    # llm.load_model(model_name_or_path="THUDM/chatglm-6b-int8", llm_device="cuda:0", use_ptuning_v2=False, use_lora=False)
    llm.load_model(model_name_or_path="/root/.cache/huggingface/hub/models--THUDM--chatglm-6b-int8/snapshots/22906aeb32fd7952ce323dc9d25e01693b270da6", llm_device="cuda:0", use_ptuning_v2=False,use_lora=False)
    llm.temperature = 1e-3
    print("模型加载完毕")
    return llm

llm = load_chatglm()

def get_answer(prompt):
    torch_gc()
    for result, history in llm._call(prompt=prompt, history=[], streaming=False):
        torch_gc()
        yield result
        torch_gc()

st.title('ChatGLM关键词抽取')
uploaded_file = st.file_uploader("请上传文本文件")

col1, col2 = st.columns([5, 1])
with col2:
    shape_option = st.selectbox( "请选择词云图形", ( 'circle', 'cardioid', 'diamond', 'triangle-forward', 'triangle', 'pentagon', 'star'))
    st.write("")
    st.write("")
    st.write("")
    len = st.slider("请选择关键词的个数", 5, 20, 10)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    prompt = "你扮演的角色是关键词抽取工具,请从输入的文本中抽取出10个最重要的关键词: \n" + content
    print("prompt:", prompt)

    for res in get_answer(prompt):
        print("回复的结果是:\n", res)
        words = [(r[r.index(".") + 1:].strip(), content.count(r[r.index(".") + 1:].strip())) for r in res.split("\n")]
        print("词云值列表是:\n", words)

        with col1:
            c = (
                WordCloud(init_opts=opts.InitOpts(width="500px", height="500px"))
                .add("",
                     words,
                     shape=shape_option,
                     word_size_range=[5, 40]
                     )
            )
            # st_pyecharts(c)
            c2html = c.render_embed()
            components.html(c2html, height=600, width=600, scrolling=True)

        with col2:
            st.download_button("下载", c2html, file_name="keyword.html", mime="html")

