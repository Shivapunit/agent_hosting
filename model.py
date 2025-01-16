from mlserver import types
from mlserver.model import MLModel
from mlserver.codecs import NumpyCodec, StringCodec
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.logging import logger

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import bs4


class LangChainApp(MLModel):
    async def load(self) -> bool:
        openai_api_key = self._settings.parameters.extra['openai_api_key']
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
        )
        self.loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = self.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        self.ready = True
        logger.info(f"---- loaded LangChain Application ----")
        return self.ready

    def unpack_input(self, payload: InferenceRequest):
        request_data = {}
        for inp in payload.inputs:
            if inp.name == 'query':
                request_data[inp.name] = (
                    NumpyCodec
                    .decode_input(inp)
                    .flatten()
                    .tolist()
                )
        return request_data
    
    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        unpacked_payload = self.unpack_input(payload)
        response = self.rag_chain.invoke(unpacked_payload['query'][0])
        outputs = [StringCodec.encode_output("response", [response])]
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=outputs,
        )
