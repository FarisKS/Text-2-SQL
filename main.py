import os
os.environ["LANGCHAIN_API_KEY"] = "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"

os.environ["LANGCHAIN_PROJECT"] = "Text2SQL Hospital Insights"

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import re
# ... rest of your imports


import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from langchain_community.utilities.sql_database import SQLDatabase

from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
import re


@st.cache_resource
def load_database():
    df = pd.read_csv("C:/Users/Usmanul Faris/Desktop/venvtutorial/data/bed_revenue.csv")
    df['Sl.No'] = range(1, len(df) + 1)

    with sqlite3.connect("hospital.db") as conn:
        df.to_sql("hospital", conn, if_exists="replace", index=False)

    engine = create_engine("sqlite:///hospital.db", poolclass=StaticPool)
    db = SQLDatabase(engine)
    return df, db

df, db = load_database()


llm = ChatGroq(
    model="qwen-qwq-32b",
    temperature=0.2,
    max_retries=2,
    api_key=""
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

system_message = prompt_template.format(dialect="SQLite", top_k=len(df))
agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)

st.title("🧠 Text2SQL Hospital Insights")
st.markdown("Ask a question in natural language, and the model will query the hospital database.")

user_query = st.text_input("💬 Enter your question:")

if st.button("Generate SQL and Get Answer"):
    with st.spinner("Generating SQL and executing..."):
        try:
            events = agent_executor.stream(
                {"messages": [("user", user_query)]},
                stream_mode="values",
            )
            sql_result = None
            for event in events:
                sql_result = event["messages"][-1].content
                st.code(sql_result, language="sql")

           
            if sql_result.strip().lower().startswith("select"):
                engine = create_engine("sqlite:///hospital.db")
                
                with engine.connect() as conn:
                    result_df = pd.read_sql_query(sql_result, conn)
                st.success("✅ Query Result:")
                st.dataframe(result_df)

              
                if len(result_df.columns) >= 2 and result_df.dtypes[1] in [int, float]:
                    st.markdown("### 📊 Chart")
                    fig, ax = plt.subplots()
                    sns.barplot(data=result_df, x=result_df.columns[0], y=result_df.columns[1], ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            else:
                st.warning("ℹ️ The response was not a SELECT query.")

        except Exception as e:
            st.error(f" Error: {e}")
