import json

from typing import List
from pprint import pprint
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field



model='meta/llama-3.1—8b-instruct'
llm = ChatNVIDIA(model=model, max_retries=3, temperature=0)

with open("emails.json", "r") as file:
    emails = [line.strip() for line in file if line.strip()]


emails_text = "\n".join([f"{i+1}. {email}" for i, email in enumerate(emails)])


syntiments_analyses_prompt_template = ChatPromptTemplate.from_messages([
    ("system",""" you will be provided with a list of emails\
Make a JSON object representing whether the email is positive or negative. \
It should have fields for: \
- the number of the email \
- Each value is a list with EXACTLY three elements in this order:
1) sentiment: "positive" or "negative"
2) location: the city mentioned (lowercase). If none, use "unknown".
3) product_category: MUST be ONE of: ["furniture", "appliances", "clothing","kitchenware", "beauty", "toys", "groceries", "services", "other"]
as example {{"1":["positive","new york","clothing"] , "2":["negative","london","appliances"], and so on}}
Don't generate any introduction, conclusion, or new lines or anything except the JSON object 
"""),
    ("human","{emails}")
])


parser = JsonOutputParser()

chain_dict = syntiments_analyses_prompt_template | llm | parser


class dictinary(BaseModel):
    """ the description of dict"""
    data : dict = Field(...,description="an analysis dict, Each key is an index, and each value is a list: [sentiment, location, product_category].")

@tool(args_schema=dictinary)
def analyze_sentiments(data):
    negative_product_counts = {}
    for value in data.values():
        if value[0] == 'negative':
            category = value[2]
            if category in negative_product_counts:
                negative_product_counts[category] += 1
            else:
                negative_product_counts[category] = 1

    negative_location_counts = {}
    for value in data.values():
        if value[0] == 'negative':
            location = value[1]
            if location in negative_location_counts:
                negative_location_counts[location] += 1
            else:
                negative_location_counts[location] = 1

    most_negative_product = max(negative_product_counts, key=negative_product_counts.get)

    most_negative_location = max(negative_location_counts, key=negative_location_counts.get)

    return most_negative_product, most_negative_location

tools = [analyze_sentiments]

system_message = """\
You are a helpful assistant capable of tool calling when helpful, necessary, and appropriate.

Think hard about whether or not you need to call a tool, \
based on your tools' descriptions and use them, but only when appropriate!

Whether or not you need to call a tool, address the user's query in a helpful informative way.
"""

agent_state_parser = RunnableLambda(lambda final_agent_state: final_agent_state['messages'][-1].content)

convert_to_agent_state = RunnableLambda(lambda prompt: {'messages': [prompt]})

agent = create_react_agent(llm,tools=tools,state_modifier=system_message)

agent_chain = convert_to_agent_state | agent | agent_state_parser

final_chain = chain_dict | RunnableLambda(
    lambda analysis_dict: f"return the most negative product and most negative location in this dictinary {analysis_dict}"
) | agent_chain

output = final_chain.invoke({
    "emails": emails_text,
})


print(output)

#The most negative product is 'furniture' and the most negative location is 'new york'.
