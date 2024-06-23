from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
from langsmith import traceable 
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

example_customer_reviews = [
    "WidgetWorld exceeded my expectations with their top-notch widgets! Our machines run smoother than ever before.",
    "Impressed by WidgetWorld's attention to detail and commitment to quality. Their widgets are a game-changer for our business.",
    "WidgetWorld's widgets are reliable and durable, making them an essential component in our manufacturing process.",
    "WidgetWorld's widgets are decent, but nothing exceptional. They get the job done adequately.",
    "Average experience with WidgetWorld. Their widgets function as expected, but nothing to write home about.",
    "WidgetWorld's widgets are alright, but there's room for improvement in terms of durability.",
    "Avoid WidgetWorld at all costs! Their widgets are cheaply made and constantly malfunction.",
    "Terrible experience with WidgetWorld. Their widgets broke down within weeks of installation.",
    "WidgetWorld's widgets are a nightmare. We've had nothing but problems since day one."
]

initial_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a world class Customer Service representative for the company WidgetWorld who responds to customer reviews. 
                  You are always polite, helpful, and professional. Be sure to respond appropriately to the sentiment of the review."""),
    ("human", "{input_text}")
])

@traceable
def generate_customer_review_responses():
    chain = initial_prompt | ChatOpenAI(temperature=0.9)
    for i in range(len(example_customer_reviews)):
        chain.invoke({"input_text": example_customer_reviews[i]})

generate_customer_review_responses()
