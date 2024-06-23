from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
from langsmith import traceable, Client
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
langsmith_client = Client()

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

customer_service_system_message = SystemMessage(content="""You are a world class Customer Service representative for the company WidgetWorld who responds to customer reviews. 
                  You are always polite, helpful, and professional. Be sure to respond appropriately to the sentiment of the review.""")

initial_prompt = ChatPromptTemplate.from_messages([
   customer_service_system_message,
    ("human", "{input_text}")
])

@traceable
def generate_customer_review_responses():
    chain = initial_prompt | ChatOpenAI(temperature=0.9)
    for i in range(len(example_customer_reviews)):
        chain.invoke({"input_text": example_customer_reviews[i]})

#Part 1: generates initial responses to sample customer reviews to track and score within langsmith
# generate_customer_review_responses()


#Part 2: Use created dataset to act as examples in a FewShotPromptTemplate
# Below creates a chat like experience which will provide a customer review response using examples collected in a dataset as context
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input_text}"),
        ("ai", "{output_text}"),
    ]
)

selected_customer_reviews_dataset = langsmith_client.list_examples(dataset_id="c2cf9a40-bb80-4d88-bfb0-c19adec94123")

dataset_examples = [
    {"input_text": input_item['data']['content'], "output_text": example.outputs["output"]["data"]["content"]}
    for example in selected_customer_reviews_dataset
    for input_item in example.inputs["input"]
    if input_item['type'] == 'human'
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=dataset_examples
)
final_prompt = ChatPromptTemplate.from_messages([
    customer_service_system_message,
    few_shot_prompt,
    ("human", "{input_text}")
])

@traceable
def invoke_final_prompt(user_input):
    return chain.invoke({"input_text": user_input}).content

chain = final_prompt | ChatOpenAI(temperature=0.9)
user_prompt = ""
while user_prompt != "exit":
    user_prompt = input("Enter a customer review, or 'exit' to stop: ")
    if user_prompt != "exit":
        print(invoke_final_prompt(user_prompt))