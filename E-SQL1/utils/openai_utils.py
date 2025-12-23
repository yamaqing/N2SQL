from typing import Dict
import dashscope
from dashscope import Generation

# 设置千问API密钥，从环境变量获取
import os
from dotenv import load_dotenv
load_dotenv()

def create_response(stage: str, prompt: str, model: str, max_tokens: int, temperature: float, top_p: float,  n: int) -> Dict:
    """
    The functions creates chat response by using chat completion

    Arguments:
        stage (str): stage in the pipeline
        prompt (str): prepared prompt 
        model (str): LLM model used to create chat completion
        max_tokens (int): The maximum number of tokens that can be generated in the chat completion
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling
        n (int): Number of chat completion for each input message

    Returns:
        response_object (Dict): Object returned by the model
    """
    
    # 根据不同阶段设置系统提示
    if stage == "question_enrichment":
        system_content = "You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions and the evidence to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items."
    elif stage == "candidate_sql_generation":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "sql_refinement":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "schema_filtering":
        system_content = "You are an excellent data scientist. You can capture the link between a question and corresponding database and determine the useful database items (tables and columns) perfectly. Your objective is to analyze and understand the essence of the given question, corresponding database schema, database column descriptions, samples and evidence and then select the useful database items such as tables and columns. This database item filtering is essential for eliminating unnecessary information in the database so that corresponding structured query language (SQL) of the question can be generated correctly in later steps."
    else:
        raise ValueError("Wrong value for stage. It can only take following values: question_enrichment, candidate_sql_generation, sql_refinement or schema_filtering.")

    # 使用千问API进行聊天补全
    response = Generation.call(
        model="qwen-max" if "qwen" not in model.lower() else model,  # 如果模型名称中包含qwen，则使用指定的模型
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        result_format="message"  # 返回格式为message，类似OpenAI格式
    )
    
    # 将千问API响应格式转换为与OpenAI兼容的格式
    response_object = {
        'choices': [
            {
                'message': {
                    'content': response.output.choices[0].message.content if response.output.choices else "",
                    'role': 'assistant'
                },
                'index': 0,
                'finish_reason': response.output.choices[0].finish_reason if response.output.choices else 'stop'
            }
        ],
        'created': response.created_at if hasattr(response, 'created_at') else 0,
        'id': response.request_id if hasattr(response, 'request_id') else '',
        'model': model,
        'object': 'chat.completion',
        'usage': {
            'completion_tokens': response.usage.output_tokens if hasattr(response, 'usage') else 0,
            'prompt_tokens': response.usage.input_tokens if hasattr(response, 'usage') else 0,
            'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0
        }
    }

    return response_object


def upload_file_to_openai(file_path: str) -> Dict:
    """
    The function uploads given file to opanai for batch processing.

    Arguments:
        file_path (str): path of the file that is going to be uplaoded
    Returns:
        file_object (FileObject): Returned file object by openai
    """
    # 千问API不支持文件上传功能，返回模拟对象
    print("Warning: Qwen API does not support file upload. This function is not implemented for Qwen.")
    file_object = {
        "id": "qwen_file_mock",
        "filename": os.path.basename(file_path),
        "purpose": "batch",
        "status": "uploaded_mock"
    }
    print("File upload is not supported by Qwen API, returning mock object")
    return file_object


def construct_request_input_object(prompt: str, id: int, model: str, system_message: str) -> Dict:
    """
    The function creates a request input object for each item in the dataset

    Arguments:
        prompt (str): prompt that is going to given to the LLM as content
        id (int); the id of the request
        model (str): LLM model name
        system_message (str): the content of the system message

    Returns:
        request_input_object (Dict): The dictionary format required to be for request input
    """
    # 千问API不支持批量请求，返回模拟对象
    print("Warning: Qwen API does not support batch requests. This function is not implemented for Qwen.")
    request_input_object = {
        "custom_id": f"qe-request-{id}",
        "method": "POST",
        "url": "/v1/chat/completions", 
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": f"{system_message}"}, 
                {"role": "user", "content": f"{prompt}"}
                ]
        }
    }
    return request_input_object