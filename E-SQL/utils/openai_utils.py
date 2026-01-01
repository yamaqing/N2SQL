# 导入OpenAI库和类型提示模块
from openai import OpenAI
from typing import Dict
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('E-SQL.env')


def create_response(stage: str, prompt: str, model: str, max_tokens: int, temperature: float, top_p: float, n: int) -> Dict:
    """
    创建OpenAI聊天响应函数，根据不同阶段生成不同的系统提示

    参数:
        stage (str): 处理阶段，决定使用哪个系统提示模板
        prompt (str): 准备好的用户提示内容
        model (str): 用于生成聊天完成的LLM模型名称
        max_tokens (int): 聊天完成中可生成的最大标记数
        temperature (float): 采样温度，控制输出的随机性
        top_p (float): 核采样参数，控制解码时考虑的标记范围
        n (int): 每个输入消息生成的聊天完成数

    返回:
        response_object (Dict): 模型返回的响应对象
    """
    # 初始化客户端，支持千问模型(OpenAI兼容API)
    client = OpenAI(
        # 从环境变量中获取API密钥，支持OpenAI和千问模型
       api_key=os.getenv("API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"), 
    )

    # 根据不同阶段选择对应的系统提示内容
    if stage == "question_enrichment":
        # 问题丰富阶段：将用户问题与数据库项目关联，生成更清晰的问题
        system_content = "You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions and the evidence to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items."
    elif stage == "candidate_sql_generation":
        # SQL生成阶段：根据问题和数据库信息生成候选SQL查询
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "sql_refinement":
        # SQL优化阶段：优化和完善生成的SQL查询
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "schema_filtering":
        # 模式过滤阶段：筛选与问题相关的数据库表和列
        system_content = "You are an excellent data scientist. You can capture the link between a question and corresponding database and determine the useful database items (tables and columns) perfectly. Your objective is to analyze and understand the essence of the given question, corresponding database schema, database column descriptions, samples and evidence and then select the useful database items such as tables and columns. This database item filtering is essential for eliminating unnecessary information in the database so that corresponding structured query language (SQL) of the question can be generated correctly in later steps."
    else:
        # 如果阶段值无效，抛出 ValueError 异常
        raise ValueError("Wrong value for stage. It can only take following values: question_enrichment, candidate_sql_generation, sql_refinement or schema_filtering.")

    # 调用OpenAI API生成聊天响应
    response_object = client.chat.completions.create(
        model=model,  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_content},  # 系统提示
            {"role": "user", "content": prompt}  # 用户提示
        ],
        max_tokens=max_tokens,  # 最大生成标记数
        response_format={"type": "json_object"},  # 指定响应格式为JSON对象
        temperature=temperature,  # 采样温度
        top_p=top_p,  # 核采样参数
        n=n,  # 生成响应数
        presence_penalty=0.0,  # 存在惩罚
        frequency_penalty=0.0  # 频率惩罚
    )

    # 返回API响应对象
    return response_object


def upload_file_to_openai(file_path: str) -> Dict:
    """
    将文件上传到OpenAI用于批量处理

    参数:
        file_path (str): 要上传的文件路径
    返回:
        file_object (Dict): OpenAI返回的文件对象
    """
    # 初始化客户端，支持千问模型(OpenAI兼容API)
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        #base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )

    # 调用OpenAI API上传文件
    file_object = client.files.create(
        file=open(file_path, "rb"),  # 以二进制模式打开文件
        purpose="batch"  # 文件用途为批量处理
    )

    # 打印上传成功信息
    print("File is uploaded to AI")
    # 返回文件对象
    return file_object


def construct_request_input_object(prompt: str, id: int, model: str, system_message: str) -> Dict:
    """
    为数据集中的每个项目创建请求输入对象

    参数:
        prompt (str): 要提供给LLM的内容提示
        id (int): 请求的唯一标识
        model (str): LLM模型名称
        system_message (str): 系统消息的内容

    返回:
        request_input_object (Dict): 请求输入所需的字典格式
    """
    # 构建请求输入对象
    request_input_object = {
        "custom_id": f"qe-request-{id}",  # 自定义请求ID，用于跟踪
        "method": "POST",  # HTTP请求方法
        "url": "/v1/chat/completions",  # API端点路径
        "body": {
            "model": model,  # 指定模型
            "messages": [
                {"role": "system", "content": f"{system_message}"},  # 系统消息
                {"role": "user", "content": f"{prompt}"}  # 用户提示
            ]
        }
    }
    # 返回构建好的请求对象
    return request_input_object
