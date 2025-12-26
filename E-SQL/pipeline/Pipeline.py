# 导入必要的模块和工具
import os
import json
# 导入提示构建相关的工具函数
from utils.prompt_utils import *
# 导入数据库操作相关的工具函数
from utils.db_utils import * 
# 导入OpenAI API调用工具函数
from utils.openai_utils import create_response
# 导入类型注解
from typing import Dict, List

class Pipeline():
    """
    Text-to-SQL系统的核心管道类，实现了将自然语言问题转换为SQL查询的完整流程。
    支持多种管道配置，包括候选SQL生成(CSG)、问题富集(QE)、SQL优化(SR)和模式过滤(SF)等阶段。
    """
    def __init__(self, args):
        """
        初始化管道类，设置各种运行参数和配置。
        
        参数:
            args: 包含管道配置参数的命名空间对象
        """
        # 运行模式属性
        self.mode = args.mode  # 运行模式（如'train'、'dev'、'test'）
        self.dataset_path = args.dataset_path  # 数据集路径

        # 管道属性
        self.pipeline_order = args.pipeline_order  # 管道阶段顺序配置

        # 模型属性
        self.model = args.model  # 使用的LLM模型名称
        self.temperature = args.temperature  # 模型生成温度参数
        self.top_p = args.top_p  # 模型生成top_p参数
        self.max_tokens = args.max_tokens  # 模型生成最大token数
        self.n = args.n  # 生成候选数量

        # 各阶段（富集、过滤、生成）属性
        self.enrichment_level = args.enrichment_level  # 问题富集级别
        self.elsn = args.enrichment_level_shot_number  # 问题富集阶段的few-shot样本数量
        self.efsse = args.enrichment_few_shot_schema_existance  # 问题富集阶段是否包含schema

        self.flsn = args.filtering_level_shot_number  # 模式过滤阶段的few-shot样本数量
        self.ffsse = args.filtering_few_shot_schema_existance  # 模式过滤阶段是否包含schema

        self.cfg = args.cfg  # 配置参数
        self.glsn = args.generation_level_shot_number  # SQL生成阶段的few-shot样本数量
        self.gfsse = args.generation_few_shot_schema_existance  # SQL生成阶段是否包含schema

        self.db_sample_limit = args.db_sample_limit  # 数据库样本提取限制
        self.rdn = args.relevant_description_number  # 相关描述数量

        self.seed = args.seed  # 随机种子

    def convert_message_content_to_dict(self, response_object: Dict) -> Dict:
        """
        将LLM响应对象的内容从JSON字符串转换为Python字典格式。
        
        参数:
            response_object (Dict): LLM返回的响应对象
            
        返回:
            response_object (Dict): 内容已转换为字典的响应对象
        """
        """
        The function gets a LLM response object, and then it converts the content of it to the Python object.

        Arguments:
            response_object (Dict): LLM response object
        Returns:
            response_object (Dict): Response object whose content changed to dictionary
        """

        response_object.choices[0].message.content = json.loads(response_object.choices[0].message.content)
        return response_object
    

    def forward_pipeline_CSG_SR(self, t2s_object: Dict) -> Dict[str, str]:
        """
        执行简单管道：仅包含候选SQL生成(CSG)和SQL优化(SR)两个阶段，跳过问题富集和模式过滤。
        
        参数:
            t2s_object (Dict): 包含问题信息的字典，如问题ID、数据库ID、问题文本、证据等
            
        返回:
            t2s_object_prediction (Dict): 包含处理后信息的字典，包括每个阶段的结果和总使用情况
        """
        """
        The function runs Candidate SQL Generation(CSG) and SQL Refinement(SR) modules respectively without any question enrichment or filtering stages.

        Arguments:
            t2s_object (Dict): Python dictionary that stores information about a question like q_id, db_id, question, evidence etc. 
        Returns:
            t2s_object_prediction (Dict):  Python dictionary that stores information about a question like q_id, db_id, question, evidence etc and also stores information after each stage
        """
        db_id = t2s_object["db_id"]
        q_id = t2s_object["question_id"]
        evidence = t2s_object["evidence"]
        question = t2s_object["question"]
        
        # 从环境变量获取BIRD-SQL数据集路径
        bird_sql_path = os.getenv('DB_PATH')
        # 构建数据库文件路径
        db_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/{db_id}.sqlite"
        # 构建数据库描述文件路径
        db_description_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/database_description"
        # 准备与问题相关的数据库描述
        db_descriptions = question_relevant_descriptions_prep(
            database_description_path=db_description_path, 
            question=question, 
            relevant_description_number=self.rdn
        )
        # 构建列含义文件路径
        database_column_meaning_path = bird_sql_path + f"/{self.mode}/column_meaning.json"
        # 准备数据库列含义信息
        db_column_meanings = db_column_meaning_prep(database_column_meaning_path, db_id)
        # 合并数据库描述和列含义
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # 提取原始数据库模式字典（表名到列名的映射）
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=db_path)

        # 标记未执行问题富集阶段
        t2s_object["question_enrichment"] = "No Question Enrichment"

        ### 第一阶段：候选SQL生成(CSG)
        # -- 使用原始问题
        # -- 使用原始数据库模式
        sql_generation_response_obj =  self.candidate_sql_generation_module(
            db_path=db_path, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            filtered_schema_dict=original_schema_dict, 
            db_descriptions=db_descriptions
        )
        try:
            # 提取生成的候选SQL
            possible_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            # 记录候选SQL生成结果
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "possible_sql": possible_sql,
                "exec_err": "",  # 初始化为空的执行错误
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,  # 提示tokens数量
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,  # 完成tokens数量
                "total_tokens": sql_generation_response_obj.usage.total_tokens,  # 总tokens数量
            }
            # 保存候选SQL
            t2s_object["possible_sql"] = possible_sql
            # 执行SQL并捕获可能的错误
            try:
                # 使用func_timeout设置30秒超时执行SQL
                possible_respose = func_timeout(30, execute_sql, args=(db_path, possible_sql))
            except FunctionTimedOut:
                # 处理超时错误
                t2s_object['candidate_sql_generation']["exec_err"] = "timeout"
            except Exception as e:
                # 处理其他执行错误
                t2s_object['candidate_sql_generation']["exec_err"] = str(e)
        except Exception as e:
            # 处理从SQL生成响应中提取内容时的错误
            logging.error(f"问题ID {q_id} 的SQL生成响应内容提取错误: {e}")
            # 记录错误信息并返回
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": "",
                "possible_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["candidate_sql_generation"]["error"] = f"{e}"
            return t2s_object
        
        ### 第二阶段：SQL优化(SR)
        # -- 使用原始问题
        # -- 使用原始数据库模式
        # -- 使用候选SQL
        # -- 使用从候选SQL中提取的可能条件进行增强
        # -- 使用候选SQL的执行错误
        # 获取候选SQL的执行错误
        exec_err = t2s_object['candidate_sql_generation']["exec_err"]
        # 调用SQL优化模块
        sql_generation_response_obj =  self.sql_refinement_module(
            db_path=db_path, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            possible_sql=possible_sql, 
            exec_err=exec_err, 
            filtered_schema_dict=original_schema_dict, 
            db_descriptions=db_descriptions
        )
        try:
            # 提取优化后的SQL
            predicted_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            # 记录SQL优化结果
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "predicted_sql": predicted_sql,  # 优化后的SQL
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            # 保存最终预测的SQL
            t2s_object["predicted_sql"] = predicted_sql
        except Exception as e:
            # 处理从SQL优化响应中提取内容时的错误
            logging.error(f"问题ID {q_id} 的SQL优化响应内容提取错误: {e}")
            # 记录错误信息并返回
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": "",
                "predicted_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["sql_refinement"]["error"] = f"{e}"
            return t2s_object

        # 计算并记录该问题处理的总token使用量
        t2s_object["total_usage"] = {
            "prompt_tokens": t2s_object['candidate_sql_generation']['prompt_tokens'] + t2s_object['sql_refinement']['prompt_tokens'],
            "completion_tokens": t2s_object['candidate_sql_generation']['completion_tokens'] + t2s_object['sql_refinement']['completion_tokens'],
            "total_tokens": t2s_object['candidate_sql_generation']['total_tokens'] + t2s_object['sql_refinement']['total_tokens']
        }

        t2s_object_prediction = t2s_object
        return t2s_object_prediction
    
    def forward_pipeline_CSG_QE_SR(self, t2s_object: Dict) -> Dict:
        """
        执行标准管道：包含候选SQL生成(CSG)、问题富集(QE)和SQL优化(SR)三个阶段，跳过模式过滤。
        
        参数:
            t2s_object (Dict): 包含问题信息的字典，如问题ID、数据库ID、问题文本、证据等
            
        返回:
            t2s_object_prediction (Dict): 包含处理后信息的字典，包括每个阶段的结果和总使用情况
        """
        """
        The function performs Candidate SQL Generation(CSG), Quesiton Enrichment(QE) and SQL Refinement(SR) modules respectively without filtering stages.
        
        Arguments:
            t2s_object (Dict): Python dictionary that stores information about a question like q_id, db_id, question, evidence etc. 
        Returns:
            t2s_object_prediction (Dict):  Python dictionary that stores information about a question like q_id, db_id, question, evidence etc and also stores information after each stage
        """
        db_id = t2s_object["db_id"]
        q_id = t2s_object["question_id"]
        evidence = t2s_object["evidence"]
        question = t2s_object["question"]
        
        bird_sql_path = os.getenv('DB_PATH')
        db_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/{db_id}.sqlite"
        db_description_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/database_description"
        db_descriptions = question_relevant_descriptions_prep(database_description_path=db_description_path, question=question, relevant_description_number=self.rdn)
        database_column_meaning_path = bird_sql_path + f"/{self.mode}/column_meaning.json"
        db_column_meanings = db_column_meaning_prep(database_column_meaning_path, db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # extracting original schema dictionary 
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=db_path)

        ### STAGE 1: Candidate SQL GENERATION
        # -- Original question is used
        # -- Original Schema is used 
        sql_generation_response_obj =  self.candidate_sql_generation_module(db_path=db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=original_schema_dict, db_descriptions=db_descriptions)
        try:
            possible_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "possible_sql": possible_sql,
                "exec_err": "",
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["possible_sql"] = possible_sql
            # execute SQL
            try:
                possible_respose = func_timeout(30, execute_sql, args=(db_path, possible_sql))
            except FunctionTimedOut:
                t2s_object['candidate_sql_generation']["exec_err"] = "timeout"
            except Exception as e:
                t2s_object['candidate_sql_generation']["exec_err"] = str(e)
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": "",
                "possible_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["candidate_sql_generation"]["error"] = f"{e}"
            return t2s_object
        
        # 从候选SQL中提取可能的条件
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=possible_sql)
        # 准备可能条件的文本表示
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)

        ### 第二阶段：问题富集(QE)
        # -- 使用原始问题
        # -- 使用原始数据库模式
        # -- 使用从候选SQL中提取的可能条件
        q_enrich_response_obj = self.question_enrichment_module(
            db_path=db_path, 
            q_id=q_id, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            possible_conditions=possible_conditions, 
            schema_dict=original_schema_dict, 
            db_descriptions=db_descriptions
        )
        try:
            # 提取富集后的问题
            enriched_question = q_enrich_response_obj.choices[0].message.content['enriched_question']
            # 提取富集推理过程
            enrichment_reasoning = q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning']
            # 记录问题富集结果
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": enrichment_reasoning,
                "enriched_question": enriched_question,
                "prompt_tokens": q_enrich_response_obj.usage.prompt_tokens,
                "completion_tokens": q_enrich_response_obj.usage.completion_tokens,
                "total_tokens": q_enrich_response_obj.usage.total_tokens,
            }
            # 将原始问题、推理过程和富集问题合并（experiment-24后添加）
            enriched_question = question + enrichment_reasoning + enriched_question
        except Exception as e:
            # 处理从问题富集响应中提取内容时的错误
            logging.error(f"问题ID {q_id} 的问题富集响应内容提取错误: {e}")
            # 记录错误信息并返回原始问题
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": "",
                "enriched_question": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["question_enrichment"]["error"] = f"{e}"
            # 如果富集失败，使用原始问题
            enriched_question = question
        
        ### 第三阶段：SQL优化(SR)
        # -- 使用富集后的问题
        # -- 使用原始数据库模式
        # -- 使用候选SQL
        # -- 使用从候选SQL中提取的可能条件进行增强
        # -- 使用候选SQL的执行错误
        # 获取候选SQL的执行错误
        exec_err = t2s_object['candidate_sql_generation']["exec_err"]
        # 调用SQL优化模块
        sql_generation_response_obj =  self.sql_refinement_module(
            db_path=db_path, 
            db_id=db_id, 
            question=enriched_question,  # 使用富集后的问题
            evidence=evidence, 
            possible_sql=possible_sql, 
            exec_err=exec_err, 
            filtered_schema_dict=original_schema_dict, 
            db_descriptions=db_descriptions
        )
        try:
            predicted_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "predicted_sql": predicted_sql,
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["predicted_sql"] = predicted_sql
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": "",
                "predicted_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["sql_refinement"]["error"] = f"{e}"
            return t2s_object

        # 计算并记录该问题处理的总token使用量（包括三个阶段）
        t2s_object["total_usage"] = {
            "prompt_tokens": t2s_object['candidate_sql_generation']['prompt_tokens'] + t2s_object['question_enrichment']['prompt_tokens'] + t2s_object['sql_refinement']['prompt_tokens'],
            "completion_tokens": t2s_object['candidate_sql_generation']['completion_tokens'] + t2s_object['question_enrichment']['completion_tokens'] + t2s_object['sql_refinement']['completion_tokens'],
            "total_tokens": t2s_object['candidate_sql_generation']['total_tokens'] + t2s_object['question_enrichment']['total_tokens'] + t2s_object['sql_refinement']['total_tokens']
        }

        t2s_object_prediction = t2s_object
        return t2s_object_prediction
    

    def forward_pipeline_SF_CSG_QE_SR(self, t2s_object: Dict) -> Dict:
        """
        执行完整管道：包含模式过滤(SF)、候选SQL生成(CSG)、问题富集(QE)和SQL优化(SR)四个阶段。
        
        参数:
            t2s_object (Dict): 包含问题信息的字典，如问题ID、数据库ID、问题文本、证据等
            
        返回:
            t2s_object_prediction (Dict): 包含处理后信息的字典，包括每个阶段的结果和总使用情况
        """
        """
        The function performs, Schema Filtering(SF) Candidate SQL Generation(CSG), Quesiton Enrichment(QE) and SQL Refinement(SR) modules respectively without filtering stages.
        
        Arguments:
            t2s_object (Dict): Python dictionary that stores information about a question like q_id, db_id, question, evidence etc. 
        Returns:
            t2s_object_prediction (Dict):  Python dictionary that stores information about a question like q_id, db_id, question, evidence etc and also stores information after each stage
        """
        db_id = t2s_object["db_id"]
        q_id = t2s_object["question_id"]
        evidence = t2s_object["evidence"]
        question = t2s_object["question"]
        
        bird_sql_path = os.getenv('DB_PATH')
        db_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/{db_id}.sqlite"
        db_description_path = bird_sql_path + f"/{self.mode}/{self.mode}_databases/{db_id}/database_description"
        db_descriptions = question_relevant_descriptions_prep(database_description_path=db_description_path, question=question, relevant_description_number=self.rdn)
        database_column_meaning_path = bird_sql_path + f"/{self.mode}/column_meaning.json"
        db_column_meanings = db_column_meaning_prep(database_column_meaning_path, db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # extracting original schema dictionary 
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=db_path)


        ### 第一阶段：数据库模式过滤(SF)
        # -- 使用原始问题
        # -- 使用原始数据库模式
        schema_filtering_response_obj = self.schema_filtering_module(
            db_path=db_path, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            schema_dict=original_schema_dict, 
            db_descriptions=db_descriptions
        )
        # print("schema_filtering_response_obj: \n", schema_filtering_response_obj)
        try:
            # 记录模式过滤结果
            t2s_object["schema_filtering"] = {
                "filtering_reasoning": schema_filtering_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "filtered_schema_dict": schema_filtering_response_obj.choices[0].message.content['tables_and_columns'],  # 过滤后的模式字典
                "prompt_tokens": schema_filtering_response_obj.usage.prompt_tokens,
                "completion_tokens": schema_filtering_response_obj.usage.completion_tokens,
                "total_tokens": schema_filtering_response_obj.usage.total_tokens,
            }
        except Exception as e:
            # 处理从模式过滤响应中提取内容时的错误
            logging.error(f"问题ID {q_id} 的模式过滤响应内容提取错误: {e}")
            # 记录错误信息并返回
            t2s_object["schema_filtering"] = f"{e}"
            return t2s_object

        ### 第一阶段.1：过滤后模式修正
        # 获取过滤后的模式字典
        filtered_schema_dict = schema_filtering_response_obj.choices[0].message.content['tables_and_columns']
        # 修正过滤后的模式（确保其与实际数据库一致）
        filtered_schema_dict, filtered_schema_problems = filtered_schema_correction(db_path=db_path, filtered_schema_dict=filtered_schema_dict) 
        # 记录模式修正结果
        t2s_object["schema_filtering_correction"] = {
            "filtered_schema_problems": filtered_schema_problems,  # 发现的问题
            "final_filtered_schema_dict": filtered_schema_dict  # 修正后的最终模式
        }

        # 生成修正后的模式的CREATE TABLE语句
        schema_statement = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        # 记录CREATE TABLE语句
        t2s_object["create_table_statement"] = schema_statement

        ### 第二阶段：候选SQL生成(CSG)
        # -- 使用原始问题
        # -- 使用过滤后的数据库模式
        sql_generation_response_obj =  self.candidate_sql_generation_module(
            db_path=db_path, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            filtered_schema_dict=filtered_schema_dict,  # 使用过滤后的模式
            db_descriptions=db_descriptions
        )
        try:
            possible_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "possible_sql": possible_sql,
                "exec_err": "",
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["possible_sql"] = possible_sql
            # execute SQL
            try:
                possible_respose = func_timeout(30, execute_sql, args=(db_path, possible_sql))
            except FunctionTimedOut:
                t2s_object['candidate_sql_generation']["exec_err"] = "timeout"
            except Exception as e:
                t2s_object['candidate_sql_generation']["exec_err"] = str(e)
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": "",
                "possible_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["candidate_sql_generation"]["error"] = f"{e}"
            return t2s_object
        
        # 从候选SQL中提取可能的条件
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=possible_sql)
        # 准备可能条件的文本表示
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)

        ### 第三阶段：问题富集(QE)
        # -- 使用原始问题
        # -- 使用过滤后的数据库模式
        # -- 使用从候选SQL中提取的可能条件
        q_enrich_response_obj = self.question_enrichment_module(
            db_path=db_path, 
            q_id=q_id, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            possible_conditions=possible_conditions, 
            schema_dict=filtered_schema_dict,  # 使用过滤后的模式
            db_descriptions=db_descriptions
        )
        try:
            enriched_question = q_enrich_response_obj.choices[0].message.content['enriched_question']
            enrichment_reasoning = q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning']
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "enriched_question": q_enrich_response_obj.choices[0].message.content['enriched_question'],
                "prompt_tokens": q_enrich_response_obj.usage.prompt_tokens,
                "completion_tokens": q_enrich_response_obj.usage.completion_tokens,
                "total_tokens": q_enrich_response_obj.usage.total_tokens,
            }
            enriched_question = question + enrichment_reasoning + enriched_question # This is added after experiment-24
        except Exception as e:
            logging.error(f"Error in reaching content from question enrichment response for question_id {q_id}: {e}")
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": "",
                "enriched_question": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["question_enrichment"]["error"] = f"{e}"
            enriched_question = question
        
        ### 第四阶段：SQL优化(SR)
        # -- 使用富集后的问题
        # -- 使用过滤后的数据库模式
        # -- 使用候选SQL
        # -- 使用从候选SQL中提取的可能条件进行增强
        # -- 使用候选SQL的执行错误
        # 获取候选SQL的执行错误
        exec_err = t2s_object['candidate_sql_generation']["exec_err"]
        # 调用SQL优化模块
        sql_generation_response_obj =  self.sql_refinement_module(
            db_path=db_path, 
            db_id=db_id, 
            question=enriched_question,  # 使用富集后的问题
            evidence=evidence, 
            possible_sql=possible_sql, 
            exec_err=exec_err, 
            filtered_schema_dict=filtered_schema_dict,  # 使用过滤后的模式
            db_descriptions=db_descriptions
        )
        try:
            predicted_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "predicted_sql": predicted_sql,
                "prompt_tokens": sql_generation_response_obj.usage.prompt_tokens,
                "completion_tokens": sql_generation_response_obj.usage.completion_tokens,
                "total_tokens": sql_generation_response_obj.usage.total_tokens,
            }
            t2s_object["predicted_sql"] = predicted_sql
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": "",
                "predicted_sql": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            t2s_object["sql_refinement"]["error"] = f"{e}"
            return t2s_object

        # 计算并记录该问题处理的总token使用量（包括四个阶段）
        t2s_object["total_usage"] = {
            "prompt_tokens": t2s_object['candidate_sql_generation']['prompt_tokens'] + t2s_object['question_enrichment']['prompt_tokens'] + t2s_object['sql_refinement']['prompt_tokens'],
            "completion_tokens": t2s_object['candidate_sql_generation']['completion_tokens'] + t2s_object['question_enrichment']['completion_tokens'] + t2s_object['sql_refinement']['completion_tokens'],
            "total_tokens": t2s_object['candidate_sql_generation']['total_tokens'] + t2s_object['question_enrichment']['total_tokens'] + t2s_object['sql_refinement']['total_tokens']
        }

        t2s_object_prediction = t2s_object
        return t2s_object_prediction

        
    def construct_question_enrichment_prompt(self, db_path: str, q_id: int, db_id: str, question: str, evidence: str, possible_conditions: str, schema_dict: Dict, db_descriptions: str) -> str:
        """
        构建问题富集阶段所需的提示(prompt)。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            q_id (int): 问题ID
            db_id (str): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            possible_conditions (str): 从候选SQL中提取的可能条件
            schema_dict (Dict[str, List[str]]): 数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            prompt (str): 构建好的问题富集提示
        """
        """
        The function constructs the prompt required for the question enrichment stage

        Arguments:
            db_path (str): path to database sqlite file
            q_id (int): question id
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            possible_conditions (str): Possible conditions extracted from the previously generated possible SQL for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            prompt (str): Question enrichment prompt
        """
        # 获取问题富集提示模板路径
        enrichment_template_path = os.path.join(os.getcwd(), "prompt_templates/question_enrichment_prompt_template.txt")
        # 提取提示模板
        question_enrichment_prompt_template = extract_question_enrichment_prompt_template(enrichment_template_path)
        # 获取few-shot示例数据路径
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        # 准备问题富集阶段的few-shot示例
        q_enrich_few_shot_examples = question_enrichment_few_shot_prep(
            few_shot_data_path, 
            q_id=q_id, 
            q_db_id=db_id, 
            level_shot_number=self.elsn, 
            schema_existance=self.efsse, 
            enrichment_level=self.enrichment_level, 
            mode=self.mode
        )
        # 提取与问题相关的数据库样本
        db_samples = extract_db_samples_enriched_bm25(
            question, 
            evidence, 
            db_path=db_path, 
            schema_dict=schema_dict, 
            sample_limit=self.db_sample_limit
        )
        # 生成数据库模式文本
        schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)
        # 填充提示模板，构建最终提示
        prompt = fill_question_enrichment_prompt_template(
            template=question_enrichment_prompt_template, 
            schema=schema, 
            db_samples=db_samples, 
            question=question, 
            possible_conditions=possible_conditions, 
            few_shot_examples=q_enrich_few_shot_examples, 
            evidence=evidence, 
            db_descriptions=db_descriptions
        )
        # print("question_enrichment_prompt: \n", prompt)
        return prompt
    
    def question_enrichment_module(self, db_path: str, q_id: int, db_id: str, question: str, evidence: str, possible_conditions: str, schema_dict: Dict, db_descriptions: str) -> Dict:
        """
        问题富集模块：使用LLM对自然语言问题进行富集。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            q_id (int): 问题ID
            db_id (str): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            possible_conditions (str): 从候选SQL中提取的可能条件
            schema_dict (Dict[str, List[str]]): 数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            response_object (Dict): LLM返回的响应对象
        """
        """
        The function enrich the given question using LLM.

        Arguments:
            db_path (str): path to database sqlite file
            q_id (int): question id
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            possible_conditions (str): possible conditions extracted from previously generated possible SQL for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        # 构建问题富集提示
        prompt = self.construct_question_enrichment_prompt(
            db_path=db_path, 
            q_id=q_id, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            possible_conditions=possible_conditions, 
            schema_dict=schema_dict, 
            db_descriptions=db_descriptions
        )
        # 调用LLM生成响应
        response_object = create_response(
            stage="question_enrichment", 
            prompt=prompt, 
            model=self.model, 
            max_tokens=self.max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            n=self.n
        )
        try:
            # 尝试将响应内容转换为字典
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            # 如果转换失败，返回原始响应
            return response_object

        return response_object
    
    def construct_candidate_sql_generation_prompt(self, db_path: str, db_id: int, question: str, evidence: str, filtered_schema_dict: Dict, db_descriptions: str)->str:
        """
        构建候选SQL生成阶段所需的提示(prompt)。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            db_id (int): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            filtered_schema_dict (Dict[str, List[str]]): 过滤后的数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            prompt (str): 构建好的候选SQL生成提示
        """
        """
        The function constructs the prompt required for the candidate SQL generation stage.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            prompt (str): prompt for SQL generation stage
        """
        # 获取SQL生成提示模板路径
        sql_generation_template_path =  os.path.join(os.getcwd(), "prompt_templates/candidate_sql_generation_prompt_template.txt")
        # 读取提示模板
        with open(sql_generation_template_path, 'r') as f:
            sql_generation_template = f.read()
            
        # 获取few-shot示例数据路径
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        # 准备SQL生成阶段的few-shot示例
        sql_generation_few_shot_examples = sql_generation_and_refinement_few_shot_prep(
            few_shot_data_path, 
            q_db_id=db_id, 
            level_shot_number=self.glsn, 
            schema_existance=self.gfsse, 
            mode=self.mode
        )
        # 提取与问题相关的数据库样本
        db_samples = extract_db_samples_enriched_bm25(
            question, 
            evidence, 
            db_path, 
            schema_dict=filtered_schema_dict, 
            sample_limit=self.db_sample_limit
        )
        # 生成过滤后的数据库模式文本
        filtered_schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        # 填充提示模板，构建最终提示
        prompt = fill_candidate_sql_prompt_template(
            template=sql_generation_template, 
            schema=filtered_schema, 
            db_samples=db_samples, 
            question=question, 
            few_shot_examples=sql_generation_few_shot_examples, 
            evidence=evidence, 
            db_descriptions=db_descriptions
        ) 
        # print("candidate_sql_prompt: \n", prompt)
        return prompt

    
    def construct_sql_refinement_prompt(self, db_path: str, db_id: int, question: str, evidence: str, possible_sql: str, exec_err: str, filtered_schema_dict: Dict, db_descriptions: str)->str:
        """
        构建SQL优化阶段所需的提示(prompt)。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            db_id (int): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            possible_sql (str): 之前生成的候选SQL
            exec_err (str): 候选SQL的执行错误
            filtered_schema_dict (Dict[str, List[str]]): 过滤后的数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            prompt (str): 构建好的SQL优化提示
        """
        """
        The function constructs the prompt required for the SQL refinement stage.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            possible_sql (str): Previously generated possible SQL for the question
            exec_err (str): Taken execution error when possible SQL is executed
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            prompt (str): prompt for SQL generation stage
        """
        # 获取SQL优化提示模板路径
        sql_generation_template_path =  os.path.join(os.getcwd(), "prompt_templates/sql_refinement_prompt_template.txt")
        # 读取提示模板
        with open(sql_generation_template_path, 'r') as f:
            sql_generation_template = f.read()
            
        # 获取few-shot示例数据路径
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        # 准备SQL优化阶段的few-shot示例
        sql_generation_few_shot_examples = sql_generation_and_refinement_few_shot_prep(
            few_shot_data_path, 
            q_db_id=db_id, 
            level_shot_number=self.glsn, 
            schema_existance=self.gfsse, 
            mode=self.mode
        )
        # 从候选SQL中提取可能的条件
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=possible_sql)
        # 准备可能条件的文本表示
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)
        # 生成过滤后的数据库模式文本
        filtered_schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        # 填充提示模板，构建最终提示
        prompt = fill_refinement_prompt_template(
            template=sql_generation_template, 
            schema=filtered_schema, 
            possible_conditions=possible_conditions, 
            question=question, 
            possible_sql=possible_sql, 
            exec_err=exec_err, 
            few_shot_examples=sql_generation_few_shot_examples, 
            evidence=evidence, 
            db_descriptions=db_descriptions
        ) 
        # print("refinement_prompt: \n", prompt)
        return prompt
    
    def construct_filtering_prompt(self, db_path: str, db_id: str, question: str, evidence: str, schema_dict: Dict, db_descriptions: str)->str:
        """
        构建数据库模式过滤阶段所需的提示(prompt)。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            db_id (str): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            schema_dict (Dict[str, List[str]]): 数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            prompt (str): 构建好的模式过滤提示
        """
        """
        The function constructs the prompt required for the database schema filtering stage

        Arguments:  
            db_path (str): The database sqlite file path.
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            prompt (str): prompt for database schema filtering stage
        """
        # 获取模式过滤提示模板路径
        schema_filtering_prompt_template_path =  os.path.join(os.getcwd(), "prompt_templates/schema_filter_prompt_template.txt")
        # 读取提示模板
        with open(schema_filtering_prompt_template_path, 'r') as f:
            schema_filtering_template = f.read()

        # 获取few-shot示例数据路径
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        # 准备模式过滤阶段的few-shot示例
        schema_filtering_few_shot_examples = schema_filtering_few_shot_prep(
            few_shot_data_path, 
            q_db_id=db_id, 
            level_shot_number=self.elsn, 
            schema_existance=self.efsse, 
            mode=self.mode
        )
        # 提取与问题相关的数据库样本
        db_samples = extract_db_samples_enriched_bm25(
            question, 
            evidence, 
            db_path=db_path, 
            schema_dict=schema_dict, 
            sample_limit=self.db_sample_limit
        )
        # 生成数据库模式文本
        schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)
        # 填充提示模板，构建最终提示
        prompt = fill_prompt_template(
            template=schema_filtering_template, 
            schema=schema, 
            db_samples=db_samples, 
            question=question, 
            few_shot_examples=schema_filtering_few_shot_examples, 
            evidence=evidence, 
            db_descriptions=db_descriptions
        )
        # print("\nSchema Filtering Prompt: \n", prompt)
    
        return prompt

    
    def candidate_sql_generation_module(self, db_path: str, db_id: int, question: str, evidence: str, filtered_schema_dict: Dict, db_descriptions: str):
        """
        候选SQL生成模块：使用LLM生成回答问题的候选SQL。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            db_id (int): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            filtered_schema_dict (Dict[str, List[str]]): 过滤后的数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            response_object (Dict): LLM返回的响应对象
        """
        """
        This function generates candidate SQL for answering the question.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        # 构建候选SQL生成提示
        prompt = self.construct_candidate_sql_generation_prompt(
            db_path=db_path, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            filtered_schema_dict=filtered_schema_dict, 
            db_descriptions=db_descriptions
        )
        # 调用LLM生成响应
        response_object = create_response(
            stage="candidate_sql_generation", 
            prompt=prompt, 
            model=self.model, 
            max_tokens=self.max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            n=self.n
        )
        try:
            # 尝试将响应内容转换为字典
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            # 如果转换失败，返回原始响应
            return response_object

        return response_object

    
    def sql_refinement_module(self, db_path: str, db_id: int, question: str, evidence: str, possible_sql: str, exec_err: str, filtered_schema_dict: Dict, db_descriptions: str):
        """
        SQL优化模块：使用LLM优化或重新生成SQL查询。
        利用候选SQL、从候选SQL中提取的可能条件以及执行错误来提高SQL质量。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            db_id (int): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            possible_sql (str): 之前生成的候选SQL
            exec_err (str): 候选SQL的执行错误
            filtered_schema_dict (Dict[str, List[str]]): 过滤后的数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            response_object (Dict): LLM返回的响应对象
        """
        """
        This function refines or re-generates a SQL query for answering the question.
        Possible SQL query, possible conditions generated from possible SQL query and execution error if it is exist are leveraged for better SQL refinement.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            possible_sql (str): Previously generated possible SQL query for the question
            exec_err (str): Taken execution error when possible SQL is executed 
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        # 构建SQL优化提示
        prompt = self.construct_sql_refinement_prompt(
            db_path=db_path, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            possible_sql=possible_sql, 
            exec_err=exec_err, 
            filtered_schema_dict=filtered_schema_dict, 
            db_descriptions=db_descriptions
        )
        # 调用LLM生成响应
        response_object = create_response(
            stage="sql_refinement", 
            prompt=prompt, 
            model=self.model, 
            max_tokens=self.max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            n=self.n
        )
        try:
            # 尝试将响应内容转换为字典
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            # 如果转换失败，返回原始响应
            return response_object

        return response_object
    

    def schema_filtering_module(self, db_path: str, db_id: str, question: str, evidence: str, schema_dict: Dict, db_descriptions: str):
        """
        模式过滤模块：通过消除不必要的表和列来过滤数据库模式。
        
        参数:
            db_path (str): 数据库SQLite文件路径
            db_id (str): 数据库ID（数据库名称）
            question (str): 自然语言问题
            evidence (str): 问题相关的证据
            schema_dict (Dict[str, List[str]]): 数据库模式字典
            db_descriptions (str): 与问题相关的数据库项（列）描述
            
        返回:
            response_object (Dict): LLM返回的响应对象
        """
        """
        The function filters the database schema by eliminating the unnecessary tables and columns

        Arguments:  
            db_path (str): The database sqlite file path.
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        # 构建模式过滤提示
        prompt = self.construct_filtering_prompt(
            db_path=db_path, 
            db_id=db_id, 
            question=question, 
            evidence=evidence, 
            schema_dict=schema_dict, 
            db_descriptions=db_descriptions
        )
        # 调用LLM生成响应
        response_object = create_response(
            stage="schema_filtering", 
            prompt=prompt, 
            model=self.model, 
            max_tokens=self.max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            n=self.n
        )
        try:
            # 尝试将响应内容转换为字典
            response_object = self.convert_message_content_to_dict(response_object)
        except:
            # 如果转换失败，返回原始响应
            return response_object

        return response_object
    
    
