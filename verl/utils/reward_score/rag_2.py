# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from generator_llms.local_inst import *

_tokenizer = SimpleTokenizer()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def answer_span_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    normalized_golden_answers = [normalize_answer(golden_answer) for golden_answer in golden_answers]
    if has_answers(normalized_prediction, normalized_golden_answers, _tokenizer, regex=False):
        score = 1
    return score


def extract_titles_and_texts(solution_str):
    """
    Extract titles and texts from information blocks, handling important documents
    and preserving their order. Properly handles duplicated documents.
    
    <important_info> tags only apply to the most recent <information> block before them,
    even if there are other tags or content in between them.
    
    If an <information> block doesn't have a corresponding <important_info> tag,
    ALL documents from that block are included.
    
    Returns a list of (title, text) tuples for each document.
    """
    try:
        # Extract all information blocks and important_info tags with their positions
        info_blocks = []
        important_infos = []
        
        # Find all information blocks with their positions
        info_pattern = re.compile(r'<information>(.*?)</information>', re.DOTALL)
        for match in info_pattern.finditer(solution_str):
            info_blocks.append({
                'position': match.start(),
                'content': match.group(1),
                'important_ids': None,  # Will be filled if there's a matching important_info
                'processed': False  # Track if this block has been processed
            })
        
        # Find all important_info tags with their positions
        important_pattern = re.compile(r'<important_info>(.*?)</important_info>', re.DOTALL)
        for match in important_pattern.finditer(solution_str):
            # Parse important doc IDs
            important_ids = []
            try:
                # Extract the content and clean it
                content = match.group(1).strip()
                
                # Extract all numbers from the content
                numbers = re.findall(r'\d+', content)
                # Convert all numbers to integers
                important_ids = [int(num) for num in numbers if int(num) in [1, 2, 3, 4, 5, 6, 7, 8]]
                
                # Remove duplicates while preserving order
                important_ids = list(dict.fromkeys(important_ids))
                
            except Exception as e:
                print(f"Warning: Error parsing important document IDs: {str(e)}")
                important_ids = []
            
            important_infos.append({
                'position': match.start(),
                'important_ids': important_ids
            })
        
        # Process each important_info tag and associate it with the closest unprocessed information block
        all_docs = []
        seen_docs = set()  # To track unique documents
        
        # First process all information blocks that have important_info tags
        for imp_info in important_infos:
            # Find the closest unprocessed information block that appears before this important_info
            closest_info = None
            min_distance = float('inf')
            
            for info_block in info_blocks:
                if not info_block['processed'] and info_block['position'] < imp_info['position']:
                    # Only consider information blocks that are before this important_info
                    # and don't have any other information blocks between them
                    has_info_between = False
                    for other_info in info_blocks:
                        if (other_info['position'] > info_block['position'] and 
                            other_info['position'] < imp_info['position']):
                            has_info_between = True
                            break
                    
                    if not has_info_between:
                        distance = imp_info['position'] - info_block['position']
                        if distance < min_distance:
                            min_distance = distance
                            closest_info = info_block
            
            # If we found a matching information block, process it
            if closest_info:
                closest_info['processed'] = True  # Mark as processed
                info_content = closest_info['content']
                important_ids = imp_info['important_ids']
                
                docs_in_block = []
                try:
                    # Extract individual documents from the info block
                    # Basic pattern that captures:
                    # - Document number
                    # - Title (with or without quotes)
                    # - Content
                    # Handles:
                    # - Optional spaces between Doc and number
                    # - Optional spaces around Title:
                    # - Case insensitive
                    doc_pattern = re.compile(
                        r'Doc\s*(\d+)\s*\(\s*Title\s*:\s*"?([^")]+)"?\)\s*(.*?)(?=Doc\s*\d+\s*\(\s*Title\s*:|$)',
                        re.DOTALL | re.IGNORECASE
                    )
                    
                    for match in doc_pattern.finditer(info_content):
                        try:
                            doc_id = int(match.group(1))
                            title = match.group(2).strip()
                            text = match.group(3).strip()
                            # print(f"Debug: Parsed document - ID: {doc_id}, Title: {title}")
                            docs_in_block.append((doc_id, title, text))
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Error parsing document: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Warning: Error extracting documents from info block: {str(e)}")
                    continue
                
                # Filter by important_ids
                try:
                    # print(f"Debug: Filtering documents. Important IDs: {important_ids}")
                    # print(f"Debug: Available document IDs: {[doc_id for doc_id, _, _ in docs_in_block]}")
                    filtered_docs = [(title, text) for doc_id, title, text in docs_in_block 
                                    if doc_id in important_ids]
                    # print(f"Debug: Filtered documents count: {len(filtered_docs)}")
                except Exception as e:
                    print(f"Warning: Error filtering documents: {str(e)}")
                    filtered_docs = []
                
                # Add unique documents to the result
                for title, text in filtered_docs:
                    if text not in seen_docs:
                        seen_docs.add(text)
                        all_docs.append((title, text))
        
        # Then process all remaining unprocessed information blocks (those without important_info tags)
        for info_block in info_blocks:
            if not info_block['processed']:
                info_content = info_block['content']
                try:
                    # Extract individual documents from the info block
                    doc_pattern = re.compile(
                        r'Doc\s*(\d+)\s*\(\s*Title\s*:\s*"?([^")]+)"?\)\s*(.*?)(?=Doc\s*\d+\s*\(\s*Title\s*:|$)',
                        re.DOTALL | re.IGNORECASE
                    )
                    
                    for match in doc_pattern.finditer(info_content):
                        try:
                            doc_id = int(match.group(1))
                            title = match.group(2).strip()
                            text = match.group(3).strip()
                            
                            # Add all documents from unprocessed blocks
                            if text not in seen_docs:
                                seen_docs.add(text)
                                all_docs.append((title, text))
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Error parsing document: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Warning: Error extracting documents from info block: {str(e)}")
                    continue
        
        return all_docs
        
    except Exception as e:
        print(f"Warning: Error in extract_titles_and_texts: {str(e)}")
        return []  # Return empty list if any unexpected error occurs



###### For MIRAGE USE ######
# def extract_titles_and_texts(solution_str):
#     """
#     Extract titles and texts from information blocks, handling important documents
#     and preserving their order.
    
#     Special rules:
#     1. Always include ALL documents from the first information block (the one following the question block)
#     2. For all other information blocks, only extract documents if they have an associated <important_info> tag
    
#     Returns a list of (title, text) tuples for each document.
#     """
#     try:
#         # Extract all information blocks and important_info tags with their positions
#         info_blocks = []
#         important_infos = []
        
#         # Find the question block to identify the first information block
#         question_match = re.search(r'<question>(.*?)</question>', solution_str, re.DOTALL)
#         question_end_pos = question_match.end() if question_match else 0
        
#         # Find all information blocks with their positions
#         info_pattern = re.compile(r'<information>(.*?)</information>', re.DOTALL)
#         for match in info_pattern.finditer(solution_str):
#             info_blocks.append({
#                 'position': match.start(),
#                 'content': match.group(1),
#                 'important_ids': None,  # Will be filled if there's a matching important_info
#                 'processed': False,  # Track if this block has been processed
#                 'is_first_after_question': match.start() > question_end_pos  # Is this the first info block after question?
#             })
        
#         # Sort information blocks by position to identify the first one after the question
#         info_blocks.sort(key=lambda x: x['position'])
        
#         # Mark the first block after the question
#         first_after_question_found = False
#         for block in info_blocks:
#             if block['position'] > question_end_pos and not first_after_question_found:
#                 block['is_first_after_question'] = True
#                 first_after_question_found = True
#             else:
#                 block['is_first_after_question'] = False
        
#         # Find all important_info tags with their positions
#         important_pattern = re.compile(r'<important_info>(.*?)</important_info>', re.DOTALL)
#         for match in important_pattern.finditer(solution_str):
#             # Parse important doc IDs
#             important_ids = []
#             try:
#                 # Extract the content and clean it
#                 content = match.group(1).strip()
                
#                 # Extract all numbers from the content
#                 numbers = re.findall(r'\d+', content)
#                 # Convert all numbers to integers
#                 important_ids = [int(num) for num in numbers if int(num) in range(1, 20)]  # Expanded range
                
#                 # Remove duplicates while preserving order
#                 important_ids = list(dict.fromkeys(important_ids))
                
#             except Exception as e:
#                 print(f"Warning: Error parsing important document IDs: {str(e)}")
#                 important_ids = []
            
#             important_infos.append({
#                 'position': match.start(),
#                 'important_ids': important_ids
#             })
        
#         # Process each important_info tag and associate it with the closest preceding information block
#         all_docs = []
#         seen_docs = set()  # To track unique documents
        
#         # Process the first information block after the question (if found)
#         for info_block in info_blocks:
#             if info_block['is_first_after_question']:
#                 info_block['processed'] = True  # Mark as processed
#                 info_content = info_block['content']
                
#                 # Extract all documents from this first block
#                 doc_pattern = re.compile(
#                     r'Doc\s*(\d+)\s*\(Title:\s*([^)]+)\)\s*(.*?)(?=Doc\s*\d+\s*\(Title:|$)',
#                     re.DOTALL
#                 )
                
#                 for match in doc_pattern.finditer(info_content):
#                     try:
#                         doc_id = int(match.group(1))
#                         title = match.group(2).strip()
#                         text = match.group(3).strip()
                        
#                         doc_key = (title, text)
#                         if doc_key not in seen_docs:
#                             seen_docs.add(doc_key)
#                             all_docs.append((title, text))
#                     except (IndexError, ValueError) as e:
#                         print(f"Warning: Error parsing document: {str(e)}")
#                         continue
#                 break  # Only process the first block after question
        
#         # Process information blocks that have important_info tags
#         for imp_info in important_infos:
#             # Find the closest preceding information block
#             closest_info = None
#             min_distance = float('inf')
            
#             for info_block in info_blocks:
#                 if not info_block['processed'] and info_block['position'] < imp_info['position']:
#                     # Calculate distance between info block and important_info tag
#                     distance = imp_info['position'] - info_block['position']
                    
#                     # Find the closest information block
#                     if distance < min_distance:
#                         min_distance = distance
#                         closest_info = info_block
            
#             # If we found a matching information block, process it
#             if closest_info:
#                 closest_info['processed'] = True  # Mark as processed
#                 info_content = closest_info['content']
#                 important_ids = imp_info['important_ids']
                
#                 # Extract documents from this block
#                 doc_pattern = re.compile(
#                     r'Doc\s*(\d+)\s*\(Title:\s*([^)]+)\)\s*(.*?)(?=Doc\s*\d+\s*\(Title:|$)',
#                     re.DOTALL
#                 )
                
#                 docs_in_block = []
#                 for match in doc_pattern.finditer(info_content):
#                     try:
#                         doc_id = int(match.group(1))
#                         title = match.group(2).strip()
#                         text = match.group(3).strip()
#                         docs_in_block.append((doc_id, title, text))
#                     except (IndexError, ValueError) as e:
#                         print(f"Warning: Error parsing document: {str(e)}")
#                         continue
                
#                 # Filter by important_ids
#                 filtered_docs = [(title, text) for doc_id, title, text in docs_in_block 
#                                 if doc_id in important_ids]
                
#                 # Add unique documents to the result
#                 for title, text in filtered_docs:
#                     doc_key = (title, text)
#                     if doc_key not in seen_docs:
#                         seen_docs.add(doc_key)
#                         all_docs.append((title, text))
        
#         # Do NOT process remaining information blocks without important_info tags
#         # (except for the first one after question, which was already processed)
        
#         return all_docs
        
    except Exception as e:
        print(f"Warning: Error in extract_titles_and_texts: {str(e)}")
        return []  # Return empty list if any unexpected error occurs
    
    

def check_answer_correct(answer, golden_answers, model="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"):
    answer_context_score = answer_span_check(
        prediction=answer,
        golden_answers=golden_answers
    )
    if answer_context_score == 0:
        answer_context_score = 1 if check_if_response_is_correct_llm(
            response=answer,
            gold_answers=golden_answers,
            model=model
        ) else 0
    return answer_context_score


def compute_score_rag(solution_str, ground_truth, zeroshot_answers, data_source, use_utility_score=True, use_generation_score=True):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        zeroshot_answers: dictionary containing cached zeroshot answers
    """
    
    utility_score = 0
    generation_score = 0
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist()
    
    # Get documents with titles, handling important documents
    response_str = solution_str.split("Now, start the loop with the following question and initial searched results:")[1]
    docs = extract_titles_and_texts(solution_str=response_str)
    
    # Build context with unique documents
    seen_docs = set()
    doc_id = 1
    context_with_info = ""
    for title, text in docs:
        doc_key = (title, text)
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            context_with_info += f"Doc {doc_id} (Title: {title})\n{text}\n\n"
            doc_id += 1
        
    if use_utility_score:
        if question in zeroshot_answers[data_source]:
            answer_zeroshot = zeroshot_answers[data_source][question]['answer']
            answer_zeroshot_score = zeroshot_answers[data_source][question]['score']
        # answer_zeroshot_score = check_answer_correct(answer=answer_zeroshot, golden_answers=golden_answers)
        else:
            print(f"[Warning] No zeroshot answer found for question: {question}")
            answer_zeroshot = generate_answer_zero_shot(prompt=question)
            answer_zeroshot_score = check_answer_correct(answer=answer_zeroshot, golden_answers=golden_answers)
    
    answer_context = generate_answer(prompt=question, context=context_with_info)
    answer_context_score = check_answer_correct(answer=answer_context, golden_answers=golden_answers)
    generation_score = answer_context_score
        
    if use_utility_score:
        utility_score = answer_context_score - answer_zeroshot_score
    
    if use_generation_score and use_utility_score:
        score = utility_score + generation_score
    elif use_generation_score:
        score = generation_score
    elif use_utility_score:
        score = utility_score
    else:
        raise ValueError("use_generation_score and use_utility_score cannot both be False")
    
    do_print = random.randint(1, 16) == 1
       
    if use_utility_score:
        if do_print:
            print(f"--------------------------------")
            print(f"Question: {question}")
            print(f"Golden answers: {ground_truth['target']}")
            print(f"Extracted answer: {answer_context}")
            print(f"Extracted zeroshot (RAG) answer: {answer_zeroshot}")
            print(f"Answer context score: {answer_context_score}")
            print(f"Answer zeroshot (RAG) score: {answer_zeroshot_score}")
            print(f"Utility score: {utility_score}")
            print(f"Generation score: {generation_score}")
            print(f"Extracted doc_info: {context_with_info}")
            print(f"Response string: {response_str}")
            
    else:
        if do_print:
            print(f"--------------------------------")
            print(f"Question: {question}")
            print(f"Golden answers: {ground_truth['target']}")
            print(f"Extracted answer: {answer_context}")
            print(f"Generation score: {generation_score}")
            print(f"Extracted doc_info: {context_with_info}")
            print(f"Response string: {response_str}")
    
    if use_utility_score:
        return score, answer_zeroshot, answer_zeroshot_score
    else:
        return score, None, None


def output_sequence(solution_str, ground_truth):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        zeroshot_answers: dictionary containing cached zeroshot answers
    """
    
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist() if isinstance(ground_truth['target'], list) else [ground_truth['target']]
    
    # Get documents with titles, handling important documents
    try:
        response_str = solution_str.split("Now, start the loop with the following question and initial searched results:")[1]
        docs = extract_titles_and_texts(solution_str=response_str)
        # Build context with unique documents
        seen_docs = set()
        doc_id = 1
        context_with_info = ""
        for title, text in docs:
            doc_key = (title, text)
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                context_with_info += f"Doc {doc_id} (Title: {title})\n{text}\n\n"
                doc_id += 1
    except Exception as e:
        try:
            response_str = solution_str.split("Options:")[1]
            docs = extract_titles_and_texts(solution_str=response_str)
            # Build context with unique documents
            seen_docs = set()
            doc_id = 1
            context_with_info = ""
            for title, text in docs:
                doc_key = (title, text)
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    context_with_info += f"Doc {doc_id} (Title: {title})\n{text}\n\n"
                    doc_id += 1
        except Exception as e:
            print(f"Warning: Error in extract_titles_and_texts: {solution_str}")
            response_str = solution_str
            context_with_info = solution_str
        
    do_print = random.randint(1, 16) == 1
        
    if do_print:
        print(f"--------------------------------")
        print(f"Question: {question}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted doc_info: {context_with_info}")
        print(f"Solution string: {solution_str}")
    
    return question, golden_answers, context_with_info, response_str
    