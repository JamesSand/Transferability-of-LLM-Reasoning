import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import os
import re
from typing import Dict, Any, List
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_solution(solution_str: str) -> str:

    solution = re.search(r"#### (\\?-?[0-9\\.\\,]+)", solution_str)
    
    if solution is not None:
        final_solution = solution.group(0)
        final_solution = final_solution.split('#### ')[1].replace(',', '')
        return final_solution
    
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
    if boxed_match:
        return boxed_match.group(1).replace(',', '')
    
    cleaned = re.sub(r'\\[a-zA-Z]+\{|\}|\\', '', solution_str.strip())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if len(cleaned) < 50 and cleaned:
        return cleaned
        
    return solution_str.strip()

def load_math_rl_dataset(dataset_name: str = "ReasoningTransferability/math_rl_48k") -> Dict[str, Any]:
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name)
        logger.info(f"Successfully loaded dataset with {len(dataset['train'])} examples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def make_map_fn(split: str = "train", data_source: str = "math_rl_48k"):
    instruction_following = "Let's think step by step and output the final answer in the \\boxed{}."
    
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:

        question = example.get('question', '')
        answer = example.get('golden_answer', '')
        original_data_source = example.get('data_source', 'unknown')
        
        enhanced_question = question + ' ' + instruction_following
        
        try:
            solution = extract_solution(answer)
        except Exception as e:
            logger.warning(f"Failed to extract solution from example {idx}: {e}")
            solution = answer
        
        verl_data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": enhanced_question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'original_data_source': original_data_source,
                'original_question': question,
                'original_answer': answer,
                'question_length': len(question),
                'answer_length': len(answer)
            }
        }
        
        return verl_data
    
    return process_fn

def convert_to_verl_format(dataset, split: str = "train", data_source: str = "math_rl_48k") -> List[Dict[str, Any]]:
    logger.info(f"Converting dataset to VERL format...")
    
    data_split = dataset[split]
    
    map_fn = make_map_fn(split=split, data_source=data_source)
    
    verl_data = []
    failed_conversions = 0
    
    for idx, example in enumerate(data_split):
        try:
            verl_example = map_fn(example, idx)
            verl_data.append(verl_example)
            
            # Log progress
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(data_split)} examples")
                
        except Exception as e:
            logger.warning(f"Failed to convert example {idx}: {e}")
            failed_conversions += 1
            continue
    
    logger.info(f"Conversion completed. Successfully converted {len(verl_data)} examples")
    if failed_conversions > 0:
        logger.warning(f"Failed to convert {failed_conversions} examples")
    
    return verl_data

def validate_verl_format(verl_data: List[Dict[str, Any]]) -> bool:
    logger.info("Validating VERL format...")
    
    required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    
    for idx, example in enumerate(verl_data[:100]):  # Validate first 100 examples
        for field in required_fields:
            if field not in example:
                logger.error(f"Missing field '{field}' in example {idx}")
                return False
        
        if not isinstance(example["prompt"], list) or len(example["prompt"]) != 1:
            logger.error(f"Invalid prompt format in example {idx}")
            return False
            
        if example["prompt"][0].get("role") != "user":
            logger.error(f"Invalid prompt role in example {idx}")
            return False
            
        reward_model = example["reward_model"]
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            logger.error(f"Invalid reward_model format in example {idx}")
            return False
    
    logger.info("VERL format validation passed")
    return True

def save_verl_data(verl_data: List[Dict[str, Any]], output_path: str, format_type: str = "parquet") -> None:
    logger.info(f"Saving {len(verl_data)} examples to {output_path} in {format_type} format")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format_type == "parquet":
        df = pd.DataFrame(verl_data)
        df.to_parquet(output_path, compression='snappy', index=False, engine='pyarrow')
        
    elif format_type == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(verl_data, f, indent=2, ensure_ascii=False)
            
    elif format_type == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in verl_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Successfully saved file ({file_size_mb:.2f} MB)")

def create_verl_metadata(verl_data: List[Dict[str, Any]], metadata_path: str) -> None:
    logger.info(f"Creating VERL metadata file: {metadata_path}")
    
    original_sources = {}
    question_lengths = []
    answer_lengths = []
    solution_lengths = []
    
    for example in verl_data:
        extra_info = example.get('extra_info', {})
        original_source = extra_info.get('original_data_source', 'unknown')
        
        original_sources[original_source] = original_sources.get(original_source, 0) + 1
        question_lengths.append(extra_info.get('question_length', 0))
        answer_lengths.append(extra_info.get('answer_length', 0))
        solution_lengths.append(len(example['reward_model']['ground_truth']))
    
    metadata = {
        'verl_format_info': {
            'format_version': '1.0',
            'description': 'Math RL dataset converted to VERL format',
            'total_examples': len(verl_data),
            'data_source': verl_data[0]['data_source'] if verl_data else 'unknown',
            'ability': 'math',
            'reward_model_style': 'rule'
        },
        'original_dataset_info': {
            'name': 'ReasoningTransferability/math_rl_48k',
            'original_sources': original_sources,
            'source_distribution': {k: f"{v} ({v/len(verl_data)*100:.1f}%)" 
                                   for k, v in original_sources.items()}
        },
        'statistics': {
            'question_length': {
                'min': min(question_lengths) if question_lengths else 0,
                'max': max(question_lengths) if question_lengths else 0,
                'mean': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            },
            'answer_length': {
                'min': min(answer_lengths) if answer_lengths else 0,
                'max': max(answer_lengths) if answer_lengths else 0,
                'mean': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            },
            'solution_length': {
                'min': min(solution_lengths) if solution_lengths else 0,
                'max': max(solution_lengths) if solution_lengths else 0,
                'mean': sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0,
            }
        },
        'verl_format_schema': {
            'data_source': 'string - Name of the dataset for reward function indexing',
            'prompt': 'list[dict] - HuggingFace chat template format with role and content',
            'ability': 'string - Task category (math)',
            'reward_model': 'dict - Contains style and ground_truth for evaluation',
            'extra_info': 'dict - Additional information about the prompt'
        }
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info("VERL metadata file created successfully")

def analyze_solution_extraction(verl_data: List[Dict[str, Any]], output_path: str = None) -> Dict[str, Any]:
    logger.info("Analyzing solution extraction quality...")
    
    analysis = {
        'total_examples': len(verl_data),
        'extraction_methods': {
            'boxed_pattern': 0,
            'direct_answer': 0,
            'cleaned_text': 0,
            'fallback': 0
        },
        'solution_types': {
            'numeric': 0,
            'algebraic': 0,
            'text': 0,
            'empty': 0
        },
        'samples': {
            'successful': [],
            'problematic': []
        }
    }
    
    for i, example in enumerate(verl_data[:1000]):
        solution = example['reward_model']['ground_truth']
        original_answer = example['extra_info']['original_answer']
        
        if not solution or solution.isspace():
            analysis['solution_types']['empty'] += 1
            analysis['samples']['problematic'].append({
                'index': i,
                'issue': 'empty_solution',
                'original_answer': original_answer[:100],
                'extracted_solution': solution
            })
        elif re.match(r'^-?\d+(\.\d+)?$', solution.replace(',', '')):
            analysis['solution_types']['numeric'] += 1
        elif any(c in solution for c in ['x', 'y', 'z', '+', '-', '*', '/', '^']):
            analysis['solution_types']['algebraic'] += 1
        else:
            analysis['solution_types']['text'] += 1
        
        if '\\boxed{' in original_answer:
            analysis['extraction_methods']['boxed_pattern'] += 1
        elif '####' in original_answer:
            analysis['extraction_methods']['direct_answer'] += 1
        elif len(solution) < 20:
            analysis['extraction_methods']['cleaned_text'] += 1
        else:
            analysis['extraction_methods']['fallback'] += 1
            
        if len(analysis['samples']['successful']) < 10 and solution and len(solution) < 50:
            analysis['samples']['successful'].append({
                'index': i,
                'original_answer': original_answer[:100],
                'extracted_solution': solution
            })
    
    logger.info(f"Solution extraction analysis completed")
    logger.info(f"Solution types: {analysis['solution_types']}")
    logger.info(f"Extraction methods: {analysis['extraction_methods']}")
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis saved to {output_path}")
    
    return analysis

def main():
    dataset_name = "ReasoningTransferability/math_rl_48k"
    data_source = "math_rl_48k"  # VERL data source name
    split = "train"
    output_dir = "./processed_data"
    
    try:
        dataset = load_math_rl_dataset(dataset_name)
        
        verl_data = convert_to_verl_format(dataset, split=split, data_source=data_source)
        
        if not validate_verl_format(verl_data):
            raise ValueError("VERL format validation failed")
        
        save_verl_data(verl_data, f"{output_dir}/math_rl_48k_verl.parquet", "parquet")
        save_verl_data(verl_data, f"{output_dir}/math_rl_48k_verl.jsonl", "jsonl")
        
        create_verl_metadata(verl_data, f"{output_dir}/verl_metadata.json")
        
        analysis = analyze_solution_extraction(verl_data, f"{output_dir}/solution_extraction_analysis.json")
        
        sample_size = min(100, len(verl_data))
        sample_data = verl_data[:sample_size]
        save_verl_data(sample_data, f"{output_dir}/sample_{sample_size}_verl.json", "json")
        
        logger.info("VERL format conversion completed successfully!")
        logger.info(f"Files created:")
        logger.info(f"  - Main dataset (Parquet): {output_dir}/math_rl_48k_verl.parquet")
        logger.info(f"  - Main dataset (JSONL): {output_dir}/math_rl_48k_verl.jsonl")
        logger.info(f"  - Metadata: {output_dir}/verl_metadata.json")
        logger.info(f"  - Analysis: {output_dir}/solution_extraction_analysis.json")
        logger.info(f"  - Sample ({sample_size} examples): {output_dir}/sample_{sample_size}_verl.json")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    main()