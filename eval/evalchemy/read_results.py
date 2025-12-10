
import json
import os

# 给我写一个脚本，给我读取 logs dir 下边的 results json 文件。

def read_results(logs_dir):
    results = {}
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"\n{'='*80}")
                print(f"File: {file_path}")
                print(f"{'='*80}")
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # 我需要你把里边每一个任务的结果给我 print 出来
                    if 'results' in data:
                        for task_name, task_data in data['results'].items():
                            print(f"\nTask: {task_name}")
                            print(f"{'-'*80}")
                            
                            # 打印基本统计信息
                            if 'num_total' in task_data:
                                print(f"  Total problems: {task_data['num_total']}")
                            if 'num_solved' in task_data:
                                print(f"  Solved: {task_data['num_solved']}")
                            if 'solved_avg' in task_data:
                                print(f"  Average solved: {task_data['solved_avg']}")
                            if 'accuracy' in task_data:
                                print(f"  Accuracy: {task_data['accuracy']:.4f} ({task_data['accuracy']*100:.2f}%)")
                            if 'accuracy_avg' in task_data:
                                print(f"  Average accuracy: {task_data['accuracy_avg']:.4f} ({task_data['accuracy_avg']*100:.2f}%)")
                            if 'accuracy_std_err' in task_data:
                                print(f"  Accuracy std error: {task_data['accuracy_std_err']}")
                            if 'num_repeat' in task_data:
                                print(f"  Number of repeats: {task_data['num_repeat']}")
                            
                            # 打印不同难度的准确率（如果存在）
                            if 'accuracy_easy_avg' in task_data:
                                print(f"  Easy accuracy: {task_data['accuracy_easy_avg']:.4f} ({task_data['accuracy_easy_avg']*100:.2f}%)")
                            if 'accuracy_medium_avg' in task_data:
                                print(f"  Medium accuracy: {task_data['accuracy_medium_avg']:.4f} ({task_data['accuracy_medium_avg']*100:.2f}%)")
                            if 'accuracy_hard_avg' in task_data:
                                print(f"  Hard accuracy: {task_data['accuracy_hard_avg']:.4f} ({task_data['accuracy_hard_avg']*100:.2f}%)")
                    
                    results[root] = data
    return results


logs_dir = "logs"
results = read_results(logs_dir)