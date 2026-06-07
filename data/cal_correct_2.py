import nltk
import json
import os
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import re
import numpy as np


def extract_answers_x(text):
    """提取符合规范的答案（FA/FB/FC）"""
    pattern = r'Answer: (F[A-C])'
    matches = re.findall(pattern, text) 
    if matches:
        answer = matches[0].upper()
        if answer in {'FA', 'FB', 'FC'}:
            return answer
    return None


def preprocess_text(text):
    """预处理文本，去除标点符号并转为小写"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def normalize_scores(scores):
    """将分数归一化到[0,1]区间"""
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)

def find_best_match(predict_text, candidates):
    """使用归一化的BLEU、ROUGE和余弦相似度匹配最接近的答案"""
    if not predict_text or not candidates:
        return None
    
    # 预处理预测文本
    predict_tokens = preprocess_text(predict_text)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    # 存储每个候选项的各项分数
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    cosine_scores = []
    
    # 计算余弦相似度
    vectorizer = TfidfVectorizer()
    all_texts = [predict_text] + candidates
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # 计算所有候选项的分数
    for idx, candidate in enumerate(candidates):
        # BLEU分数
        candidate_tokens = preprocess_text(candidate)
        bleu = sentence_bleu([candidate_tokens], predict_tokens, weights=(0.5, 0.5))
        bleu_scores.append(bleu)
        
        # ROUGE分数
        rouge_scores = scorer.score(predict_text.lower(), candidate.lower())
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        
        # 余弦相似度已经计算好了
        cosine_scores.append(cosine_similarities[idx])
    
    # 归一化所有分数
    norm_bleu = normalize_scores(np.array(bleu_scores))
    norm_rouge1 = normalize_scores(np.array(rouge1_scores))
    norm_rouge2 = normalize_scores(np.array(rouge2_scores))
    norm_cosine = normalize_scores(np.array(cosine_scores))
    
    # 计算综合得分
    weights = np.array([0.3, 0.2, 0.2, 0.3])  # BLEU, ROUGE-1, ROUGE-2, Cosine的权重
    combined_scores = (norm_bleu * weights[0] + 
                      norm_rouge1 * weights[1] + 
                      norm_rouge2 * weights[2] +
                      norm_cosine * weights[3])
    
    best_idx = np.argmax(combined_scores)
    max_score = combined_scores[best_idx]
    
    # 设置阈值，避免错误匹配
    if max_score < 0.3:
        return None
    
    return chr(65 + best_idx)


answer_array = [  
    ["A.Outside the railway operation zone.", "B.On the ballast.", "C.On the tracks."],
    ["A.Stationary.", "B.Movement that poses a threat", "C.Movement that does not pose a threat"],
    ["FA.safe.", "FB.potential threat.", "FC.serious threat."],
    ["FA.safe.", "FB.potential threat.", "FC.serious threat."]
]

def get_dataset_path(predictions_path):
    """获取对应的数据集路径"""
    offline_num = predictions_path.split('offline')[-1][0]
    dataset_path = f"/root/data1/TianYL/ZhangQY/LLaMA-Factory/data/offline{offline_num}.json"
    return dataset_path

def main():
    correct = 0
    total = 0
    incorrect_cases = []
    similarity_matched_cases = []  # 新增：记录通过相似度匹配的案例
    
    answer_path = "/root/data1/TianYL/ZhangQY/LLaMA-Factory/saves/Qwen2-VL-7B-Instruct/lora/eval_2025-02-22-13-32-21_offline8/generated_predictions.jsonl"
    type_path = "/root/data1/TianYL/ZhangQY/LLaMA-Factory/data/que_type_8.json"
    
    # 加载原始数据集以获取图片路径
    dataset_path = get_dataset_path(answer_path)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(type_path, 'r', encoding='utf-8') as file:
        cont = json.load(file)
        with open(answer_path, 'r', encoding='utf-8') as f:
            for idx, (line, type_id) in enumerate(zip(f, cont['type'])):
                type_id = int(type_id)
                entry = json.loads(line.strip())
                predict_answer = extract_answers_x(entry['predict'])
                label_answer = entry['label'].strip().upper()

                if type_id > len(answer_array):
                    print(f"类型超出范围: {type_id}")
                    continue

                current_answers = answer_array[type_id - 1]
                is_correct = False
                matched_option = None
                is_similarity_matched = False  # 新增：标记是否通过相似度匹配

                if predict_answer:
                    total += 1
                    is_correct = predict_answer == label_answer
                    matched_option = predict_answer
                else:
                    matched_option = find_best_match(entry['predict'], current_answers)
                    if matched_option:
                        # 将单字母答案转换为F+字母格式
                        matched_option = 'F' + matched_option
                        total += 1
                        is_correct = matched_option == label_answer
                        is_similarity_matched = True  # 标记为相似度匹配

                if is_correct:
                    correct += 1
                
                # 记录相似度匹配的案例
                if is_similarity_matched:
                    similarity_matched_cases.append({
                        'index': idx,
                        'predict_text': entry['predict'],
                        'matched_answer': matched_option,
                        'label': label_answer,
                        'is_correct': is_correct,
                        'type': type_id
                    })

    # 输出结果
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n总体统计:")
        print(f"正确率: {accuracy:.2f}% (正确数: {correct}, 总数: {total})")
        
        # 打印相似度匹配统计
        similarity_total = len(similarity_matched_cases)
        similarity_correct = sum(1 for case in similarity_matched_cases if case['is_correct'])
        print(f"通过相似度匹配得到答案的数量: {similarity_total}")
        print(f"其中正确的数量: {similarity_correct}")
    else:
        print("没有有效数据")

if __name__ == "__main__":
    main()

