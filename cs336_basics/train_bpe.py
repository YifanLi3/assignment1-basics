import os
import regex
from collections import Counter
from .pretokenization_example import find_chunk_boundaries

# GPT-2 预分词正则表达式
GPT2_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 1. 初始化词表（256个字节）
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    if vocab_size < 256 + len(special_tokens):
        raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

    # 2. 读取文本
    with open(input_path, 'rb') as f:
        data = f.read()
    
    # 3. 预分词：将文本分割成单元（单词级别）
    text = data.decode('utf-8', errors='replace')
    
    # 使用GPT-2正则表达式进行预分词
    # 同时需要保护特殊token
    pretokens = []
    
    # 先找出所有特殊token的位置
    special_positions = []
    for special_token in special_tokens:
        start = 0
        while True:
            pos = text.find(special_token, start)
            if pos == -1:
                break
            special_positions.append((pos, pos + len(special_token), special_token))
            start = pos + 1
    special_positions.sort()
    
    # 将文本分成：普通文本（需要预分词）和特殊token（保持完整）
    current_pos = 0
    for sp_start, sp_end, sp_token in special_positions:
        # 处理特殊token之前的普通文本
        if current_pos < sp_start:
            normal_text = text[current_pos:sp_start]
            # 对普通文本进行GPT-2预分词
            for match in GPT2_PAT.finditer(normal_text):
                pretokens.append(match.group().encode('utf-8'))
        # 添加特殊token作为单独的预分词单元（用负数ID标记）
        current_pos = sp_end
    
    # 处理最后一个特殊token之后的文本
    if current_pos < len(text):
        normal_text = text[current_pos:]
        for match in GPT2_PAT.finditer(normal_text):
            pretokens.append(match.group().encode('utf-8'))
    
    # 4. 统计每个预分词单元出现的次数
    pretoken_counts = Counter(pretokens)

    # 5. 将每个预分词单元转换为token ID列表
    # word_to_tokens: 预分词单元 -> 对应的token ID列表
    # word_counts: 预分词单元 -> 出现次数
    word_tokens = {}  # bytes -> list of token IDs
    for word_bytes, count in pretoken_counts.items():
        word_tokens[word_bytes] = list(word_bytes)  # 初始时每个字节就是token ID
    
    # 6. 迭代合并
    num_merges = vocab_size - 256 - len(special_tokens)
    for merge_idx in range(num_merges):
        # a. 统计所有预分词单元中的字节对频率
        pair_counts = Counter()
        for word_bytes, tokens in word_tokens.items():
            word_count = pretoken_counts[word_bytes]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_counts[pair] += word_count
        
        if not pair_counts:
            break

        # b. 找到最频繁的字节对
        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        # Tie-breaking: 先按第一个 token 的字节序列，再按第二个 token 的字节序列，选择最大的
        best_pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))

        # Debug: index 29-35
        if 29 <= merge_idx <= 35:
            print(f"\n=== Merge {merge_idx} ===")
            print(f"Max count: {max_count}, candidates: {len(candidates)}")
            # 显示 top 5 pairs
            top5 = sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
            for p, c in top5:
                print(f"  {p}: {c} -> {vocab[p[0]]!r}+{vocab[p[1]]!r}")
            print(f"Selected: {best_pair} -> {vocab[best_pair[0]]!r}+{vocab[best_pair[1]]!r}")

        # c. 创建新的 token ID
        new_token_id = len(vocab)
        
        # d. 将合并后的字节加入词表
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # e. 记录合并历史
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # f. 更新所有预分词单元：用新 ID 替换所有该 pair
        for word_bytes in word_tokens:
            word_tokens[word_bytes] = merge_tokens(word_tokens[word_bytes], best_pair, new_token_id)

        if (merge_idx + 1) % 100 == 0:
            print(f'Completed {merge_idx + 1} merges')


    # 5. 添加特殊token到词表末尾
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    
    # 6. 返回vocab和merges
    return vocab, merges



def get_pair_counts(tokens: list[int]) -> Counter:
    pairs = Counter()
    for i in range(len(tokens) - 1):
        # 跳过包含负数ID的pairs（特殊token占位符）
        if tokens[i] < 0 or tokens[i+1] < 0:
            continue
        pair = (tokens[i], tokens[i+1])
        pairs[pair] += 1
    return pairs

def merge_tokens(tokens: list[int],
                 pair: tuple[int, int],
                 new_token_id: int) -> list[int]:
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(new_token_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens