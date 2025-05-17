import streamlit as st
import pandas as pd
import spacy
import json
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
import os
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from itertools import islice

matplotlib.use('Agg')  # バックエンドをAggに設定
import japanize_matplotlib  # 日本語フォントのサポート

# タイトルとサイドバーの設定
st.set_page_config(page_title="LAT35 on the web (simple version)", layout="wide")
st.title("LAT35 on the web")
st.sidebar.header("setting...")

# 言語モデルの選択肢
language_models = {
    "日本語": ["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
    "英語": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
    "韓国語": ["ko_core_news_sm", "ko_core_news_md", "ko_core_news_lg"],
    "中国語": ["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg"],
    "フランス語": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"]
}

# 言語選択
selected_language = st.sidebar.selectbox(
    "言語を選択",
    list(language_models.keys()),
    index=0
)

# 選択された言語のモデルを表示
selected_model = st.sidebar.selectbox(
    "言語モデルを選択",
    language_models[selected_language],
    index=0
)

# 言語ごとの品詞マッピング
pos_mappings = {
    "日本語": {
        "名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ", "副詞": "ADV",
        "助詞": "ADP", "助動詞": "AUX", "記号": "PUNCT", "空白": "SPACE",
        "接続詞": "CCONJ", "感動詞": "INTJ"
    },
    "英語": {
        "名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ", "副詞": "ADV",
        "前置詞": "ADP", "助動詞": "AUX", "記号": "PUNCT", "空白": "SPACE",
        "接続詞": "CCONJ", "感嘆詞": "INTJ"
    },
    "韓国語": {
        "名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ", "副詞": "ADV",
        "助詞": "ADP", "助動詞": "AUX", "記号": "PUNCT", "空白": "SPACE",
        "接続詞": "CCONJ", "感動詞": "INTJ"
    },
    "中国語": {
        "名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ", "副詞": "ADV",
        "介詞": "ADP", "助動詞": "AUX", "記号": "PUNCT", "空白": "SPACE",
        "連詞": "CCONJ", "感嘆詞": "INTJ"
    },
    "フランス語": {
        "名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ", "副詞": "ADV",
        "前置詞": "ADP", "助動詞": "AUX", "記号": "PUNCT", "空白": "SPACE",
        "接続詞": "CCONJ", "間投詞": "INTJ"
    }
}

# 言語ごとのデフォルト除外品詞
default_exclude_pos = {
    "日本語": ["助詞", "助動詞", "記号", "空白"],
    "英語": ["前置詞", "助動詞", "記号", "空白"],
    "韓国語": ["助詞", "助動詞", "記号", "空白"],
    "中国語": ["介詞", "助動詞", "記号", "空白"],
    "フランス語": ["前置詞", "助動詞", "記号", "空白"]
}

# 言語ごとのデフォルトストップワード
default_stop_words = {
    "日本語": "です,ます,ございます,はい,えー,あの,その,この",
    "英語": "the,a,an,and,or,but,is,are,was,were,be,been,being,have,has,had,do,does,did,will,would,shall,should,can,could,may,might,must",
    "韓国語": "이,그,저,이것,그것,저것,이거,그거,저거,이런,그런,저런,하다,있다,되다,없다,않다",
    "中国語": "的,了,和,是,在,有,就,不,也,这,那,我,你,他,她,它,们",
    "フランス語": "le,la,les,un,une,des,du,de,et,ou,mais,donc,car,est,sont,était,étaient,être,avoir,faire,aller"
}

# 言語ごとのフォント設定
font_settings = {
    "日本語": ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic',
               'Noto Sans CJK JP'],
    "英語": ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Noto Sans'],
    "韓国語": ['Malgun Gothic', 'Gulim', 'Dotum', 'Batang', 'Noto Sans CJK KR'],
    "中国語": ['SimHei', 'SimSun', 'NSimSun', 'FangSong', 'Noto Sans CJK SC'],
    "フランス語": ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Noto Sans']
}


# spaCyの言語モデルをロード
@st.cache_resource
def load_nlp_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"{model_name}モデルをダウンロードしています...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)


# 選択されたモデルをロード
nlp = load_nlp_model(selected_model)

# フォント設定を適用
plt.rcParams['font.sans-serif'] = font_settings[selected_language]

# CSVファイルのアップロード
uploaded_file = st.sidebar.file_uploader("upload Transcript(CSV format) to server", type=["csv"])

# 分析設定
analysis_options = st.sidebar.multiselect(
    "分析項目を選択",
    ["頻出単語分析", "発言者別分析", "発言パターン分析", "単語共起ネットワーク", "時系列分析",
     "インタラクションパターン分析", "教師・児童生徒発言量分析", "注目語累積分析"],
    default=["頻出単語分析", "発言者別分析", "インタラクションパターン分析", "教師・児童生徒発言量分析",
             "注目語累積分析"]
)

# 除外する品詞
exclude_pos = st.sidebar.multiselect(
    "除外する品詞",
    list(pos_mappings[selected_language].keys()),
    default=default_exclude_pos[selected_language]
)

# 除外する単語
stop_words = st.sidebar.text_area("除外する単語（カンマ区切り）", default_stop_words[selected_language])
stop_words = [word.strip() for word in stop_words.split(",")]

# 選択された言語の品詞マッピングを使用
pos_mapping = pos_mappings[selected_language]

# 除外する品詞のspaCy形式への変換
exclude_pos_spacy = [pos_mapping.get(pos, pos) for pos in exclude_pos]


def analyze_text(df):
    # 全テキストを結合
    all_text = " ".join(df["発言内容"].astype(str).tolist())

    # 形態素解析
    doc = nlp(all_text)

    # 単語と品詞のリストを作成（除外する品詞と単語を除く）
    words = [token.text for token in doc if
             token.pos_ not in exclude_pos_spacy and token.text.strip() and token.text not in stop_words]

    # 発言者ごとのテキスト
    speaker_texts = {}
    for speaker in df["発言者"].unique():
        speaker_df = df[df["発言者"] == speaker]
        speaker_text = " ".join(speaker_df["発言内容"].astype(str).tolist())
        speaker_texts[speaker] = speaker_text

    # 発言者ごとの形態素解析
    speaker_docs = {}
    for speaker, text in speaker_texts.items():
        speaker_docs[speaker] = nlp(text)

    return doc, words, speaker_texts, speaker_docs


def save_figure(fig, filename):
    # 保存ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    return filepath


def save_analysis_results(results):
    # 結果をJSONで保存
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "analysis_results.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return filepath


# n連結パターンを抽出する関数
def extract_n_grams(sequence, n):
    """
    シーケンスからn連結パターンを抽出する
    """
    n_grams = []
    for i in range(len(sequence) - n + 1):
        n_gram = tuple(sequence[i:i + n])
        n_grams.append(n_gram)
    return n_grams


# パターンの出現位置を特定する関数
def find_pattern_positions(sequence, pattern):
    """
    シーケンス内のパターン出現位置を全て見つける
    """
    positions = []
    pattern_length = len(pattern)
    for i in range(len(sequence) - pattern_length + 1):
        if tuple(sequence[i:i + pattern_length]) == pattern:
            positions.append(i)
    return positions


# 発言パターンの時系列可視化関数
def visualize_interaction_patterns(df, n_gram_size=2, top_n=10):
    """
    発言パターンを時系列で可視化する
    """
    speaker_sequence = df["発言者"].tolist()

    # n連結パターンの抽出
    n_grams = extract_n_grams(speaker_sequence, n_gram_size)

    # パターンの頻度カウント
    pattern_counts = Counter(n_grams)
    top_patterns = pattern_counts.most_common(top_n)

    # 結果を返す辞書の初期化
    result = {
        "top_patterns": {" → ".join(pattern): count for pattern, count in top_patterns},
        "pattern_positions": {}
    }

    # 時系列可視化のための準備
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams['font.sans-serif'] = font_settings[selected_language]

    # 各発言者に色を割り当て
    speakers = sorted(set(speaker_sequence))
    colors = plt.cm.tab20(np.linspace(0, 1, len(speakers)))
    speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}

    # 発言順序を時系列でプロット
    for i, speaker in enumerate(speaker_sequence):
        ax.scatter(i, 0, color=speaker_colors[speaker], s=100, alpha=0.7)

    # 凡例の作成
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=speaker_colors[speaker], markersize=10,
                                  label=speaker) for speaker in speakers]
    ax.legend(handles=legend_elements, loc='upper right')

    # 上位パターンの可視化
    pattern_y_positions = {}
    for i, (pattern, count) in enumerate(top_patterns, 1):
        # パターンの出現位置を特定
        positions = find_pattern_positions(speaker_sequence, pattern)
        result["pattern_positions"][" → ".join(pattern)] = positions

        # パターンごとに異なるY座標を割り当て
        y_pos = i * 0.5
        pattern_y_positions[pattern] = y_pos

        # パターンの各出現位置に印をつける
        for pos in positions:
            # パターンの範囲を強調表示
            ax.plot([pos, pos + n_gram_size - 1], [y_pos, y_pos],
                    linewidth=2, alpha=0.7,
                    color=plt.cm.tab10(i % 10))

            # パターンの開始点と終了点を強調
            ax.scatter([pos, pos + n_gram_size - 1], [y_pos, y_pos],
                       s=80, alpha=0.8,
                       color=plt.cm.tab10(i % 10))

    # グラフの設定
    ax.set_xlabel('発言順序')
    ax.set_yticks([0] + [pattern_y_positions[pattern] for pattern, _ in top_patterns])
    ax.set_yticklabels(['発言順序'] + [f"{' → '.join(pattern)} (出現回数: {count})" for pattern, count in top_patterns])
    ax.set_title(f'発言パターン分析 ({n_gram_size}連結)')
    ax.grid(True, alpha=0.3)

    return fig, result


# 発言パターンの遷移ネットワーク可視化関数
def visualize_pattern_network(df, n_gram_size=2, min_count=2):
    """
    発言パターンの遷移ネットワークを可視化する
    """
    speaker_sequence = df["発言者"].tolist()

    # n連結パターンの抽出
    n_grams = extract_n_grams(speaker_sequence, n_gram_size)

    # パターンの頻度カウント
    pattern_counts = Counter(n_grams)

    # 最小出現回数でフィルタリング
    filtered_patterns = {pattern: count for pattern, count in pattern_counts.items() if count >= min_count}

    # グラフの作成
    G = nx.DiGraph()

    # ノードの追加（パターンの先頭n-1要素と末尾n-1要素）
    for pattern in filtered_patterns:
        prefix = pattern[:-1]
        suffix = pattern[1:]

        if len(prefix) > 0:
            G.add_node(" → ".join(prefix))
        if len(suffix) > 0:
            G.add_node(" → ".join(suffix))

    # エッジの追加
    for pattern, count in filtered_patterns.items():
        prefix = pattern[:-1]
        suffix = pattern[1:]

        if len(prefix) > 0 and len(suffix) > 0:
            G.add_edge(" → ".join(prefix), " → ".join(suffix), weight=count)

    # 孤立したノードを削除
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    if G.number_of_nodes() > 0:
        # グラフの描画
        fig, ax = plt.subplots(figsize=(14, 12))
        plt.rcParams['font.sans-serif'] = font_settings[selected_language]
        pos = nx.spring_layout(G, seed=42)

        # エッジの太さを重みに比例させる
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_edge_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [1 + 3 * (weight / max_edge_weight) for weight in edge_weights]

        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue", alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.6,
                               arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", ax=ax)

        # エッジラベルの描画
        edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

        plt.title(f"{n_gram_size}連結パターンの遷移ネットワーク")
        plt.axis("off")

        return fig, {
            "nodes": list(G.nodes()),
            "edges": {f"{u} → {v}": weight for u, v, weight in G.edges(data="weight")}
        }
    else:
        return None, {"nodes": [], "edges": {}}


# 発言パターンのヒートマップ可視化関数
def visualize_pattern_heatmap(df, n_gram_size=2, top_n=20):
    """
    発言パターンの出現頻度をヒートマップで可視化する
    """
    speaker_sequence = df["発言者"].tolist()

    # n連結パターンの抽出
    n_grams = extract_n_grams(speaker_sequence, n_gram_size)

    # パターンの頻度カウント
    pattern_counts = Counter(n_grams)
    top_patterns = pattern_counts.most_common(top_n)

    if not top_patterns:
        return None, {}

    # ヒートマップ用のデータ作成
    patterns = [" → ".join(pattern) for pattern, _ in top_patterns]
    counts = [count for _, count in top_patterns]

    # ヒートマップの描画
    fig, ax = plt.subplots(figsize=(12, len(patterns) * 0.4 + 2))
    plt.rcParams['font.sans-serif'] = font_settings[selected_language]

    # カラーマップの作成（青から赤へのグラデーション）
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#4575b4", "#ffffbf", "#d73027"])

    # ヒートマップの描画
    sns.heatmap(
        pd.DataFrame(counts, index=patterns, columns=["出現回数"]),
        cmap=cmap,
        annot=True,
        fmt="d",
        linewidths=0.5,
        ax=ax
    )

    plt.title(f"上位{len(patterns)}個の{n_gram_size}連結パターン出現頻度")
    plt.tight_layout()

    return fig, {pattern: count for pattern, count in zip(patterns, counts)}


# 発言パターンの時間的分布可視化関数
def visualize_pattern_distribution(df, n_gram_size=2, top_n=5):
    """
    上位n個のパターンの時間的分布を可視化する
    """
    speaker_sequence = df["発言者"].tolist()

    # n連結パターンの抽出
    n_grams = extract_n_grams(speaker_sequence, n_gram_size)

    # パターンの頻度カウント
    pattern_counts = Counter(n_grams)
    top_patterns = pattern_counts.most_common(top_n)

    if not top_patterns:
        return None, {}

    # 時間的分布の可視化
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams['font.sans-serif'] = font_settings[selected_language]

    # 各パターンの出現位置を記録
    pattern_positions = {}

    # 各パターンの出現位置をプロット
    for i, (pattern, count) in enumerate(top_patterns):
        positions = find_pattern_positions(speaker_sequence, pattern)
        pattern_positions[" → ".join(pattern)] = positions

        # 各位置にマーカーを配置
        ax.scatter(positions, [i] * len(positions),
                   s=100, alpha=0.7,
                   color=plt.cm.tab10(i % 10),
                   label=f"{' → '.join(pattern)} ({count}回)")

    # グラフの設定
    ax.set_xlabel('発言順序')
    ax.set_yticks(range(len(top_patterns)))
    ax.set_yticklabels([f"{' → '.join(pattern)}" for pattern, _ in top_patterns])
    ax.set_title(f'上位{len(top_patterns)}個の{n_gram_size}連結パターンの時間的分布')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    return fig, pattern_positions


# 教師と児童生徒の発言量を可視化する関数
def visualize_teacher_student_utterances(df, teacher_keywords=None):
    """
    教師と児童生徒の発言量を棒グラフで可視化する
    教師の発言は下向き、児童生徒の発言は上向きに表示
    """
    if teacher_keywords is None:
        teacher_keywords = ["教師", "先生", "T", "Teacher"]

    # 発言者が教師かどうかを判定
    df["is_teacher"] = df["発言者"].apply(lambda x: any(keyword in x for keyword in teacher_keywords))

    # 発言長を計算
    df["発言長"] = df["発言内容"].str.len()

    # 教師と児童生徒のデータを分離
    teacher_df = df[df["is_teacher"]]
    student_df = df[~df["is_teacher"]]

    # 最大発言長を取得（教師と児童生徒それぞれ）
    max_teacher_length = teacher_df["発言長"].max() if not teacher_df.empty else 0
    max_student_length = student_df["発言長"].max() if not student_df.empty else 0

    # Y軸の最大値を設定（最大発言長 + 10）
    y_max = max(max_teacher_length, max_student_length) + 10

    # グラフの作成
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams['font.sans-serif'] = font_settings[selected_language]

    # 全発言番号のリスト
    all_utterance_numbers = df["発言番号"].tolist()

    # 教師の発言を下向きに表示
    if not teacher_df.empty:
        teacher_utterances = []
        teacher_lengths = []

        for num in all_utterance_numbers:
            teacher_utterance = teacher_df[teacher_df["発言番号"] == num]
            if not teacher_utterance.empty:
                teacher_utterances.append(num)
                teacher_lengths.append(-teacher_utterance["発言長"].values[0])  # マイナス値で表示
            else:
                teacher_utterances.append(num)
                teacher_lengths.append(0)

        ax.bar(teacher_utterances, teacher_lengths, color='blue', alpha=0.7, label='教師')

    # 児童生徒の発言を上向きに表示
    if not student_df.empty:
        student_utterances = []
        student_lengths = []

        for num in all_utterance_numbers:
            student_utterance = student_df[student_df["発言番号"] == num]
            if not student_utterance.empty:
                student_utterances.append(num)
                student_lengths.append(student_utterance["発言長"].values[0])
            else:
                student_utterances.append(num)
                student_lengths.append(0)

        ax.bar(student_utterances, student_lengths, color='orange', alpha=0.7, label='児童生徒')

    # グラフの設定
    ax.set_xlabel('発言番号')
    ax.set_ylabel('発言量（文字数）')
    ax.set_title('教師と児童生徒の発言量')
    ax.set_ylim(-y_max, y_max)  # Y軸の範囲を対称に設定
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # X軸を強調
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y軸のラベルを正の値で表示
    yticks = ax.get_yticks()
    ax.set_yticklabels([str(abs(int(y))) for y in yticks])

    return fig


# 注目する語の累積相対度数を可視化する関数
def visualize_word_cumulative_frequency(df, target_words):
    """
    注目する語の累積相対度数をグラフで可視化する
    """
    if not target_words:
        return None

    # 最大10語に制限
    target_words = target_words[:10]

    # 発言内容を形態素解析
    utterances = df["発言内容"].tolist()
    utterance_numbers = df["発言番号"].tolist()

    # 各単語の出現位置と累積カウントを記録
    word_occurrences = {word: [] for word in target_words}
    word_cumulative_counts = {word: [] for word in target_words}

    # 各発言を分析
    for i, utterance in enumerate(utterances):
        doc = nlp(utterance)
        utterance_number = utterance_numbers[i]

        # 発言内の単語をチェック
        utterance_words = [token.text.lower() for token in doc]

        for word in target_words:
            # 現在の累積カウント
            current_count = word_cumulative_counts[word][-1] if word_cumulative_counts[word] else 0

            # 単語が発言内に含まれているかチェック
            if word.lower() in utterance_words:
                # 出現回数をカウント
                count = utterance_words.count(word.lower())
                word_occurrences[word].append((utterance_number, count))
                word_cumulative_counts[word].append(current_count + count)
            else:
                # 出現しない場合は前回の累積値を維持
                word_cumulative_counts[word].append(current_count)

    # グラフの作成
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams['font.sans-serif'] = font_settings[selected_language]

    # 色のリスト
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_words)))

    # 各単語の累積相対度数をプロット
    for i, word in enumerate(target_words):
        if not word_cumulative_counts[word]:
            continue

        # 最大値を取得して正規化
        max_count = word_cumulative_counts[word][-1]
        if max_count == 0:
            continue

        # 正規化された累積カウント
        normalized_counts = [count / max_count for count in word_cumulative_counts[word]]

        # 正規化された発言番号
        normalized_numbers = [num / max(utterance_numbers) for num in utterance_numbers[:len(normalized_counts)]]

        # ステップ状のグラフを描画（垂直に上昇）
        ax.step(normalized_numbers, normalized_counts, where='post',
                label=f"{word} (合計: {max_count}回)",
                color=colors[i], linewidth=2, alpha=0.8)

        # 出現位置にマーカーを追加
        for occurrence in word_occurrences[word]:
            utterance_num, count = occurrence
            # 正規化された位置
            x_pos = utterance_num / max(utterance_numbers)
            y_pos = word_cumulative_counts[word][utterance_numbers.index(utterance_num)] / max_count
            ax.scatter(x_pos, y_pos, color=colors[i], s=50, alpha=0.8)

    # グラフの設定
    ax.set_xlabel('発言番号（相対値）')
    ax.set_ylabel('出現回数（相対値）')
    ax.set_title('注目語の累積相対度数')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    return fig


if uploaded_file is not None:
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)

    # カラム名の確認と修正
    required_columns = ["発言番号", "発言者", "発言内容"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"CSVファイルには次のカラムが必要です: {', '.join(required_columns)}")
    else:
        st.write("データプレビュー:")
        st.dataframe(df.head())

        # テキスト分析の実行
        doc, words, speaker_texts, speaker_docs = analyze_text(df)

        # 分析結果を保存するための辞書
        analysis_results = {}

        # タブの作成
        tabs = st.tabs(["分析サマリー"] + [option for option in analysis_options])

        with tabs[0]:
            st.header("分析サマリー")
            st.write(f"総発言数: {len(df)}")
            st.write(f"発言者数: {df['発言者'].nunique()}")
            st.write(f"総単語数: {len(words)}")
            st.write(f"使用言語モデル: {selected_model}")

            # 発言者ごとの発言数
            speaker_counts = df["発言者"].value_counts()
            st.subheader("発言者ごとの発言数")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.rcParams['font.sans-serif'] = font_settings[selected_language]
            speaker_counts.plot(kind="bar", ax=ax)
            plt.title("発言者ごとの発言数")
            plt.ylabel("発言数")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # 図の保存
            save_path = save_figure(fig, "speaker_counts.png")
            st.success(f"発言者ごとの発言数グラフを保存しました: {save_path}")

            # 分析結果に追加
            analysis_results["発言者ごとの発言数"] = speaker_counts.to_dict()
            analysis_results["使用言語モデル"] = selected_model

        tab_index = 1

        # 頻出単語分析
        if "頻出単語分析" in analysis_options:
            with tabs[tab_index]:
                st.header("頻出単語分析")

                # 頻出単語のカウント
                word_counts = Counter(words)
                most_common = word_counts.most_common(30)

                # 頻出単語のグラフ
                st.subheader("頻出単語トップ30")
                fig, ax = plt.subplots(figsize=(12, 8))
                plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                words_df = pd.DataFrame(most_common, columns=["単語", "出現回数"])
                sns.barplot(x="出現回数", y="単語", data=words_df, ax=ax)
                plt.title("頻出単語トップ30")
                plt.tight_layout()
                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "frequent_words.png")
                st.success(f"頻出単語グラフを保存しました: {save_path}")

                # ワードクラウド
                st.subheader("WordCloud")
                try:
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color="white",
                        font_path="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf" if os.path.exists(
                            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf") else None,
                        max_words=100
                    ).generate(" ".join(words))

                    fig, ax = plt.subplots(figsize=(12, 8))
                    plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

                    # 図の保存
                    save_path = save_figure(fig, "wordcloud.png")
                    st.success(f"ワードクラウドを保存しました: {save_path}")
                except Exception as e:
                    st.warning(f"ワードクラウドの生成中にエラーが発生しました: {e}")
                    st.info("選択した言語によっては、適切なフォントが必要な場合があります。")

                # 分析結果に追加
                analysis_results["頻出単語"] = dict(most_common)

            tab_index += 1

        # 発言者別分析
        if "発言者別分析" in analysis_options:
            with tabs[tab_index]:
                st.header("発言者別分析")

                # 発言者ごとの特徴的な単語
                speaker_word_counts = {}
                for speaker, doc in speaker_docs.items():
                    speaker_words = [token.text for token in doc if
                                     token.pos_ not in exclude_pos_spacy and token.text.strip() and token.text not in stop_words]
                    speaker_word_counts[speaker] = Counter(speaker_words).most_common(10)

                # 発言者ごとの特徴的な単語のグラフ
                for speaker, counts in speaker_word_counts.items():
                    st.subheader(f"{speaker}の特徴的な単語")
                    if counts:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                        words_df = pd.DataFrame(counts, columns=["単語", "出現回数"])
                        sns.barplot(x="出現回数", y="単語", data=words_df, ax=ax)
                        plt.title(f"{speaker}の特徴的な単語")
                        plt.tight_layout()
                        st.pyplot(fig)

                        # 図の保存
                        save_path = save_figure(fig, f"speaker_{speaker}_words.png")
                        st.success(f"{speaker}の特徴的な単語グラフを保存しました: {save_path}")
                    else:
                        st.write("分析可能な単語がありません")

                # 発言者ごとの発言長さ
                speaker_lengths = df.groupby("発言者")["発言内容"].apply(lambda x: x.str.len().mean())
                st.subheader("発言者ごとの平均発言長")
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                speaker_lengths.plot(kind="bar", ax=ax)
                plt.title("発言者ごとの平均発言長（文字数）")
                plt.ylabel("平均文字数")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "speaker_lengths.png")
                st.success(f"発言者ごとの平均発言長グラフを保存しました: {save_path}")

                # 分析結果に追加
                analysis_results["発言者ごとの特徴的な単語"] = {speaker: dict(counts) for speaker, counts in
                                                                speaker_word_counts.items()}
                analysis_results["発言者ごとの平均発言長"] = speaker_lengths.to_dict()

            tab_index += 1

        # 発言パターン分析
        if "発言パターン分析" in analysis_options:
            with tabs[tab_index]:
                st.header("発言パターン分析")

                # 発言の順序を分析
                st.subheader("発言の順序パターン")

                # 発言順序の可視化
                speaker_sequence = df["発言者"].tolist()
                transitions = {}

                for i in range(len(speaker_sequence) - 1):
                    current = speaker_sequence[i]
                    next_speaker = speaker_sequence[i + 1]

                    if current not in transitions:
                        transitions[current] = {}

                    if next_speaker not in transitions[current]:
                        transitions[current][next_speaker] = 0

                    transitions[current][next_speaker] += 1

                # 遷移グラフの作成
                G = nx.DiGraph()

                for speaker in df["発言者"].unique():
                    G.add_node(speaker)

                for source, targets in transitions.items():
                    for target, weight in targets.items():
                        G.add_edge(source, target, weight=weight)

                # グラフの描画
                fig, ax = plt.subplots(figsize=(12, 10))
                plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                pos = nx.spring_layout(G, seed=42)

                # ノードの描画
                nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue", ax=ax)

                # エッジの描画（太さを重みに比例させる）
                edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
                max_weight = max(edge_weights) if edge_weights else 1
                edge_widths = [1 + 5 * (weight / max_weight) for weight in edge_weights]

                nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray",
                                       arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax)

                # ラベルの描画
                nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", ax=ax)

                # エッジラベルの描画
                edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

                plt.title("発言の遷移パターン")
                plt.axis("off")
                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "speaker_transitions.png")
                st.success(f"発言の遷移パターングラフを保存しました: {save_path}")

                # 分析結果に追加
                analysis_results["発言遷移パターン"] = {source: {target: weight for target, weight in targets.items()}
                                                        for source, targets in transitions.items()}

            tab_index += 1

        # 単語共起ネットワーク
        if "単語共起ネットワーク" in analysis_options:
            with tabs[tab_index]:
                st.header("単語共起ネットワーク")

                # 共起ネットワークの作成
                st.subheader("単語の共起関係")

                # 文ごとに分割して共起関係を分析
                sentences = []
                for text in df["発言内容"]:
                    doc = nlp(text)
                    for sent in doc.sents:
                        sentences.append(sent)

                # 共起関係の計算
                word_pairs = {}

                for sent in sentences:
                    words_in_sent = [token.text for token in sent if
                                     token.pos_ not in exclude_pos_spacy and token.text.strip() and token.text not in stop_words]

                    for i, word1 in enumerate(words_in_sent):
                        for word2 in words_in_sent[i + 1:]:
                            pair = tuple(sorted([word1, word2]))

                            if pair not in word_pairs:
                                word_pairs[pair] = 0

                            word_pairs[pair] += 1

                # 出現頻度でフィルタリング
                min_count = st.slider("最小共起回数", min_value=1, max_value=10, value=2)
                filtered_pairs = {pair: count for pair, count in word_pairs.items() if count >= min_count}

                # グラフの作成
                G_cooccurrence = nx.Graph()

                # 単語の出現回数を計算
                word_counts = Counter(words)

                # ノードの追加（出現回数の多い上位50単語）
                top_words = [word for word, _ in word_counts.most_common(50)]
                for word in top_words:
                    G_cooccurrence.add_node(word, count=word_counts[word])

                # エッジの追加
                for (word1, word2), count in filtered_pairs.items():
                    if word1 in top_words and word2 in top_words:
                        G_cooccurrence.add_edge(word1, word2, weight=count)

                # 孤立したノードを削除
                isolated_nodes = list(nx.isolates(G_cooccurrence))
                G_cooccurrence.remove_nodes_from(isolated_nodes)

                if G_cooccurrence.number_of_nodes() > 0:
                    # グラフの描画
                    fig, ax = plt.subplots(figsize=(14, 12))
                    plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                    pos = nx.spring_layout(G_cooccurrence, seed=42)

                    # ノードサイズを出現回数に比例させる
                    node_sizes = [300 * G_cooccurrence.nodes[node]["count"] / max(word_counts.values()) for node in
                                  G_cooccurrence.nodes]

                    # エッジの太さを共起回数に比例させる
                    edge_weights = [G_cooccurrence[u][v]["weight"] for u, v in G_cooccurrence.edges()]
                    max_edge_weight = max(edge_weights) if edge_weights else 1
                    edge_widths = [1 + 3 * (weight / max_edge_weight) for weight in edge_weights]

                    nx.draw_networkx_nodes(G_cooccurrence, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8,
                                           ax=ax)
                    nx.draw_networkx_edges(G_cooccurrence, pos, width=edge_widths, edge_color="gray", alpha=0.5, ax=ax)
                    nx.draw_networkx_labels(G_cooccurrence, pos, font_size=10, font_family="sans-serif", ax=ax)

                    plt.title("単語共起ネットワーク")
                    plt.axis("off")
                    st.pyplot(fig)

                    # 図の保存
                    save_path = save_figure(fig, "word_cooccurrence.png")
                    st.success(f"単語共起ネットワークを保存しました: {save_path}")

                    # 分析結果に追加
                    analysis_results["単語共起関係"] = {f"{word1}-{word2}": count for (word1, word2), count in
                                                        filtered_pairs.items()
                                                        if word1 in top_words and word2 in top_words}
                else:
                    st.warning("共起関係が見つかりませんでした。最小共起回数を下げてみてください。")

            tab_index += 1

        # 時系列分析
        if "時系列分析" in analysis_options:
            with tabs[tab_index]:
                st.header("時系列分析")

                # 発言番号に基づく時系列分析
                st.subheader("発言の時系列パターン")

                # 発言番号ごとの発言長さ
                df["発言長"] = df["発言内容"].str.len()

                fig, ax = plt.subplots(figsize=(14, 6))
                plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                plt.plot(df["発言番号"], df["発言長"], marker="o", linestyle="-", alpha=0.7)
                plt.title("発言の時系列パターン（発言長）")
                plt.xlabel("発言番号")
                plt.ylabel("発言長（文字数）")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "utterance_length_time_series.png")
                st.success(f"発言長の時系列グラフを保存しました: {save_path}")

                # 発言者ごとの時系列パターン
                st.subheader("発言者ごとの時系列パターン")

                # 発言者ごとに色分け
                fig, ax = plt.subplots(figsize=(14, 8))
                plt.rcParams['font.sans-serif'] = font_settings[selected_language]
                speakers = df["発言者"].unique()

                for i, speaker in enumerate(speakers):
                    speaker_df = df[df["発言者"] == speaker]
                    plt.scatter(speaker_df["発言番号"], speaker_df["発言長"],
                                label=speaker, alpha=0.7, s=50)

                plt.title("発言者ごとの時系列パターン")
                plt.xlabel("発言番号")
                plt.ylabel("発言長（文字数）")
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "speaker_time_series.png")
                st.success(f"発言者ごとの時系列グラフを保存しました: {save_path}")

                # 分析結果に追加
                analysis_results["時系列パターン"] = {
                    "発言番号": df["発言番号"].tolist(),
                    "発言長": df["発言長"].tolist(),
                    "発言者": df["発言者"].tolist()
                }

            tab_index += 1

        # インタラクションパターン分析
        if "インタラクションパターン分析" in analysis_options:
            with tabs[tab_index]:
                st.header("インタラクションパターン分析")

                # n連結の設定
                n_gram_sizes = st.multiselect(
                    "分析する連結数を選択",
                    [2, 3, 4, 5],
                    default=[2, 3]
                )

                # 各n連結サイズごとの分析
                for n_gram_size in n_gram_sizes:
                    st.subheader(f"{n_gram_size}連結パターン分析")

                    # 1. 発言パターンの時系列可視化
                    st.write(f"#### {n_gram_size}連結パターンの時系列可視化")
                    top_n = st.slider(f"表示する上位パターン数（{n_gram_size}連結）",
                                      min_value=5, max_value=20, value=10, key=f"top_n_{n_gram_size}")

                    fig, pattern_results = visualize_interaction_patterns(df, n_gram_size=n_gram_size, top_n=top_n)
                    st.pyplot(fig)

                    # 図の保存
                    save_path = save_figure(fig, f"interaction_patterns_{n_gram_size}.png")
                    st.success(f"{n_gram_size}連結パターンの時系列可視化を保存しました: {save_path}")

                    # 2. パターンの出現頻度ヒートマップ
                    st.write(f"#### {n_gram_size}連結パターンの出現頻度")

                    fig, heatmap_results = visualize_pattern_heatmap(df, n_gram_size=n_gram_size, top_n=top_n)
                    if fig:
                        st.pyplot(fig)

                        # 図の保存
                        save_path = save_figure(fig, f"pattern_heatmap_{n_gram_size}.png")
                        st.success(f"{n_gram_size}連結パターンの出現頻度ヒートマップを保存しました: {save_path}")
                    else:
                        st.warning(f"{n_gram_size}連結パターンが見つかりませんでした。")

                    # 3. パターンの時間的分布
                    st.write(f"#### {n_gram_size}連結パターンの時間的分布")

                    fig, distribution_results = visualize_pattern_distribution(df, n_gram_size=n_gram_size, top_n=5)
                    if fig:
                        st.pyplot(fig)

                        # 図の保存
                        save_path = save_figure(fig, f"pattern_distribution_{n_gram_size}.png")
                        st.success(f"{n_gram_size}連結パターンの時間的分布を保存しました: {save_path}")
                    else:
                        st.warning(f"{n_gram_size}連結パターンが見つかりませんでした。")

                    # 4. パターン遷移ネットワーク
                    st.write(f"#### {n_gram_size}連結パターンの遷移ネットワーク")

                    min_count = st.slider(f"最小出現回数（{n_gram_size}連結）",
                                          min_value=1, max_value=10, value=2, key=f"min_count_{n_gram_size}")

                    fig, network_results = visualize_pattern_network(df, n_gram_size=n_gram_size, min_count=min_count)
                    if fig:
                        st.pyplot(fig)

                        # 図の保存
                        save_path = save_figure(fig, f"pattern_network_{n_gram_size}.png")
                        st.success(f"{n_gram_size}連結パターンの遷移ネットワークを保存しました: {save_path}")
                    else:
                        st.warning(f"最小出現回数{min_count}以上の{n_gram_size}連結パターンが見つかりませんでした。")

                    # 分析結果に追加
                    analysis_results[f"{n_gram_size}連結パターン分析"] = {
                        "上位パターン": pattern_results["top_patterns"],
                        "パターン出現位置": pattern_results["pattern_positions"],
                        "パターン出現頻度": heatmap_results if fig else {},
                        "パターン時間的分布": distribution_results if fig else {},
                        "パターン遷移ネットワーク": network_results if fig else {"nodes": [], "edges": {}}
                    }

                # 発言順序の時系列可視化（全体）
                st.subheader("発言順序の時系列可視化（全体）")

                # 発言者に色を割り当て
                speaker_sequence = df["発言者"].tolist()
                speakers = sorted(set(speaker_sequence))
                colors = plt.cm.tab20(np.linspace(0, 1, len(speakers)))
                speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}

                # 発言順序を時系列でプロット
                fig, ax = plt.subplots(figsize=(15, 6))
                plt.rcParams['font.sans-serif'] = font_settings[selected_language]

                for i, speaker in enumerate(speaker_sequence):
                    ax.scatter(i, 0, color=speaker_colors[speaker], s=100, alpha=0.7)

                    # 発言者名を表示（間引き）
                    if i % 10 == 0 or i == len(speaker_sequence) - 1:
                        ax.text(i, 0.1, speaker, rotation=90, fontsize=8,
                                ha='center', va='bottom')

                # 凡例の作成
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=speaker_colors[speaker], markersize=10,
                                              label=speaker) for speaker in speakers]
                ax.legend(handles=legend_elements, loc='upper right')

                ax.set_xlabel('発言順序')
                ax.set_yticks([])
                ax.set_title('発言順序の時系列可視化')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "speaker_sequence_timeline.png")
                st.success(f"発言順序の時系列可視化を保存しました: {save_path}")

                # 分析結果に追加
                analysis_results["発言順序時系列"] = {
                    "発言順序": speaker_sequence,
                    "発言者一覧": list(speakers)
                }

            tab_index += 1

        # 教師・児童生徒発言量分析
        if "教師・児童生徒発言量分析" in analysis_options:
            with tabs[tab_index]:
                st.header("教師・児童生徒発言量分析")

                # 教師を識別するキーワードの設定
                st.subheader("教師を識別するキーワードの設定")

                # 言語に応じたデフォルトの教師キーワード
                teacher_keywords_default = {
                    "日本語": "教師,先生,T,Teacher",
                    "英語": "teacher,instructor,T,professor",
                    "韓国語": "교사,선생님,T,Teacher",
                    "中国語": "老师,教师,T,Teacher",
                    "フランス語": "professeur,enseignant,T,Teacher"
                }

                teacher_keywords_input = st.text_area(
                    "教師を識別するキーワード（カンマ区切り）",
                    teacher_keywords_default[selected_language]
                )
                teacher_keywords = [keyword.strip() for keyword in teacher_keywords_input.split(",")]

                # 教師と児童生徒の発言量を可視化
                st.subheader("教師と児童生徒の発言量")

                fig = visualize_teacher_student_utterances(df, teacher_keywords)
                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "teacher_student_utterances.png")
                st.success(f"教師と児童生徒の発言量グラフを保存しました: {save_path}")

                # 分析結果に追加
                df["is_teacher"] = df["発言者"].apply(lambda x: any(keyword in x for keyword in teacher_keywords))
                teacher_utterances = df[df["is_teacher"]]["発言長"].sum()
                student_utterances = df[~df["is_teacher"]]["発言長"].sum()

                st.write(f"教師の総発言量: {teacher_utterances} 文字")
                st.write(f"児童生徒の総発言量: {student_utterances} 文字")

                if teacher_utterances > 0:
                    st.write(f"児童生徒/教師の発言量比率: {student_utterances / teacher_utterances:.2f}")
                    ratio = float(student_utterances / teacher_utterances)
                else:
                    st.write("教師の発言が見つかりません。キーワードを確認してください。")
                    ratio = float('inf')

                analysis_results["教師児童生徒発言量"] = {
                    "教師総発言量": int(teacher_utterances),
                    "児童生徒総発言量": int(student_utterances),
                    "発言量比率": ratio
                }

            tab_index += 1

        # 注目語累積分析
        if "注目語累積分析" in analysis_options:
            with tabs[tab_index]:
                st.header("注目語累積分析")

                # 注目する語の入力
                st.subheader("注目する語の入力")

                # 言語に応じたデフォルトの注目語
                default_target_words = {
                    "日本語": "学習,考える,理解,問題,発表",
                    "英語": "learn,think,understand,problem,present",
                    "韓国語": "학습,생각,이해,문제,발표",
                    "中国語": "学习,思考,理解,问题,发表",
                    "フランス語": "apprendre,penser,comprendre,problème,présenter"
                }

                target_words_input = st.text_area(
                    "注目する語（カンマ区切り、最大10語）",
                    default_target_words[selected_language]
                )
                target_words = [word.strip() for word in target_words_input.split(",")][:10]  # 最大10語に制限

                if target_words:
                    # 注目語の累積相対度数を可視化
                    st.subheader("注目語の累積相対度数")

                    fig = visualize_word_cumulative_frequency(df, target_words)
                    if fig:
                        st.pyplot(fig)

                        # 図の保存
                        save_path = save_figure(fig, "word_cumulative_frequency.png")
                        st.success(f"注目語の累積相対度数グラフを保存しました: {save_path}")

                        # 各単語の出現回数を計算
                        word_counts = {}
                        for word in target_words:
                            count = sum(1 for text in df["発言内容"] if word in text)
                            word_counts[word] = count

                        # 出現回数の表示
                        st.subheader("注目語の出現回数")
                        counts_df = pd.DataFrame(list(word_counts.items()), columns=["単語", "出現回数"])
                        counts_df = counts_df.sort_values("出現回数", ascending=False)
                        st.table(counts_df)

                        # 分析結果に追加
                        analysis_results["注目語出現回数"] = word_counts
                    else:
                        st.warning("注目語が発言内容に見つかりませんでした。")
                else:
                    st.warning("注目する語を入力してください。")

            tab_index += 1

        # 分析結果をJSONで保存
        if st.sidebar.button("分析結果をJSONで保存"):
            json_path = save_analysis_results(analysis_results)
            st.sidebar.success(f"分析結果をJSONで保存しました: {json_path}")
else:
    st.info("CSVファイルをアップロードしてください。")

    # 言語モデルの情報表示
    st.header("言語モデルについて")
    st.write(f"現在選択されている言語: {selected_language}")
    st.write(f"現在選択されているモデル: {selected_model}")

    # 各言語モデルの説明
    model_descriptions = {
        "sm": "小サイズモデル: 軽量で高速、基本的な言語処理に適しています。",
        "md": "中サイズモデル: バランスの取れた性能と速度、一般的な分析に推奨。",
        "lg": "大サイズモデル: 高精度だが処理速度は遅い、詳細な分析に適しています。"
    }

    st.subheader("モデルサイズの説明")
    for size, desc in model_descriptions.items():
        st.write(f"**{size}**: {desc}")

    st.subheader("使用方法")
    st.write("1. サイドバーから言語と言語モデルを選択")
    st.write("2. CSVファイルをアップロード（発言番号、発言者、発言内容の列が必要）")
    st.write("3. 分析項目を選択して結果を確認")
    st.write("4. 必要に応じて分析結果をJSONで保存")