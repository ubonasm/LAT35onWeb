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
st.set_page_config(page_title="授業分析ツール", layout="wide")
st.title("授業分析ツール")
st.sidebar.header("設定")


# spaCyの日本語モデルをロード
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("ja_core_news_sm")
    except OSError:
        st.warning("日本語モデルをダウンロードしています...")
        spacy.cli.download("ja_core_news_sm")
        return spacy.load("ja_core_news_sm")


nlp = load_nlp_model()

# CSVファイルのアップロード
uploaded_file = st.sidebar.file_uploader("授業記録CSVファイルをアップロード", type=["csv"])

# 分析設定
analysis_options = st.sidebar.multiselect(
    "分析項目を選択",
    ["頻出単語分析", "発言者別分析", "発言パターン分析", "単語共起ネットワーク", "時系列分析",
     "インタラクションパターン分析"],
    default=["頻出単語分析", "発言者別分析", "インタラクションパターン分析"]
)

# 除外する品詞
exclude_pos = st.sidebar.multiselect(
    "除外する品詞",
    ["助詞", "助動詞", "記号", "空白", "接続詞", "感動詞"],
    default=["助詞", "助動詞", "記号", "空白"]
)

# 除外する単語
stop_words = st.sidebar.text_area("除外する単語（カンマ区切り）", "です,ます,ございます,はい,えー,あの,その,この")
stop_words = [word.strip() for word in stop_words.split(",")]

# 品詞マッピング
pos_mapping = {
    "名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ", "副詞": "ADV",
    "助詞": "ADP", "助動詞": "AUX", "記号": "PUNCT", "空白": "SPACE",
    "接続詞": "CCONJ", "感動詞": "INTJ"
}

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
    plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

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
        plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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
    plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

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
    plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

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

            # 発言者ごとの発言数
            speaker_counts = df["発言者"].value_counts()
            st.subheader("発言者ごとの発言数")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao',
                                               'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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
                plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color="white",
                    font_path="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf" if os.path.exists(
                        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf") else None,
                    max_words=100
                ).generate(" ".join(words))

                fig, ax = plt.subplots(figsize=(12, 8))
                plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                # 図の保存
                save_path = save_figure(fig, "wordcloud.png")
                st.success(f"ワードクラウドを保存しました: {save_path}")

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
                        plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic',
                                                           'Noto Sans CJK JP']
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
                plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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
                plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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
                    plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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
                plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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
                plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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

        # インタラクションパターン分析（新機能）
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
                plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

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

        # 分析結果をJSONで保存
        if st.sidebar.button("分析結果をJSONで保存"):
            json_path = save_analysis_results(analysis_results)
            st.sidebar.success(f"分析結果をJSONで保存しました: {json_path}")