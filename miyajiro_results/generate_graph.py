import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_aucroc_by_method(df, save_dir=None):
    """
    Given a DataFrame, plots AUC-ROC vs. Abstraction State/Epsilon for each abstraction method.
    """
    # abstraction_methodごとに異なるサブプロットを作成
    unique_methods = df['abstraction_method'].unique()
    num_methods = len(unique_methods)

    # 各abstraction_methodに対してグラフをプロット
    for i, method in enumerate(unique_methods):
        is_grid = method == 'Grid'
        is_epsilon = method in ['DBSCAN', 'OPTICS']

        # 対応するabstraction_methodのデータを選択
        method_df = df[df['abstraction_method'] == method]

        # DBSCANとOPTICSの場合は`epsilon`を使用し、それ以外は`abstract_state_num`を使用
        method_df['x_axis'] = method_df['epsilon'] if is_epsilon else method_df['partition_num'] if is_grid else method_df['abstract_state_num']

        x_ticks = method_df['x_axis'].unique()
        
        if not is_epsilon:
            x_ticks = [int(x) for x in x_ticks]

        # サブプロットのレイアウトを設定
        fig, ax = plt.subplots(figsize=(10, 4))

        # pca_dimごとにデータをグループ化し、各グループに対して折れ線グラフをプロット
        for pca_dim in method_df['pca_dim'].unique():
            sub_df = method_df[method_df['pca_dim'] == pca_dim]
            if is_grid:
                ax.plot(sub_df['x_axis'], sub_df['aucroc'], label=f'PCA Dim: {pca_dim}')
            else:
                ax.semilogx(sub_df['x_axis'], sub_df['aucroc'], label=f'PCA Dim: {pca_dim}', marker='o')  # 点を強調

        # x軸の目盛りを設定
        ax.set_xticks(x_ticks)
        # 小数点以下を表示しないように整数に変換
        ax.set_xticklabels(x_ticks)

        # y軸の範囲を0.50から1.00に固定
        ax.set_ylim([0.50, 1.00])

        # サブプロットのタイトルと軸ラベルを設定
        ax.set_title(f'State Space Partitioning Method: {method}')
        ax.set_xlabel('Epsilon' if is_epsilon else 'Number of Clusters')
        ax.set_ylabel('AUC-ROC')
        ax.legend()

        # グラフのレイアウトを調整
        plt.tight_layout()

        # グラフを保存または表示
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'graph_{method}.png'))
        else:
            plt.show()
        plt.close(fig)


# メイン実行部
if __name__ == '__main__':
    # CSVファイルを読み込む
    csv_file_path = 'miyajiro_results/result.csv'
    df = pd.read_csv(csv_file_path)
    save_dir = 'miyajiro_results'

    # グラフ描画関数を呼び出す
    plot_aucroc_by_method(df, save_dir)
