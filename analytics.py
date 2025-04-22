# solution converges to the optimal solution
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def calculate_shortest_route_per_iteration(routes):
    df = pd.DataFrame(routes)
    df_grouped = df.groupby('iteration').min().reset_index()
    return df_grouped

def plot_shortest_distance(df, output_path='./graphs/shortest_distance_100_iterations.png'):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define the parameters
    ant_population = 10
    iterations = 20
    alphas = [2]
    betas = [5]
    rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['rho'], df['distance'], marker='o', label='Shortest Distance')
    plt.title('Shortest Distance per rho')
    plt.suptitle(f'Parameters: ant_population={ant_population}, iterations={iterations}, '
                 f'alpha={alphas}, beta={betas}, rho={rhos}', fontsize=10, y=.95)
    plt.xlabel('rho')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.legend()
    plt.show()

    # plt.savefig(output_path)
    plt.close()


def load_data(data_dir: Path):
    alphas = [1, 2, 5]
    betas = [1, 2, 5]
    rhos = [0.1]

    records = []

    for alpha in alphas:
        for beta in betas:
            for rho in rhos:
                filename = data_dir / f"routes_10_alpha_{alpha}_beta_{beta}_rho_{rho}.json"
                if not filename.is_file():
                    print(f"Warning: {filename} not found, skipping.")
                    continue

                # load the per-iteration records
                with open(filename, 'r') as f:
                    routes = json.load(f)

                # append each record with its parameters
                for rec in routes:
                    records.append({
                        'iteration': 20,
                        'distance': rec['distance'],
                        'alpha': alpha,
                        'beta': beta,
                        'rho': rho
                    })

    # build one DataFrame
    df = pd.DataFrame.from_records(records)
    return df.groupby(['alpha', 'beta']).min().reset_index()


def plot_grid_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    # Pivot to create a matrix: rows=alpha, columns=beta
    matrix = df.pivot(index='alpha', columns='beta', values='distance')

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot heatmap with the first row at the bottom
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(
        matrix.values,
        aspect='auto',
        origin='lower'
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel('β (beta)')
    ax.set_ylabel('α (alpha)')
    ax.set_title('Distance heatmap for α and β')
    fig.colorbar(cax, ax=ax, label='Distance')

    plt.savefig(output_path)
    plt.show()
    plt.close()




if __name__ == '__main__':
    # routes = json.load(open('data/routes_100.json'))
    # df = calculate_shortest_route_per_iteration(routes)
    # plot_shortest_distance(df)
    df = load_data(Path('./data'))
    print(df)
    # plot_grid_heatmap(df, output_path=Path('graphs/heatmap_iter10.png'))
    # plot_shortest_distance(df, output_path='./graphs/plot_rhos.png')