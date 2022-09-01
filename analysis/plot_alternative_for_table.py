import matplotlib.pyplot as plt
import pandas as pd

import groups_cols

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']


def plotting(data, name='name'):
    lbsl = [groups_cols.abm_dummies_show[l] for l in data.index.tolist()]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, each in enumerate(['POLICIES_buy', 'POLICIES_rent', 'POLICIES_wage', 'POLICIES_no_policy']):
        for mr in data.index:
            ax.scatter(mr, data.loc[mr, each], color=colors[i], alpha=.6, marker='o')
    plt.vlines(x=range(len(lbsl)), ymin=0, ymax=1, colors='lightgrey', lw=.8, alpha=.5)
    plt.xticks(range(len(lbsl)), lbsl, rotation='vertical', fontsize=9)
    ax.legend(['Purchase', 'Rent vouchers', 'Monetary aid', 'no-policy baseline'],
              edgecolor='white', loc='upper center', facecolor='white', framealpha=1)
    plt.ylabel("Percentage of Metropolitan Regions' optimal cases per policy")
    leg = ax.get_legend()
    # hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}

    for i in range(len(colors)):
        leg.legendHandles[i].set_color(colors[i])

    plt.tight_layout()
    plt.savefig(f'graph_sorted_{name}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    d = pd.read_csv('IQR.csv', sep=';')
    d.drop('Unnamed: 0', inplace=True, axis=1)
    pivoted_d = pd.pivot(d, index='ACP', columns='pol', values='mean')

    pivoted_d.sort_values(by='POLICIES_no_policy', ascending=False, inplace=True)
    plotting(pivoted_d, name='POLICIES_no_policy')
