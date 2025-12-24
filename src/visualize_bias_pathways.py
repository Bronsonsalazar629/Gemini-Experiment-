"""
Visualize causal bias pathways as a network graph.
Creates a visual diagram showing how bias flows through the system.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pc_algorithm_clinical import PCAlgorithmClinical

print("="*80)
print("VISUALIZING CAUSAL BIAS PATHWAYS")
print("="*80)

# Load COMPAS data
print("\n[1/3] Loading data...")
df = pd.read_csv('../propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv')
df_clean = pd.DataFrame({
    'age': 1 - df['Age_Below_TwentyFive'],
    'sex': 1 - df['Female'],
    'priors_count': df['Number_of_Priors'],
    'c_charge_degree': 1 - df['Misdemeanor'],
    'race_binary': df['African_American'],
    'two_year_recid': df['Two_yr_Recidivism']
})

# Run PC algorithm
print("[2/3] Running causal discovery...")
temporal_order = {
    'race_binary': 0,
    'age': 1,
    'sex': 1,
    'priors_count': 2,
    'c_charge_degree': 2,
    'two_year_recid': 3
}

pc_algo = PCAlgorithmClinical(
    data=df_clean,
    protected_attr='race_binary',
    outcome='two_year_recid',
    temporal_order=temporal_order,
    alpha=0.05,
    n_bootstrap=50
)

result = pc_algo.run()

# Create visualization
print("[3/3] Creating network visualization...")

# Get causal graph
G = result['causal_graph']

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# LEFT PLOT: Full causal graph
ax1.set_title('Complete Causal Graph\n(All Discovered Edges)', fontsize=14, fontweight='bold')
pos1 = nx.spring_layout(G, seed=42, k=2)

# Color nodes
node_colors = []
for node in G.nodes():
    if node == 'race_binary':
        node_colors.append('#ff4444')  # Red for protected attribute
    elif node == 'two_year_recid':
        node_colors.append('#4444ff')  # Blue for outcome
    else:
        node_colors.append('#44ff44')  # Green for mediators

nx.draw_networkx_nodes(G, pos1, node_color=node_colors, node_size=3000, alpha=0.9, ax=ax1)
nx.draw_networkx_labels(G, pos1, font_size=9, font_weight='bold', ax=ax1)
nx.draw_networkx_edges(G, pos1, edge_color='gray', arrows=True,
                       arrowsize=20, width=2, alpha=0.6, ax=ax1)

ax1.axis('off')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444',
               markersize=12, label='Protected Attribute (Race)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#44ff44',
               markersize=12, label='Mediator Variables'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444ff',
               markersize=12, label='Outcome (Recidivism)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# RIGHT PLOT: Bias pathways only (simplified)
ax2.set_title('Critical Bias Pathways\n(Direct + Shortest Indirect)', fontsize=14, fontweight='bold')

# Create simplified graph with key pathways
G_simple = nx.DiGraph()
direct_pathway = None
shortest_indirect = None

for pathway in result['bias_pathways']:
    if pathway.pathway_type == 'direct_discrimination':
        direct_pathway = pathway
    elif shortest_indirect is None or len(pathway.path) < len(shortest_indirect.path):
        if pathway.pathway_type == 'systemic_mediator' and len(pathway.path) <= 3:
            shortest_indirect = pathway

# Add edges from key pathways
if direct_pathway:
    for i in range(len(direct_pathway.path) - 1):
        G_simple.add_edge(direct_pathway.path[i], direct_pathway.path[i+1],
                         pathway_type='direct', robustness=direct_pathway.sensitivity_robustness)

if shortest_indirect:
    for i in range(len(shortest_indirect.path) - 1):
        G_simple.add_edge(shortest_indirect.path[i], shortest_indirect.path[i+1],
                         pathway_type='indirect', robustness=shortest_indirect.sensitivity_robustness)

pos2 = nx.spring_layout(G_simple, seed=42, k=3)

# Color nodes
node_colors2 = []
for node in G_simple.nodes():
    if node == 'race_binary':
        node_colors2.append('#ff4444')
    elif node == 'two_year_recid':
        node_colors2.append('#4444ff')
    else:
        node_colors2.append('#ffaa44')  # Orange for key mediators

nx.draw_networkx_nodes(G_simple, pos2, node_color=node_colors2, node_size=4000, alpha=0.9, ax=ax2)
nx.draw_networkx_labels(G_simple, pos2, font_size=10, font_weight='bold', ax=ax2)

# Draw edges with different colors for pathway types
direct_edges = [(u, v) for u, v, d in G_simple.edges(data=True) if d.get('pathway_type') == 'direct']
indirect_edges = [(u, v) for u, v, d in G_simple.edges(data=True) if d.get('pathway_type') == 'indirect']

if direct_edges:
    nx.draw_networkx_edges(G_simple, pos2, edgelist=direct_edges, edge_color='red',
                          arrows=True, arrowsize=25, width=3, alpha=0.8,
                          label='Direct Discrimination', ax=ax2)
if indirect_edges:
    nx.draw_networkx_edges(G_simple, pos2, edgelist=indirect_edges, edge_color='orange',
                          arrows=True, arrowsize=25, width=3, alpha=0.8,
                          label='Systemic Bias', ax=ax2)

ax2.axis('off')
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()

# Save figure
output_path = 'figures/causal_bias_network.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[OK] Network visualization saved: {output_path}")

# Display summary
print("\n" + "="*80)
print("BIAS PATHWAY SUMMARY")
print("="*80)
print(f"\nTotal bias pathways discovered: {len(result['bias_pathways'])}")
print(f"Direct discrimination pathways: {sum(1 for p in result['bias_pathways'] if p.pathway_type == 'direct_discrimination')}")
print(f"Systemic bias pathways: {sum(1 for p in result['bias_pathways'] if p.pathway_type == 'systemic_mediator')}")

if direct_pathway:
    print(f"\nDirect pathway: {' -> '.join(direct_pathway.path)}")
    print(f"  Robustness: {direct_pathway.sensitivity_robustness:.1%}")

if shortest_indirect:
    print(f"\nShortest systemic pathway: {' -> '.join(shortest_indirect.path)}")
    print(f"  Robustness: {shortest_indirect.sensitivity_robustness:.1%}")

print("\n" + "="*80)
print(f"Open the visualization: {output_path}")
print("="*80)
