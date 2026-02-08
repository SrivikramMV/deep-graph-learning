# Deep Graph Learning: Cross-Modal Attention and GNN Failure Analysis

In this project, I investigated how graph structure, heterogeneous feature spaces and message-passing design affect node classification.

Rather than treating a graph neural network as a model that will automatically benefit from relationships between nodes, I wanted to understand **when those relationships are useful, when they can harm performance, and how an architecture can be adapted to the structure of the data**.

The project brings together three investigations:

1. **Cross-modal attention** for a heterogeneous drug–protein graph.
2. **Topology analysis and controlled graph interventions** across two node-classification graphs.
3. **Failure diagnosis and custom architecture design** for a graph where a standard GCN performs substantially worse than a feature-only model.

I implemented the main attention and message-passing components using PyTorch tensor operations, and used quantitative diagnostics and visualisations to investigate the models’ behaviour.

## 1. Cross-modal attention for a drug–protein graph

The first part of the project works with a bipartite graph containing two types of nodes:

- **Drug nodes**, represented by sparse chemical fingerprints.
- **Protein nodes**, represented by dense sequence embeddings.

The goal is to classify drugs using both their own features and their connections to proteins. The difficulty is that the two node types exist in different feature spaces, with different dimensions and semantics.

My initial analysis showed that the drug features were **95% zero-valued**, while the protein features were dense. Drugs also had approximately **41.88 protein neighbours on average**. This made uniform averaging a questionable choice: potentially informative protein interactions could be diluted by a large number of less useful neighbours.

I used feature-distribution analysis, a fixed random projection followed by t-SNE, and visualisations of sampled bipartite subgraphs to examine the gap between the two modalities before building the model.

### Implementing attention from basic operations

I implemented a cross-modal attention layer in PyTorch without relying on prebuilt graph neural network or attention layers.

For each edge, the layer:

1. Projects query, key and value features into a shared hidden dimension.
2. Constructs an attention score from the query node and its neighbour.
3. Normalises scores separately within each node’s neighbourhood.
4. Aggregates the weighted neighbour values.
5. Adds the result to a projected representation of the query node through a residual connection.

The implementation uses a custom **segment softmax** to normalise attention scores independently for each neighbourhood. Within each segment, I subtract the maximum score before exponentiation to improve numerical stability. Weighted edge messages are then accumulated back to their destination nodes using PyTorch’s indexed aggregation operations.

### Two-hop heterogeneous message passing

A single drug update can directly incorporate information from connected proteins, but it cannot incorporate information from another drug unless that information first passes through a shared protein.

To address this, I built a two-stage heterogeneous classifier:

```text
Drug features
     ↓
Update proteins from connected drugs
     ↓
Update drugs from connected proteins
     ↓
Drug classification
```

This allows drug information to propagate through **drug → protein → drug** paths, while keeping the direction of each attention operation explicit.

### Testing a cross-modal alignment objective

I also implemented an alignment loss that pulls the learned representations of connected drug–protein pairs closer together. I compared a classification-only objective with a combined classification and alignment objective, using the same seed and broader model configuration.

| Setting | Best validation macro-F1 | Test macro-F1 |
|---|---:|---:|
| No alignment loss | 0.9733 | 0.9867 |
| Alignment weight 0.05 | 0.8393 | 0.8256 |

In this experiment, the alignment term slowed convergence and reduced classification performance. My interpretation is that the additional objective constrained the shared representation space more strongly than was useful for the classification task. The comparison was useful precisely because the more elaborate objective did **not** improve the recorded result.

## 2. Understanding how topology affects a GCN

The second investigation examines two graphs with different connectivity patterns. I looked at the graph structure from several complementary perspectives:

- **Node degree**, to understand local connectivity.
- **Betweenness centrality**, to identify potential bridges and bottlenecks.
- **Ollivier–Ricci curvature**, to examine the local geometry of edges.
- **Node-feature distributions**, graph visualisations and learned embeddings.
- **Graph spectra**, as another view of structural differences.

I implemented a configurable multilayer GCN and compared training on each graph independently with training on their union. I also used t-SNE visualisations of learned representations to inspect how the models organised the nodes.

### Independent and combined training

In the recorded comparison, validation F1 was:

| Training setup | Validation F1 |
|---|---:|
| Graph 1 independently | 0.7333 |
| Graph 2 independently | 0.8000 |
| Naïve training on the union | 0.7667 |

The shared model’s result fell between the two graph-specific results. I investigated the possibility that a single model was having to accommodate different structural regimes, rather than assuming that combining graphs would necessarily improve performance.

As an additional experiment, I implemented a **domain-conditioned GCN**. It adds a learned graph-identity embedding to the node features so that a shared model can condition its predictions on which graph component a node belongs to.

In the recorded run, this increased the best union validation F1 from **0.7667 to 0.7833**. This was a modest improvement in this setup and gave me a concrete way to examine the potential cost of sharing one model across structurally different graphs.

### Controlled topology experiments

To separate the contribution of node features from the contribution of graph structure, I compared the original graphs with two deliberately different structural conditions:

- **Topology removal:** removed the original edges so that, after self-loops were added, aggregation became effectively self-only.
- **Oracle topology:** connected same-label nodes for diagnostic analysis, using labels to construct an idealised graph.

The oracle graphs reached **1.0000 validation F1** in the recorded experiment. Because labels were used to construct those graphs, this is a diagnostic result, not a deployable method or a fair estimate of performance on an unknown graph. It illustrates how strongly label-aligned connectivity can influence a GCN’s behaviour.

I also designed a **label-free augmentation strategy** based on feature similarity and graph density. For the sparser graph, it adds cosine-similarity nearest-neighbour edges. For the denser graph, it retains a feature-consistent subset of existing neighbours.

In the recorded comparison, the final validation F1 remained **0.7333 for Graph 1** and **0.8000 for Graph 2**, matching the corresponding original-graph results. The experiment provided a useful check on a plausible augmentation strategy without assuming that a structural change had to improve the model.

## 3. Diagnosing a GCN failure on a mystery graph

The final part of the project starts with an intentionally surprising result: a graph model performed much worse than an MLP that ignored the graph entirely.

I trained two baselines under the notebook’s evaluation protocol:

| Model | Test macro-F1 |
|---|---:|
| Feature-only MLP | 0.8542 |
| Mean-aggregation GCN | 0.4908 |

The question was not simply how to raise the score, but **why adding neighbourhood information damaged performance**.

### Investigating the failure

I measured the relationship between edge types and node labels, examined the class mix in incoming neighbourhoods, and compared representations before and after mean aggregation.

The supplied graph had two edge values with sharply different relationships to the labels:

- For edges with `edge_w = 1`, the connected nodes had the same label in the analysed graph.
- For edges with `edge_w = 0`, the connected nodes had different labels in the analysed graph.

Overall directed-edge homophily was **0.50**, and the incoming neighbourhoods had an equal mix of same-class and different-class neighbours in the recorded analysis. Treating every neighbour identically therefore mixed two opposing kinds of information.

I also compared raw node features with mean-aggregated features:

| Diagnostic | Raw features | Mean-aggregated features |
|---|---:|---:|
| Mean class-centroid margin | 0.7553 | 0.0033 |
| Linear-probe test macro-F1 | 0.8625 | 0.4899 |

These diagnostics supported a concrete explanation: mean aggregation was collapsing much of the class-discriminative signal by averaging together relationships with different meanings.

### Designing a relation-aware GNN

I designed a custom message-passing model that separates the two edge types instead of averaging them into one undifferentiated neighbourhood.

The model:

1. Aggregates messages separately for edges weighted by `edge_w` and `1 - edge_w`.
2. Combines the node’s own features with the two neighbourhood summaries.
3. Uses a learned gate in the first layer to control their contributions.
4. Applies separate transformations to the self, positive-channel and negative-channel representations.
5. Repeats relation-aware aggregation in a second layer before classification.

In the recorded notebook run, the custom model achieved:

| Model | Test macro-F1 |
|---|---:|
| Mean-aggregation GCN | 0.4908 |
| Feature-only MLP | 0.8542 |
| **Relation-aware GNN** | **0.9833** |

The result is specific to this graph and depends on an important assumption: **the edge values carry meaningful relation information**. On this dataset, those values are exceptionally informative about whether connected nodes share a label. The experiment shows how exploiting that information can address the observed failure, but it does not establish that the same architecture will generalise to arbitrary graphs.

## Technologies and methods

- Python, PyTorch and NumPy
- Cross-modal attention and neighbourhood-wise segment softmax
- Heterogeneous graphs and two-hop message passing
- Residual connections and connected-pair alignment loss
- Graph convolution and learned domain conditioning
- Degree, betweenness centrality and Ollivier–Ricci curvature
- Feature and embedding visualisation with t-SNE
- Graph topology removal, oracle controls and label-free augmentation
- Homophily analysis, centroid margins and linear probes
- Relation-aware message passing and learned gating
- Macro-F1 evaluation and validation-based model selection

## Running the project

The project is organised as a Jupyter notebook with the accompanying graph datasets and helper code:

```text
.
├── DGL_CW1_2026.ipynb
├── graph_ricci_curvature.py
├── requirements.txt
└── data/
    ├── synth_drug_protein_np.npz
    ├── q2_G1_train.json
    ├── q2_G1_eval.json
    ├── q2_G2_train.json
    ├── q2_G2_eval.json
    └── mystery_graph.json
```

Install the listed dependencies and open the notebook from the project directory:

```bash
pip install -r requirements.txt
jupyter notebook DGL_CW1_2026.ipynb
```

The notebook contains the implementations, recorded outputs, plots and written analysis for each investigation. A Jupyter environment is required to run it.

## What I took from the project

The most interesting part of this work was seeing that **more graph information is not automatically better information**. In the mystery graph, the standard GCN failed because its aggregation rule discarded a distinction that was essential to the task.

By combining controlled comparisons, structural diagnostics and a targeted architecture change, I could connect the failure to a specific property of the data and test a model designed around that property. The same idea ran through the rest of the project: understand the structure first, then choose how information should move through it.
