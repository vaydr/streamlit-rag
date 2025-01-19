import streamlit as st
st.set_page_config(layout="wide", page_title="VKGQA")
import pandas as pd
import networkx as nx
from pyvis.network import Network

import re
import seaborn as sns
from networkx.algorithms import community as nx_community
import community.community_louvain as community_louvain

def create_graph_from_csv(df, node1_col, edge_col, node2_col):
    """
    Given a DataFrame and the names of the columns that correspond
    to node_1, edge, node_2, build a NetworkX graph.
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        node1 = str(row[node1_col]).strip()
        node2 = str(row[node2_col]).strip()
        edge_label = str(row[edge_col]).strip()
        G.add_node(node1)
        G.add_node(node2)
        # Store the relationship in 'title' so PyVis can display it on hover
        G.add_edge(node1, node2, title=edge_label)
    return G

def color_communities_girvan_newman(G):
    """
    Detect communities using Girvan–Newman and assign each community
    a unique color.  Modifies G in-place by setting G.nodes[node]['color'].
    """
    communities_generator = nx_community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    communities_list = sorted(map(sorted, top_level_communities))

    palette = sns.color_palette("hls", len(communities_list)).as_hex()
    for idx, community_nodes in enumerate(communities_list):
        color = palette[idx]
        for node in community_nodes:
            G.nodes[node]['color'] = color
    return G

def color_communities_louvain(G):
    """
    Detect communities using Louvain and assign each community a unique color.
    """
    partition = community_louvain.best_partition(G)
    comm_dict = {}
    for node, comm_id in partition.items():
        comm_dict.setdefault(comm_id, []).append(node)

    communities_list = list(comm_dict.values())
    palette = sns.color_palette("hls", len(communities_list)).as_hex()
    for idx, community_nodes in enumerate(communities_list):
        color = palette[idx]
        for node in community_nodes:
            G.nodes[node]['color'] = color
    return G

def draw_graph_reset(G, output_html="graph_reset.html"):
    """
    1) Renders a PyVis network with forceAtlas2Based physics for 2500 iterations,
    2) Then disables physics once stabilization completes, so the graph is "frozen."
    3) No user-accessible sliders or controls.
    """
    net = Network(height="750px", width="100%", notebook=False, cdn_resources="remote")

    # Convert from the given NetworkX graph
    net.from_nx(G)

    # We'll start with physics ON (forceAtlas2Based, 2500 iter), then disable it post-stabilization
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "maxVelocity": 50,
        "stabilization": {
          "enabled": true,
          "iterations": 500
        }
      }
    }
    """)

    # Save the standard PyVis HTML (no net.show_buttons())
    net.save_graph(output_html)

    # ~~~~~ Post-process the HTML ~~~~~
    # Insert a small snippet that: once physics is done, we disable it.
    with open(output_html, "r", encoding="utf-8") as f:
        html_code = f.read()

    # This snippet runs after "network = new vis.Network(...)" lines
    inject_snippet = """
network.once("stabilizationIterationsDone", function() {
  network.setOptions({ physics: { enabled: false } });
});
"""

    # We'll insert the snippet right after the line:
    #    network = new vis.Network(container, data, options);
    # so the final layout is stable, then frozen.
    replacement = "network = new vis.Network(container, data, options);\n" + inject_snippet

    updated_code = html_code.replace(
        "network = new vis.Network(container, data, options);",
        replacement,
        1  # replace only the first occurrence
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(updated_code)

    return output_html

def main():
    st.title("Visual Knowledge Graph Question-Answering")
    st.write("Upload a CSV or a TXT file with at least 3 columns/triplets.")

    # Let the user upload either CSV or TXT
    uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            # assume .txt => parse lines as "n1|edge|n2"
            raw_text = uploaded_file.read().decode("utf-8", errors="replace")
            lines = raw_text.splitlines()
            rows = []
            for line in lines:
                parts = line.split('|')
                if len(parts) < 3:
                    continue
                node1, edge, node2 = (p.strip() for p in parts)
                rows.append([node1, edge, node2])
            df = pd.DataFrame(rows, columns=["node_1", "edge", "node_2"])

        st.write("#### Optional: Number of lines to keep (for debug):")
        num_lines = st.text_input("(Leave blank or zero to keep all)", value="")
        try:
            n_val = int(num_lines)
            if n_val > 0:
                df = df.iloc[:n_val].copy()
        except ValueError:
            pass

        st.write("### Data Preview")
        st.dataframe(df.head())

        columns = list(df.columns)
        if len(columns) < 3:
            st.warning("Please upload a file with at least 3 columns or triplet lines.")
            return

        st.write("#### Select which columns correspond to each field:")
        node1_col = st.selectbox("Select the 'node_1' column", columns)
        edge_col = st.selectbox("Select the 'edge' column", columns)
        node2_col = st.selectbox("Select the 'node_2' column", columns)

        if st.button("Generate Graph"):
            # Build the graph from DF
            G = create_graph_from_csv(df, node1_col, edge_col, node2_col)

            # Color with Louvain
            G = color_communities_louvain(G)

            # Draw with forceAtlas2Based => then freeze once stable
            html_path = draw_graph_reset(G, "graph_reset.html")

            st.write("### Graph Visualization:")
            with open(html_path, "r", encoding="utf-8") as f:
                html_code = f.read()

            st.components.v1.html(html_code, height=750, scrolling=False)


if __name__ == "__main__":
    main()
