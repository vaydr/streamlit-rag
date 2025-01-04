import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network

import random
import seaborn as sns
from networkx.algorithms import community as nx_community
import community.community_louvain as community_louvain

import re
from pyvis.network import Network

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
        # Store the relationship as 'title' so PyVis can display it on hover
        G.add_edge(node1, node2, title=edge_label)
    return G

def color_communities_girvan_newman(G):
    """
    Detect communities using Girvan–Newman and assign each community
    a unique color.  Modifies G in-place by setting G.nodes[node]['color'].
    """
    # 1) Run Girvan–Newman (take the first partition level)
    communities_generator = nx_community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    communities_list = sorted(map(sorted, top_level_communities))

    # 2) Generate a color palette
    palette = sns.color_palette("hls", len(communities_list)).as_hex()
    #random.shuffle(palette)

    # 3) Assign each community a color
    for idx, community_nodes in enumerate(communities_list):
        color = palette[idx]
        for node in community_nodes:
            # Store color in the node attributes
            G.nodes[node]['color'] = color
    return G

def color_communities_louvain(G):
    partition = community_louvain.best_partition(G)
    # partition is a dict: node -> community_id
    # Group them by community_id
    comm_dict = {}
    for node, comm_id in partition.items():
        comm_dict.setdefault(comm_id, []).append(node)

    # Generate a color palette
    communities_list = list(comm_dict.values())
    palette = sns.color_palette("hls", len(communities_list)).as_hex()
    random.shuffle(palette)

    # Assign each node a color
    for idx, community_nodes in enumerate(communities_list):
        color = palette[idx]
        for node in community_nodes:
            G.nodes[node]['color'] = color

    return G

def draw_config_on_right(G, output_html="graph_config_right.html"):
    """
    1) Creates a PyVis network with net.show_buttons(...)
       (which in your version uses <div id="config">)
    2) Saves the raw PyVis HTML
    3) Extracts <head>, all <script>, <div id="mynetwork">, and <div id="config">
    4) Rebuilds a new <body> that places #mynetwork on the left,
       and #config in a right 300px panel (via CSS flexbox).
    """

    # ----- A) Build & save PyVis HTML -----
    net = Network(height="750px", width="100%", notebook=False, cdn_resources="remote")
    net.from_nx(G)
    # You can add more filters if you want more sliders:
    net.show_buttons(filter_=["physics"])  
    net.save_graph(output_html)

    with open(output_html, "r", encoding="utf-8") as f:
        original_html = f.read()

    # ----- B) Extract <head>, <script> blocks, #mynetwork, #config -----
    head_match = re.search(r"(?s)<head>(.*?)</head>", original_html, re.IGNORECASE)
    head_content = head_match.group(1) if head_match else ""

    # Grab all <script> tags
    script_regex = re.compile(r"(?s)<script.*?>.*?</script>", re.IGNORECASE)
    scripts = script_regex.findall(original_html)
    # Remove them from the main text so we don’t double them
    no_scripts = script_regex.sub("", original_html)

    # #mynetwork
    netw_match = re.search(r'(?s)(<div[^>]+id="mynetwork"[^>]*>.*?</div>)', no_scripts, re.IGNORECASE)
    mynetwork_html = netw_match.group(1) if netw_match else "<div>There was a PyVis error in the network</div>"

    # #config (this is where PyVis is placing your sliders!)
    config_match = re.search(r'(?s)(<div[^>]+id="config"[^>]*>.*?</div>)', no_scripts, re.IGNORECASE)
    config_html = config_match.group(1) if config_match else "<div>There was a PyVis error in the configuration</div>"

    # ----- C) Build new CSS & <body> with Flexbox -----
    # We force display:flex with !important so it can't stack vertically.
    injected_css = """
<style>
  #flexContainer {
    display: flex !important;
    flex-wrap: nowrap !important;
    width: 100%;
    height: 100vh;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    overflow: hidden; 
  }
  #leftPane {
    flex: 1 1 auto;
    overflow: hidden;
    border-right: 1px solid #ccc;
    position: relative;
  }
  #rightPane {
    width: 300px;
    min-width: 300px;
    box-sizing: border-box;
    background: #f5f5f5;
    color: #333;
    padding: 10px;
    overflow: auto;
  }
</style>
"""

    new_body = f"""
<body>
{injected_css}
<div id="flexContainer">
  <div id="leftPane">
    {mynetwork_html}
  </div>
  <div id="rightPane">
    {config_html}
  </div>
</div>

<!-- re-inject all PyVis scripts below -->
{''.join(scripts)}
</body>
"""

    # ----- D) Rebuild final HTML including <head>... -----
    html_start = re.search(r"(?i)<html.*?>", original_html)
    if html_start:
        start_tag = html_start.group(0)
    else:
        start_tag = "<html>"
    html_end = "</html>"

    final_html = f"""
{start_tag}
<head>
{head_content}
</head>
{new_body}
{html_end}
"""

    # ----- E) Write back -----
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    return output_html


def main():
    st.title("Knowledge Graph Visualizer with Column Selectors")
    st.write("Upload a CSV with **at least 3 columns** so we can map them to node_1, edge, node_2.")

    # 1) Let the user upload a CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        # 2) Read it into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("### CSV Preview")
        st.dataframe(df.head())

        # 3) Let the user pick which columns map to node_1, edge, node_2
        columns = list(df.columns)
        if len(columns) < 3:
            st.warning("Please upload a CSV with at least 3 columns.")
            return  # stop here

        st.write("#### Select which columns correspond to each field:")
        node1_col = st.selectbox("Select the 'node_1' column", columns)
        edge_col = st.selectbox("Select the 'edge' column", columns)
        node2_col = st.selectbox("Select the 'node_2' column", columns)

        # 4) Button to confirm graph creation
        if st.button("Generate Graph"):
            # Create the graph based on the selected columns
            G = create_graph_from_csv(df, node1_col, edge_col, node2_col)
            G = color_communities_girvan_newman(G)
            # G = color_communities_louvain(G)
            html_path = draw_config_on_right(G, "graph_config_right.html")
            # Display inside Streamlit:
            st.write("### Graph Visualization:")
            with open(html_path, 'r', encoding='utf-8') as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=1250, scrolling=True)

if __name__ == "__main__":
    main()
