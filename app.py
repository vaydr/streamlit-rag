import streamlit as st
st.set_page_config(layout="wide")
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
    width: 550px;
    min-width: 300px;
    box-sizing: border-box;
    background: #f5f5f5;
    color: #333;
    padding: 10px;
    overflow: auto;
  }
  #config {
    width: 100% !important; 
    box-sizing: border-box !important; 
    float: none !important; 
  }
  #mynetwork {
    border: none !important;
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

    updated_html = final_html

    # 6) Write the final cleaned-up HTML
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(updated_html)

    return output_html

import re
from pyvis.network import Network

def draw_graph_with_lasso_and_textbox(G, output_html="graph_lasso.html"):
    """
    1) Builds a PyVis network with "physics" sliders on the right (#config).
    2) Lasso selection (borderWidth=5 for newly lassoed nodes).
       - Clears old lasso selection so previous nodes revert to normal.
    3) A text box & "Submit" button -> highlight half the selected nodes in gold,
       clearing any prior gold highlight from a previous question.
    4) If lasso is off, we re-enable normal canvas panning/zoom and 
       disable the overlay canvas pointer events so user can drag/zoom again.
    """

    import re
    from pyvis.network import Network

    # 1) Create PyVis network
    net = Network(height="750px", width="100%", notebook=False, cdn_resources="remote")
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])  # just the physics panel
    net.save_graph(output_html)

    with open(output_html, "r", encoding="utf-8") as f:
        original_html = f.read()

    # Grab <head> content
    head_match = re.search(r"(?s)<head>(.*?)</head>", original_html, re.IGNORECASE)
    head_content = head_match.group(1) if head_match else ""

    # Extract all <script> blocks
    script_regex = re.compile(r"(?s)<script.*?>.*?</script>", re.IGNORECASE)
    scripts = script_regex.findall(original_html)
    # Remove them from the text so we can place them later
    no_scripts = script_regex.sub("", original_html)

    # Find #mynetwork and #config divs
    net_match = re.search(r'(?s)(<div[^>]+id="mynetwork"[^>]*>.*?</div>)', no_scripts, re.IGNORECASE)
    mynetwork_html = net_match.group(1) if net_match else "<div>CouldNotFind_mynetwork</div>"

    cfg_match = re.search(r'(?s)(<div[^>]+id="config"[^>]*>.*?</div>)', no_scripts, re.IGNORECASE)
    config_html = cfg_match.group(1) if cfg_match else "<div>CouldNotFind_config</div>"

    # =========================
    # A) CSS for layout + overlay
    # =========================
    custom_css = """
<style>
  #flexContainer {
    display: flex !important;
    width: 100%;
    height: 100vh;
    margin: 0; 
    padding: 0;
    box-sizing: border-box;
    overflow: hidden; 
  }
  #leftPane {
    flex: 1 1 auto;
    border-right: 1px solid #ccc;
    position: relative; /* container for overlay */
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #mynetwork {
    flex: 1 1 auto;
    position: relative;
    border: none !important;
  }
  #lassoOverlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none; /* toggled to 'auto' if lasso is ON */
    cursor: crosshair;
  }
  #toolbar {
    background: #eee;
    padding: 8px;
  }
  #statusBox {
    margin-top: 5px;
    background: #fff;
    border: 1px solid #ccc;
    width: 95%;
    height: 60px;
    resize: vertical;
    overflow: auto;
  }
  #rightPane {
    width: 400px;
    min-width: 300px;
    box-sizing: border-box;
    background: #f5f5f5;
    color: #333;
    padding: 10px;
    overflow: auto;
  }
  #config {
    width: 100% !important;
    float: none !important;
    box-sizing: border-box !important;
  }
</style>
"""

    # =========================
    # B) JavaScript: lasso + text box
    # =========================
    custom_js = r"""
<script>
document.addEventListener("DOMContentLoaded", function() {
    var net = window.network;
    var container = document.getElementById('mynetwork');

    // Insert overlay canvas for lasso lines
    var overlay = document.createElement('canvas');
    overlay.id = "lassoOverlay";
    container.appendChild(overlay);

    // Insert toolbar + status box
    var toolbar = document.createElement('div');
    toolbar.id = "toolbar";
    toolbar.innerHTML = `
      <div style="margin-bottom:5px;">
        <input type="checkbox" id="lassoCheckbox"/>
        <label for="lassoCheckbox"> Lasso Mode (freeze canvas)</label>
      </div>
      <div>
        <input type="text" id="questionInput" placeholder="Ask a question..." style="width:60%;"/>
        <button id="submitButton">Submit</button>
      </div>
      <div id="statusBox"></div>
    `;
    var leftPane = document.getElementById("leftPane");
    leftPane.appendChild(toolbar);

    var ctx = overlay.getContext('2d');
    var lassoCheckbox = document.getElementById("lassoCheckbox");
    var questionInput = document.getElementById("questionInput");
    var submitButton = document.getElementById("submitButton");
    var statusBox = document.getElementById("statusBox");

    var isDrawing = false;
    var lassoPoints = [];
    var selectedNodeIDs = [];
    // We'll store previously gold-highlighted nodes to revert them when next question
    var oldGoldNodes = [];

    // Save default drag/zoom
    var defaultDragView = net.interactionHandler.options.dragView;
    var defaultZoomView = net.interactionHandler.options.zoomView;
    var defaultDragNodes = net.interactionHandler.options.dragNodes;

    // Resizing overlay
    function resizeOverlay() {
      let rect = container.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;
    }
    resizeOverlay();
    window.addEventListener('resize', resizeOverlay);

    function logStatus(msg) {
      statusBox.innerHTML += msg + "<br/>";
      statusBox.scrollTop = statusBox.scrollHeight;
    }

    // Lasso checkbox toggles
    lassoCheckbox.addEventListener("change", function(){
      let lassoOn = this.checked;
      if(lassoOn) {
        net.interactionHandler.options.dragView = false;
        net.interactionHandler.options.zoomView = false;
        net.interactionHandler.options.dragNodes = false;
        overlay.style.pointerEvents = "auto";
        logStatus("Lasso: ON");
      } else {
        net.interactionHandler.options.dragView = defaultDragView;
        net.interactionHandler.options.zoomView = defaultZoomView;
        net.interactionHandler.options.dragNodes = defaultDragNodes;
        overlay.style.pointerEvents = "none";
        logStatus("Lasso: OFF");
      }
    });

    // MOUSEDOWN => start capturing
    overlay.addEventListener("mousedown", function(e){
      if(lassoCheckbox.checked) {
        isDrawing = true;
        lassoPoints = [ getLocalPos(e) ];
        clearOverlay();
      }
    });

    // MOUSEMOVE => draw lines
    overlay.addEventListener("mousemove", function(e){
      if(isDrawing && lassoCheckbox.checked) {
        lassoPoints.push(getLocalPos(e));
        drawPolygon(lassoPoints, false);
      }
    });

    // MOUSEUP => finalize
    overlay.addEventListener("mouseup", function(e){
      if(isDrawing && lassoCheckbox.checked){
        isDrawing = false;
        drawPolygon(lassoPoints, true);

        // Revert old selected nodes to normal
        if(selectedNodeIDs.length > 0) {
          revertNodesToDefault(selectedNodeIDs);
        }

        // Now find new selection
        let nodeIDs = net.body.data.nodes.getIds();
        let newlySelected = [];
        for(let nid of nodeIDs) {
          let posObj = net.getPositions(nid);
          let domPos = net.canvasToDOM(posObj[nid]);
          let overlayPos = getOverlayCoords(domPos.x, domPos.y);
          if(isInsidePolygon(overlayPos.x, overlayPos.y, lassoPoints)){
            newlySelected.push(nid);
          }
        }
        selectedNodeIDs = newlySelected;

        // highlight them w/ borderWidth=5
        highlightNodes(newlySelected, 5, null);

        logStatus("New lasso selection: "+newlySelected.join(", "));
        lassoPoints = [];
        clearOverlay();
      }
    });

    // On "Submit," revert old gold, highlight new subset in gold
    submitButton.addEventListener("click", function(){
      let question = questionInput.value.trim();
      if(!question) return;

      // revert old gold back to normal
      revertNodesToDefault(oldGoldNodes);
      oldGoldNodes = [];

      let msg = `Response to question "${question}" on selected [${selectedNodeIDs.join(", ")}]`;
      logStatus(msg);

      // for demonstration: gold highlight half
      let half = Math.floor(selectedNodeIDs.length/2);
      let subset = selectedNodeIDs.slice(0, half);

      // highlight w/ gold border => ignoring the old borderWidth
      highlightNodes(subset, null, "gold");
      oldGoldNodes = subset.slice(); // store them to revert later
    });


    // ~~~~~~~~~~~~~ HELPER FUNCS ~~~~~~~~~~~~~

    function highlightNodes(nodeIDs, borderWidthVal, borderColorVal){
      let updates = [];
      for(let nid of nodeIDs){
        let obj = {id:nid};
        if(borderWidthVal!==null) obj.borderWidth = borderWidthVal;
        if(borderColorVal!==null) obj.color = {border: borderColorVal};
        updates.push(obj);
      }
      net.body.data.nodes.update(updates);
    }

    // revert to default (borderWidth=1, border color=undefined -> old color)
    function revertNodesToDefault(nodeIDs){
      let updates=[];
      for(let nid of nodeIDs){
        updates.push({
          id:nid,
          borderWidth:1,
          color: { border: undefined }
        });
      }
      net.body.data.nodes.update(updates);
    }

    function clearOverlay(){
      ctx.clearRect(0,0,overlay.width,overlay.height);
    }

    function drawPolygon(points, close){
      clearOverlay();
      ctx.beginPath();
      if(points.length>0){
        ctx.moveTo(points[0].x, points[0].y);
        for(let i=1; i<points.length; i++){
          ctx.lineTo(points[i].x, points[i].y);
        }
        if(close) ctx.closePath();
      }
      ctx.strokeStyle="rgba(255,0,0,0.7)";
      ctx.lineWidth=2;
      ctx.stroke();
    }

    function getLocalPos(e){
      let rect = overlay.getBoundingClientRect();
      return { x:e.clientX-rect.left, y:e.clientY-rect.top };
    }

    // net.canvasToDOM => doc coords
    // We want overlay coords => subtract container rect
    function getOverlayCoords(domX, domY){
      let rect = container.getBoundingClientRect();
      return {x: domX-rect.left, y:domY-rect.top};
    }

    // ray-casting
    function isInsidePolygon(px,py, poly){
      let inside=false;
      for(let i=0, j=poly.length-1; i<poly.length; j=i++){
        let xi=poly[i].x, yi=poly[i].y;
        let xj=poly[j].x, yj=poly[j].y;
        let inter=((yi>py)!=(yj>py)) && (px<(xj-xi)*(py-yi)/(yj-yi)+xi);
        if(inter) inside=!inside;
      }
      return inside;
    }
});
</script>
"""

    # We'll place everything in #leftPane
    left_pane_html = f'''
<div id="leftPane">
  {mynetwork_html}
</div>
'''

    custom_body = f"""
<body>
{custom_css}
<div id="flexContainer">
  {left_pane_html}
  <div id="rightPane">
    {config_html}
  </div>
</div>
{''.join(scripts)}
{custom_js}
</body>
"""

    # Rebuild the final HTML
    html_start = re.search(r"(?i)<html.*?>", original_html)
    if html_start:
        start_tag = html_start.group(0)
    else:
        start_tag = "<html>"
    final_html = f"""
{start_tag}
<head>
{head_content}
</head>
{custom_body}
</html>
"""

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
            html_path = draw_graph_with_lasso_and_textbox(G, "graph_lasso.html")
            # Display inside Streamlit:
            st.write("### Graph Visualization:")
            with open(html_path, 'r', encoding='utf-8') as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=1250, scrolling=True)

if __name__ == "__main__":
    main()
