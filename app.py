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

def draw_graph_with_lasso_and_textbox(G, output_html="graph_lasso.html"):
    """
    1) Builds a PyVis network with "physics" sliders on the right (#config).
    2) Lasso selection => black border, borderWidth=5 (reverts old selection).
    3) “Submit” => highlight half of the selected nodes with gold border,
       revert old gold from prior question.
    4) Revert => background=original color, border=slightly darker color
       so we can distinguish the border from the fill.
    5) If lasso is off => re-enable normal canvas panning/zoom; if on => freeze.
    """

    # ~~~~~ Step A: Capture original node color so we can revert later ~~~~~
    nodeColors = {}
    for node, data in G.nodes(data=True):
        if 'color' not in data:
            data['color'] = '#97c2fc'  # default fill
        nodeColors[node] = data['color']  # can be a hex string or {border:..., background:...} dict

    # Build the PyVis
    net = Network(height="750px", width="100%", notebook=False, cdn_resources="remote")
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])
    net.save_graph(output_html)

    with open(output_html, "r", encoding="utf-8") as f:
        original_html = f.read()

    head_match = re.search(r"(?s)<head>(.*?)</head>", original_html, re.IGNORECASE)
    head_content = head_match.group(1) if head_match else ""

    script_regex = re.compile(r"(?s)<script.*?>.*?</script>", re.IGNORECASE)
    scripts = script_regex.findall(original_html)
    no_scripts = script_regex.sub("", original_html)

    netw_match = re.search(r'(?s)(<div[^>]+id="mynetwork"[^>]*>.*?</div>)', no_scripts, re.IGNORECASE)
    mynetwork_html = netw_match.group(1) if netw_match else "<div>CouldNotFind_mynetwork</div>"

    cfg_match = re.search(r'(?s)(<div[^>]+id="config"[^>]*>.*?</div>)', no_scripts, re.IGNORECASE)
    config_html = cfg_match.group(1) if cfg_match else "<div>CouldNotFind_config</div>"

    # ~~~~~ Step B: Minimal CSS for layout + overlay ~~~~~
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
    position: relative;
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
    pointer-events: none; 
    cursor: crosshair;
  }
  /* We no longer need #toolbar below the canvas,
     because we want it below the physics config in #rightPane. */
  #toolbar {
    background: #eee;
    padding: 8px;
    margin-top: 20px; /* some spacing from the config panel */
  }
  #statusBox {
    margin-top: 5px;
    background: #fff;
    border: 1px solid #ccc;
    width: 95%;
    height: 200px;
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
    height: 40% !important;
    float: none !important;
    box-sizing: border-box !important;
    transform: scale(0.8) !important;
    transform-origin: top left !important;
  }
</style>
"""

    # Convert Python dict of nodeColors => a JSON-like string we can parse in JS
    node_colors_json = repr(nodeColors)

    # ~~~~~ Step C: JavaScript ~~~~~
    # We only change where we append the #toolbar: now it goes in #rightPane.
    custom_js = rf"""
<script>
document.addEventListener("DOMContentLoaded", function() {{
    var net = window.network;
    var container = document.getElementById('mynetwork');

    var originalNodeColors = {node_colors_json};

    function parseOriginalNodeColor(nid) {{
      let col = originalNodeColors[nid];
      if(!col) {{
        return {{border:"#697fa0", background:"#97c2fc"}};
      }}
      if(typeof col==="string") {{
        let darker = darkenColor(col, 0.2);
        return {{border:darker, background:col}};
      }} else if(typeof col==="object") {{
        let bg = col.background || col.border || "#97c2fc";
        let darker = darkenColor(bg, 0.2);
        return {{border:darker, background:bg}};
      }} 
      return {{border:"#697fa0", background:"#97c2fc"}};
    }}

    function darkenColor(hex, amt) {{
      let c = hex.replace(/^#/, "");
      if(c.length===3) c=c[0]+c[0]+c[1]+c[1]+c[2]+c[2];
      let r = parseInt(c.slice(0,2),16);
      let g = parseInt(c.slice(2,4),16);
      let b = parseInt(c.slice(4,6),16);
      r = Math.floor(r*(1-amt));
      g = Math.floor(g*(1-amt));
      b = Math.floor(b*(1-amt));
      if(r<0) r=0; if(g<0)g=0;if(b<0)b=0;
      let hr = r.toString(16).padStart(2,"0");
      let hg = g.toString(16).padStart(2,"0");
      let hb = b.toString(16).padStart(2,"0");
      return "#"+hr+hg+hb;
    }}

    var oldLassoNodes = [];
    var oldGoldNodes = [];

    // Insert overlay canvas in #mynetwork
    var overlay = document.createElement('canvas');
    overlay.id = "lassoOverlay";
    container.appendChild(overlay);
    var ctx = overlay.getContext('2d');

    // Instead of appending #toolbar to the left pane,
    // we append it *after* the config in #rightPane
    var rightPane = document.getElementById("rightPane");
    var toolbar = document.createElement('div');
    toolbar.id = "toolbar";
    toolbar.innerHTML=`
      <div style="margin-bottom:5px;">
        <input type="checkbox" id="lassoCheckbox"/>
        <label for="lassoCheckbox">Lasso Mode</label>
      </div>
      <div>
        <input type="text" id="questionInput" placeholder="Ask a question..." style="width:60%;"/>
        <button id="submitButton">Submit</button>
      </div>
      <div id="statusBox"></div>
    `;
    rightPane.appendChild(toolbar);

    var lassoCheckbox = document.getElementById("lassoCheckbox");
    var questionInput = document.getElementById("questionInput");
    var submitButton = document.getElementById("submitButton");
    var statusBox = document.getElementById("statusBox");

    var defaultDragView = net.interactionHandler.options.dragView;
    var defaultZoomView = net.interactionHandler.options.zoomView;
    var defaultDragNodes = net.interactionHandler.options.dragNodes;

    var isDrawing=false;
    var lassoPoints=[];
    var selectedNodeIDs=[];

    function logStatus(msg){{
      statusBox.innerHTML+= msg+"<br/>";
      statusBox.scrollTop = statusBox.scrollHeight;
    }}

    function resizeOverlay(){{
      let rect = container.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;
    }}
    resizeOverlay();
    window.addEventListener("resize", resizeOverlay);

    lassoCheckbox.addEventListener("change", function(){{
      if(this.checked){{
        net.interactionHandler.options.dragView = false;
        net.interactionHandler.options.zoomView = false;
        net.interactionHandler.options.dragNodes = false;
        overlay.style.pointerEvents = "auto";
        logStatus("Lasso: ON");
      }} else {{
        net.interactionHandler.options.dragView = defaultDragView;
        net.interactionHandler.options.zoomView = defaultZoomView;
        net.interactionHandler.options.dragNodes = defaultDragNodes;
        overlay.style.pointerEvents = "none";
        logStatus("Lasso: OFF");
      }}
    }});

    overlay.addEventListener("mousedown", function(e){{
      if(lassoCheckbox.checked){{
        isDrawing=true;
        lassoPoints=[ getOverlayPos(e) ];
        clearOverlay();
      }}
    }});

    overlay.addEventListener("mousemove", function(e){{
      if(isDrawing && lassoCheckbox.checked){{
        lassoPoints.push(getOverlayPos(e));
        drawPolygon(lassoPoints, false);
      }}
    }});

    overlay.addEventListener("mouseup", function(e){{
      if(isDrawing && lassoCheckbox.checked){{
        isDrawing=false;
        drawPolygon(lassoPoints, true);

        // revert old lasso
        revertNodes(oldLassoNodes);
        oldLassoNodes=[];

        // also revert old gold if we want them undone on new lasso
        revertNodes(oldGoldNodes);
        oldGoldNodes=[];

        let nodeIDs = net.body.data.nodes.getIds();
        let newlySelected=[];
        for(let nid of nodeIDs){{
          let posObj = net.getPositions(nid);
          let domPos = net.canvasToDOM(posObj[nid]);
          let overlayPos = toOverlay(domPos.x, domPos.y);
          if(isInsidePolygon(overlayPos, lassoPoints)){{
            newlySelected.push(nid);
          }}
        }}
        selectedNodeIDs = newlySelected.slice();
        highlightNodes(selectedNodeIDs, 5, "#000");
        oldLassoNodes = selectedNodeIDs.slice();

        logStatus("New lasso selection: "+newlySelected.join(", "));
        lassoPoints=[];
        clearOverlay();
      }}
    }});

    submitButton.addEventListener("click", function(){{
      let q = questionInput.value.trim();
      if(!q) return;

      // revert old gold
      revertNodes(oldGoldNodes);
      oldGoldNodes=[];

      let msg = "Response to question " + q + " on selected nodes: [" + selectedNodeIDs.join(", ") + "]";
      logStatus(msg);

      let half = Math.floor(selectedNodeIDs.length/2);
      let subset = selectedNodeIDs.slice(0, half);

      highlightNodes(subset, null, "gold");
      oldGoldNodes = subset.slice();
    }});

    // ~~~~~ HELPER FUNCS ~~~~~
    function highlightNodes(nids, bw, borderC){{
      let ups=[];
      for(let nid of nids){{
        let obj={{id:nid}};
        if(bw!==null) obj.borderWidth=bw;
        if(borderC!==null) obj.color={{border:borderC}};
        ups.push(obj);
      }}
      net.body.data.nodes.update(ups);
    }}

    function revertNodes(nids){{
      let ups=[];
      for(let nid of nids){{
        let c = parseOriginalNodeColor(nid);
        ups.push({{id:nid, borderWidth:1, color:c}});
      }}
      net.body.data.nodes.update(ups);
    }}

    function clearOverlay(){{
      ctx.clearRect(0,0,overlay.width,overlay.height);
    }}

    function drawPolygon(points, closePath){{
      clearOverlay();
      ctx.beginPath();
      if(points.length>0){{
        ctx.moveTo(points[0].x, points[0].y);
        for(let i=1; i<points.length; i++){{
          ctx.lineTo(points[i].x, points[i].y);
        }}
        if(closePath) ctx.closePath();
      }}
      ctx.strokeStyle="rgba(255,50,255,0.7)";
      ctx.lineWidth=2;
      ctx.stroke();
    }}

    function getOverlayPos(e){{
      let rect=overlay.getBoundingClientRect();
      return {{x:e.clientX-rect.left, y:e.clientY-rect.top}};
    }}

    function toOverlay(docX, docY){{
      let cRect = container.getBoundingClientRect();
      return {{x:docX-cRect.left, y:docY-cRect.top}};
    }}

    function isInsidePolygon(pt, poly){{
      let inside=false;
      for(let i=0,j=poly.length-1; i<poly.length; j=i++){{
        let xi=poly[i].x, yi=poly[i].y;
        let xj=poly[j].x, yj=poly[j].y;
        let inter=((yi>pt.y)!=(yj>pt.y)) && (pt.x<(xj-xi)*(pt.y-yi)/(yj-yi)+xi);
        if(inter) inside=!inside;
      }}
      return inside;
    }}
}});
</script>
"""

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
    <!-- We now place the toolbar here, after #config, at runtime in JS. -->
  </div>
</div>
{''.join(scripts)}
{custom_js}
</body>
"""

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
    st.title("Visual Knowledge Graph Question-Answering")
    st.write("Upload a CSV with **at least 3 columns** and map them to node_1, edge, node_2.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### CSV Preview")
        st.dataframe(df.head())

        columns = list(df.columns)
        if len(columns) < 3:
            st.warning("Please upload a CSV with at least 3 columns.")
            return  # stop here

        st.write("#### Select which columns correspond to each field:")
        node1_col = st.selectbox("Select the 'node_1' column", columns)
        edge_col = st.selectbox("Select the 'edge' column", columns)
        node2_col = st.selectbox("Select the 'node_2' column", columns)

        if st.button("Generate Graph"):
            G = create_graph_from_csv(df, node1_col, edge_col, node2_col)
            G = color_communities_girvan_newman(G)
            # or color_communities_louvain(G)
            html_path = draw_graph_with_lasso_and_textbox(G, "graph_lasso.html")

            st.write("### Graph Visualization:")
            with open(html_path, 'r', encoding='utf-8') as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=1000, scrolling=True)


if __name__ == "__main__":
    main()
