import streamlit as st
st.set_page_config(layout="wide", page_title="VKGQA")
import pandas as pd
import networkx as nx
from pyvis.network import Network

import re
import random
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
        G.add_edge(node1, node2, title=edge_label)
    return G

def color_communities_louvain(G):
    partition = community_louvain.best_partition(G)
    comm_dict = {}
    for node, comm_id in partition.items():
        comm_dict.setdefault(comm_id, []).append(node)

    communities_list = list(comm_dict.values())
    palette = sns.color_palette("pastel", len(communities_list)).as_hex()
    for idx, community_nodes in enumerate(communities_list):
        color = palette[idx]
        for node in community_nodes:
            G.nodes[node]['color'] = color
    return G

def draw_graph_reset(G, output_html="graph_reset.html"):
    """
    1) Renders a PyVis network with repulsion ~500 iterations,
    2) Freezes physics,
    3) Injects question input in same HTML,
    4) BFS on random nodes -> highlight in BLUE,
    5) Reverts edges => #121212,
    6) Reverts on user canvas-click,
    7) Replaces "->(title)->" text with clickable arrows that show a tooltip instantly,
       toggle color on click between magenta and old color, etc.
    """
    net = Network(height="750px", width="100%", notebook=False, cdn_resources="remote")
    net.from_nx(G)

    # Start with some physics on, then disable after 300 iteration stabilization
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "solver": "repulsion",
        "maxVelocity": 50,
        "stabilization": {
          "enabled": true,
          "iterations": 300
        }
      }
    }
    """)

    # Generate PyVis HTML
    net.save_graph(output_html)

    with open(output_html, "r", encoding="utf-8") as f:
        html_code = f.read()

    # 1) snippet to disable physics
    disable_physics_snippet = """
network.once("stabilizationIterationsDone", function() {
  network.setOptions({ physics: { enabled: false } });
});
"""

    replaced_html = html_code.replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options);\n" + disable_physics_snippet,
        1
    )

    # 2) BFS logic + question box + arrow handling
    injection_snippet = r"""
<!-- Some minimal CSS for arrows and instant custom tooltips -->
<style>
.arrow {
  color: blue;
  text-decoration: none;
  cursor: pointer;
  margin: 0 4px;
}
.arrow:hover {
  color: gray;
  transition: none;
}

/* Tooltip styling that appears instantly on hover */
.arrow[data-tooltip] {
  position: relative;
}

.arrow[data-tooltip]:hover::after {
  content: attr(data-tooltip);
  position: absolute;
  left: 0;
  top: -1.5em;
  background: #333;
  color: #fff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8em;
  white-space: nowrap;
  z-index: 9999;
}
</style>

<div id="questionPanel" style="width:100%; height:200px; margin-top:10px; padding:10px; border:1px solid #ccc;">
  <input type="text" id="questionInput" placeholder="Ask a question about the graph..." style="margin-left:10px;width:250px;"/>
  <button id="submitQuestionBtn">Submit</button>
  <div id="answerBox" style="margin-top:8px; font-weight:bold; color:#333;"></div>
</div>

<script>
(function(){
   var adjacency = {};
   var isHighlighted = false;

   // Build adjacency from edges
   var allNodeIds = nodes.getIds();
   allNodeIds.forEach(nid => { adjacency[nid] = []; });
   var allEdgeIds = edges.getIds();
   allEdgeIds.forEach(eid => {
     var e = edges.get(eid);
     adjacency[e.from].push(e.to);
     adjacency[e.to].push(e.from);
   });

   // BFS
   function bfsShortestPath(start, end){
     if(!adjacency[start] || !adjacency[end]) return [];
     var queue = [[start]];
     var visited = new Set([start]);
     while(queue.length>0){
       var path = queue.shift();
       var last = path[path.length-1];
       if(last===end){
         return path;
       }
       adjacency[last].forEach(nbr=>{
         if(!visited.has(nbr)){
           visited.add(nbr);
           var newp = path.slice();
           newp.push(nbr);
           queue.push(newp);
         }
       });
     }
     return [];
   }

   function revertAllEdgesAndNodes(){
     var nids = nodes.getIds();
     var nup = [];
     nids.forEach(nid=>{
       var c = nodeColors[nid];
       if(typeof c==="string"){
         nup.push({id:nid, borderWidth:1, color:{border:c, background:c}, zIndex:1});
       } else {
         nup.push({id:nid, borderWidth:1, color:c, zIndex:1});
       }
     });
     nodes.update(nup);

     var eids = edges.getIds();
     var eup = [];
     eids.forEach(eid=>{
       eup.push({id: eid, width:1, color:{color:"#121212"}, zIndex:1});
     });
     edges.update(eup);
     isHighlighted=false;
   }

   function highlightPath(path){
     var nodeUps=[];
     path.forEach(nid=>{
       nodeUps.push({id:nid, borderWidth:20, color:{border:'blue'}, zIndex:9999});
     });
     nodes.update(nodeUps);

     for(var i=0; i<path.length-1; i++){
       var A= path[i], B= path[i+1];
       var eId = edges.getIds().find(eid=>{
         var ed= edges.get(eid);
         return (ed.from===A && ed.to===B)||(ed.from===B && ed.to===A);
       });
       if(eId){
         edges.update([{id:eId, width:20, color:{color:'blue'}, zIndex:9999}]);
       }
     }
     isHighlighted=true;
   }

   // Build HTML string with clickable arrows, using data-tooltip instead of title
   function buildPathWithEdges(path){
     if(path.length<2) return JSON.stringify(path);
     var result = path[0];
     for(var i=0; i<path.length-1; i++){
       var A = path[i], B = path[i+1];
       var eId = edges.getIds().find(eid=>{
         var ed = edges.get(eid);
         return (ed.from===A && ed.to===B)||(ed.from===B && ed.to===A);
       });
       var eTitle = "";
       if(eId){
         var edata = edges.get(eId);
         eTitle = edata.title || "";
       }
       // Instead of title='<edge>', we use data-tooltip='<edge>' so tooltip appears instantly
       result += " <span class='arrow' data-edgeid='"+(eId||"")+"' data-tooltip='"+ eTitle +"'>&#8594;</span> " + B;
     }
     return result;
   }

   // Clicking on the canvas reverts if highlighted
   network.on("click", function(params){
     if(isHighlighted){
       revertAllEdgesAndNodes();
     }
   });

   // Clicking on an arrow toggles color
   document.addEventListener("click", function(ev){
     var arrow = ev.target.closest(".arrow");
     if(!arrow) return;
     // Prevent the normal revert
     ev.stopPropagation();

     var eId = arrow.getAttribute("data-edgeid");
     if(!eId) return;
     var current = edges.get(eId);
     if(!current) return;
     // Toggle color between #FF00FF and the previous color
     var oldColor = current.color.color || "#121212";
     if(oldColor.toLowerCase()==="#ff00ff"){
       // set back to arrow's "previous" color
       var revertTo = arrow.getAttribute("data-oldcolor") || "#121212";
       edges.update([{id:eId, color:{color: revertTo}}]);
       arrow.style.color = revertTo; // arrow color sync
     } else {
       // set it to magenta
       if(!arrow.hasAttribute("data-oldcolor")){
         arrow.setAttribute("data-oldcolor", oldColor);
       }
       edges.update([{id:eId, color:{color:"#FF00FF"}}]);
       arrow.style.color = "#FF00FF";
     }
   }, true);

   // Submit question
   var btn = document.getElementById("submitQuestionBtn");
   btn.addEventListener('click', function(){
     var qBox = document.getElementById("questionInput");
     var question = qBox.value.trim()||"...";
     var ansBox = document.getElementById("answerBox");
     ansBox.innerHTML = "Answer to question '"+question+"': {sample text}.";

     var nids = nodes.getIds();
     if(nids.length<2){
       ansBox.innerHTML += "<br/>Not enough nodes for path highlight!";
       return;
     }
     var A = nids[Math.floor(Math.random()*nids.length)];
     var B = nids[Math.floor(Math.random()*nids.length)];
     var tries=0;
     while(B===A && tries<10){
       B=nids[Math.floor(Math.random()*nids.length)];
       tries++;
     }
     var path = bfsShortestPath(A,B);
     if(path.length===0){
       ansBox.innerHTML += "<br/>No path from '"+A+"' to '"+B+"'. Possibly disconnected?";
       return;
     }
     var pathHTML = buildPathWithEdges(path);
     ansBox.innerHTML += "<br/>Reasoning path: "+ pathHTML;

     revertAllEdgesAndNodes();
     highlightPath(path);
   });
})();
</script>
"""

    # Insert that snippet at end
    if "</body>" in replaced_html:
        final_html = replaced_html.replace("</body>", injection_snippet + "\n</body>", 1)
    else:
        final_html = replaced_html + "\n" + injection_snippet

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    return output_html


def main():
    st.title("Visual Knowledge Graph Question-Answering")
    st.write("Upload a CSV or TXT file with at least 3 columns or triplets.")

    uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv","txt"])
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8", errors="replace")
            lines = raw_text.splitlines()
            rows = []
            for line in lines:
                parts = line.split('|')
                if len(parts)<3:
                    continue
                node1, edge, node2 = (p.strip() for p in parts)
                rows.append([node1, edge, node2])
            df = pd.DataFrame(rows, columns=["node_1","edge","node_2"])

        st.write("#### (Debug) Number of edges to keep:")
        num_lines = st.text_input("(Leave blank or zero to keep all)", "")
        try:
            n_val = int(num_lines)
            if n_val>0:
                df = df.iloc[:n_val].copy()
        except ValueError:
            pass

        st.write("### Data Preview")
        st.dataframe(df.head())

        columns = list(df.columns)
        if len(columns)<3:
            st.warning("Please upload a file with at least 3 columns/triplets.")
            return

        st.write("#### Select the columns for node_1, edge, node_2")
        node1_col = st.selectbox("node_1 column", columns)
        edge_col  = st.selectbox("edge column", columns)
        node2_col = st.selectbox("node_2 column", columns)

        if st.button("Generate Graph"):
            G = create_graph_from_csv(df, node1_col, edge_col, node2_col)
            G = color_communities_louvain(G)
            html_path = draw_graph_reset(G, "graph_reset.html")

            st.write("### Graph Visualization:")
            with open(html_path, "r", encoding="utf-8") as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=1000, scrolling=False)


if __name__=="__main__":
    main()
