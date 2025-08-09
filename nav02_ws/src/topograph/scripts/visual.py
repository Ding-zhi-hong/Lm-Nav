import networkx as nx
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
import math
import os

def visualize_graphml(graphml_path):
    # 解析GraphML文件
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    
    # 注册命名空间
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    
    # 创建有向图
    G = nx.Graph()
    
    # 存储节点位置和朝向
    positions = {}
    orientations = {}
    
    # 解析所有节点
    for node in root.findall('.//graphml:node', ns):
        node_id = node.get('id')
        x = float(node.find(".//graphml:data[@key='d0']", ns).text)
        y = float(node.find(".//graphml:data[@key='d1']", ns).text)
        theta = float(node.find(".//graphml:data[@key='d2']", ns).text)
        
        G.add_node(node_id)
        positions[node_id] = (x, y)
        orientations[node_id] = theta
    
    # 解析所有边
    for edge in root.findall('.//graphml:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        weight = float(edge.find(".//graphml:data[@key='d3']", ns).text)
        
        G.add_edge(source, target, weight=weight)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # 绘制节点
    nx.draw_networkx_nodes(G, positions, node_size=500, node_color='skyblue')
    
    # 绘制边（用权重控制线宽）
    edge_widths = [G[u][v]['weight'] * 20 for u, v in G.edges()]
    nx.draw_networkx_edges(G, positions, width=edge_widths, edge_color='gray')
    
    # 绘制节点朝向箭头
    for node, pos in positions.items():
        theta = orientations[node]
        dx = math.cos(theta) * 0.3
        dy = math.sin(theta) * 0.3
        ax.arrow(pos[0], pos[1], dx, dy, head_width=0.15, head_length=0.2, fc='red', ec='red')
    
    # 添加标签
    labels = {node: f"{node}\n({pos[0]:.2f}, {pos[1]:.2f})\nθ: {orientations[node]:.2f}" 
              for node, pos in positions.items()}
    nx.draw_networkx_labels(G, positions, labels=labels, font_size=9)
    
    # 添加边权重标签
    for (u, v), data in G.edges.items():
        weight = data['weight']
        x = (positions[u][0] + positions[v][0]) / 2
        y = (positions[u][1] + positions[v][1]) / 2
        plt.text(x, y, f"{weight:.4f}", fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 设置图形属性
    plt.title('Office Map Visualization')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(graphml_path), 'office_map_visualization.png')
    plt.savefig(output_path, dpi=120)
    plt.close()
    
    return output_path

if __name__ == "__main__":
    graphml_file = "/home/rosnoetic/maps/office_map/office_map.graphml"
    output_image = visualize_graphml(graphml_file)
    print(f"Visualization saved to: {output_image}")