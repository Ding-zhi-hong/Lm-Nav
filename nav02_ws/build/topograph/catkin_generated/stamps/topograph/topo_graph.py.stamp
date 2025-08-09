#!/usr/bin/env python3
import numpy as np
import networkx as nx
import os
import cv2
import pickle
import rospy

class TopoNode:
    def __init__(self, node_id, image, position, features):
        """
        拓扑图节点
        :param node_id: 节点ID
        :param image: ROS图像消息或图像路径
        :param position: 位置坐标 (x, y, theta)
        :param features: 特征向量
        """
        self.id = node_id
        self.image = image
        self.position = position
        self.features = features
        self.timestamp = rospy.Time.now()
        self.landmark_probs = {}  # 存储地标概率分布
    
    def get_image_path(self, save_dir):
        """保存节点图像到文件并返回路径"""
        path = os.path.join(save_dir, f"node_{self.id}.jpg")
        cv2.imwrite(path, self.image)
        return path
        
    def set_landmark_probs(self, landmark, probs):
        """设置该节点的地标概率分布"""
        self.landmark_probs[landmark] = probs

class TopoGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.node_counter = 0
        self.feature_dim = 512  # ViNG特征维度
    
    def add_node(self, image, position, features):
        """添加新节点"""
        node_id = self.node_counter
        new_node = TopoNode(node_id, image, position, features)
        self.nodes[node_id] = new_node
        self.graph.add_node(node_id, position=position, features=features)
        self.node_counter += 1
        return node_id
    
    def add_edge(self, node_id1, node_id2, distance):
        """添加边及距离"""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.graph.add_edge(node_id1, node_id2, weight=distance)
    
    def get_neighbors(self, node_id):
        """获取相邻节点"""
        return list(self.graph.neighbors(node_id))
    
    def get_edge_weight(self, node_id1, node_id2):
        """获取边权重"""
        return self.graph[node_id1][node_id2]['weight']
    
    def find_nearest_node(self, position):
        """根据位置寻找最近节点"""
        min_dist = float('inf')
        nearest_id = None
        
        for node_id, node in self.nodes.items():
            dist = np.linalg.norm(np.array(node.position[:2]) - np.array(position[:2]))
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        return nearest_id, min_dist
    
    def save(self, filename):
        """保存拓扑图到文件"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'nodes': self.nodes,
                'node_counter': self.node_counter
            }, f)
    
    def load(self, filename):
        """从文件加载拓扑图"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.nodes = data['nodes']
            self.node_counter = data['node_counter']
    
    def visualize(self):
        """可视化图结构"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        
        # 绘制节点
        positions = {n: (n.position[0], n.position[1]) for n in self.nodes.values()}
        nx.draw(self.graph, positions, with_labels=True, node_size=200, font_size=10)
        
        # 添加节点ID标签
        labels = {node.id: node.id for node in self.nodes.values()}
        nx.draw_networkx_labels(self.graph, positions, labels)
        
        # 绘制边权重
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, positions, edge_labels)
        
        plt.title("Topology Graph")
        plt.grid(True)
        plt.show()



import matplotlib.pyplot as plt
import pickle
import networkx as nx
import os
import numpy as np

# 加载拓扑图文件
file_path = "/home/rosnoetic/maps/office_map/office_map.graph"
with open(file_path, 'rb') as f:
    graph_data = pickle.load(f)
'''
# 创建NetworkX图对象
G = graph_data['graph']
nodes = graph_data['nodes']

# 设置画布
plt.figure(figsize=(12, 10))

# 创建位置字典 (使用节点存储的位置)
pos = {}
node_colors = []
node_sizes = []
labels = {}

for node_id in G.nodes():
    # 仅使用x,y位置（忽略theta）
    pos[node_id] = (nodes[node_id].position[0], nodes[node_id].position[1])
    
    # 根据节点ID设置颜色
    node_colors.append(plt.cm.tab10(node_id % 10))
    node_sizes.append(200 + 50 * len(list(G.predecessors(node_id))))
    labels[node_id] = f"{node_id}"

# 绘制拓扑图
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=1.5, 
                      connectionstyle="arc3,rad=0.1",
                      arrowstyle='->', arrowsize=15)
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

# 添加标题和网格
plt.title('Office Topological Map Visualization', fontsize=14)
plt.grid(True, alpha=0.2)
plt.xlabel('X Position (m)', fontsize=12)
plt.ylabel('Y Position (m)', fontsize=12)

# 保存和显示图像
plt.tight_layout()
output_path = "/home/rosnoetic/maps/office_map/office_visualization.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"可视化图像已保存至: {output_path}")'''