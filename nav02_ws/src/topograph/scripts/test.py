#!/usr/bin/env python3

import numpy as np
import os
import cv2
import dill
import torch
import clip
import time
import math
import networkx as nx
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import glob
import shutil

class TopoNode:
    def __init__(self, node_id, image, position, features, timestamp=None):
        self.id = node_id
        self.image = image
        self.position = position
        self.features = features
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.landmark_probs = {}
    
    def get_image_path(self, save_dir):
        path = os.path.join(save_dir, f"node_{self.id}.jpg")
        cv2.imwrite(path, self.image)
        return path
        
    def set_landmark_probs(self, landmark, probs):
        self.landmark_probs[landmark] = probs

class TopoGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = {}
        self.node_counter = 0
        self.feature_dim = 512
    
    def add_node(self, image, position, features):
        node_id = self.node_counter
        features = features.flatten()
        new_node = TopoNode(node_id, image, position, features)
        self.nodes[node_id] = new_node
        self.graph.add_node(node_id, position=position, features=features)
        self.node_counter += 1
        return node_id
    
    def add_edge(self, node_id1, node_id2, distance):
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.graph.add_edge(node_id1, node_id2, weight=distance)
    
    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))
    
    def get_edge_weight(self, node_id1, node_id2):
        return self.graph[node_id1][node_id2]['weight']
    
    def find_nearest_node(self, position):
        min_dist = float('inf')
        nearest_id = None
        
        for node_id, node in self.nodes.items():
            dist = np.linalg.norm(np.array(node.position[:2]) - np.array(position[:2]))
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        return nearest_id, min_dist
    
    def save(self, filename):
        serializable_data = {
            'graph': self._serialize_graph(),
            'nodes': self._serialize_nodes(),
            'node_counter': self.node_counter
        }
        
        # 使用临时文件确保写入完整
        temp_filename = filename + '.tmp'
        with open(temp_filename, 'wb') as f:
            dill.dump(serializable_data, f)
        
        # 重命名替换原文件
        os.rename(temp_filename, filename)
        print(f"拓扑图已保存至: {filename}")
    
    def _serialize_graph(self):
        serialized_graph = {
            'nodes': {},
            'edges': []
        }
        
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            serialized_graph['nodes'][node_id] = {
                'position': node_data['position'],
                'features': node_data['features'].tolist()
            }
        
        for u, v, data in self.graph.edges(data=True):
            serialized_graph['edges'].append({
                'source': u,
                'target': v,
                'weight': data['weight']
            })
        
        return serialized_graph
    
    def _serialize_nodes(self):
        serialized_nodes = {}
        for node_id, node in self.nodes.items():
            # 确保图像数据是可序列化的
            if isinstance(node.image, np.ndarray):
                image_data = node.image.tolist()
            elif isinstance(node.image, list):
                image_data = node.image
            else:
                image_data = []
            
            serialized_nodes[node_id] = {
                'id': node.id,
                'image': image_data,
                'position': node.position,
                'features': node.features.tolist(),
                'timestamp': node.timestamp,
                'landmark_probs': node.landmark_probs
            }
        return serialized_nodes
    
    def load(self, filename):
        # 检查文件是否存在且不为空
        if not os.path.exists(filename):
            raise FileNotFoundError(f"图文件不存在: {filename}")
        
        file_size = os.path.getsize(filename)
        if file_size == 0:
            raise ValueError(f"图文件为空: {filename}")
        
        try:
            with open(filename, 'rb') as f:
                data = dill.load(f)
                self.node_counter = data['node_counter']
                self.graph = self._deserialize_graph(data['graph'])
                self.nodes = self._deserialize_nodes(data['nodes'])
            print(f"成功加载拓扑图: {filename} (包含 {len(self.nodes)} 个节点)")
            return True
        except EOFError:
            raise RuntimeError(f"图文件损坏或格式不正确: {filename}")
        except Exception as e:
            raise RuntimeError(f"加载图文件时出错: {e}")
    
    def _deserialize_graph(self, serialized_graph):
        graph = nx.Graph()
        
        for node_id, node_data in serialized_graph['nodes'].items():
            graph.add_node(node_id, 
                           position=node_data['position'],
                           features=np.array(node_data['features']))
        
        for edge in serialized_graph['edges']:
            graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        return graph
    
    def _deserialize_nodes(self, serialized_nodes):
        nodes = {}
        for node_id, node_data in serialized_nodes.items():
            # 处理图像数据
            if isinstance(node_data['image'], list):
                try:
                    image = np.array(node_data['image'], dtype=np.uint8)
                except:
                    print(f"警告: 节点 {node_id} 的图像数据转换失败，使用占位图像")
                    image = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                image = np.zeros((128, 128, 3), dtype=np.uint8)
            
            node = TopoNode(
                node_id=node_data['id'],
                image=image,
                position=node_data['position'],
                features=np.array(node_data['features']),
                timestamp=node_data['timestamp']
            )
            node.landmark_probs = node_data['landmark_probs']
            nodes[node_id] = node
        return nodes
    
    def print_landmark_probs(self, limit=None):
        """打印所有节点的地标概率分布"""
        print("\n=== 地标概率分布检查 ===")
        print(f"节点总数: {len(self.nodes)}")
        
        for i, (node_id, node) in enumerate(self.nodes.items()):
            if limit is not None and i >= limit:
                print(f"... 以及{len(self.nodes)-limit}个更多节点")
                break
                
            print(f"\n节点 {node_id} 位置: {node.position}")
            
            if node.landmark_probs:
                # 按概率降序排序
                sorted_probs = sorted(
                    node.landmark_probs.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # 仅显示概率大于0.01的地标
                for landmark, prob in sorted_probs:
                    if prob > 0.01:
                        print(f"  {landmark:<15}: {prob:.4f}")
            else:
                print("  无地标概率数据")

class VNMLandmarkMatcher:
    def __init__(self, graph_path, device='cuda'):
        self.graph_path = graph_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.map_dir = os.path.dirname(graph_path)
        
        print("加载CLIP模型...")
        self.model, self.preprocess = self._init_clip()
        
        print(f"加载拓扑图: {graph_path}")
        self.topo_graph = TopoGraph()
        
        # 尝试加载主图文件
        try:
            if self.topo_graph.load(graph_path):
                return
        except Exception as e:
            print(f"加载主图文件失败: {e}")
        
        # 尝试查找并加载备份文件
        backup_files = self.find_backup_files()
        if backup_files:
            print(f"找到 {len(backup_files)} 个备份文件，尝试加载最新的...")
            latest_backup = backup_files[0]  # 最新备份
            try:
                if self.topo_graph.load(latest_backup):
                    print(f"成功从备份加载拓扑图: {latest_backup}")
                    # 将备份文件复制为主图文件
                    shutil.copy(latest_backup, graph_path)
                    print(f"已将备份恢复为主图文件: {graph_path}")
                    return
            except Exception as e:
                print(f"加载备份文件失败: {e}")
        
        # 如果所有尝试都失败，创建新的空拓扑图
        print("无法加载任何拓扑图文件，创建新的空拓扑图")
        self.topo_graph = TopoGraph()
    
    def find_backup_files(self):
        """在相同目录下查找所有备份文件"""
        pattern = os.path.join(self.map_dir, "*.graph_*")
        backup_files = glob.glob(pattern)
        
        # 按修改时间排序（最新的在前）
        backup_files.sort(key=os.path.getmtime, reverse=True)
        return backup_files
    
    def _init_clip(self):
        model, _ = clip.load("ViT-B/32", device=self.device)
        custom_preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711)),
        ])
        return model, custom_preprocess

    def process_landmarks(self, landmarks):
        if not landmarks:
            print("警告: 没有提供地标")
            return None, None
            
        print(f"处理地标: {', '.join(landmarks)}")
        text_features = self._encode_text(landmarks)
        total_nodes = len(self.topo_graph.nodes)
        
        if total_nodes == 0:
            print("警告: 拓扑图中没有节点")
            return None, None
            
        start_time = time.time()
        print(f"处理 {total_nodes} 个节点...")
        
        # 调整批次大小以避免内存问题
        batch_size = min(8, total_nodes)
        if total_nodes > 50:
            batch_size = 8
        elif total_nodes > 20:
            batch_size = 4
        else:
            batch_size = 2
        
        print(f"使用批次大小: {batch_size}")
        
        node_ids = list(self.topo_graph.nodes.keys())
        batch_probs = [None] * len(node_ids)
        
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i+batch_size]
            images = self._load_images_for_batch(batch_ids)
            batch_results = self._process_image_batch(images, text_features, landmarks)
            
            for j in range(len(batch_ids)):
                idx = i + j
                batch_probs[idx] = dict(zip(landmarks, batch_results[j].tolist()))
                
            processed = min(i + batch_size, total_nodes)
            print(f"已处理 {processed}/{total_nodes} 节点")
        
        # 更新节点
        for node_id, probs in zip(node_ids, batch_probs):
            self.topo_graph.nodes[node_id].landmark_probs = probs
        
        # 保存更新后的拓扑图
        timestamp = int(time.time())
        backup_path = self.graph_path.replace(".graph", f"_{timestamp}.graph")
        self.topo_graph.save(backup_path)
        print(f"创建备份文件: {backup_path}")
        
        self.topo_graph.save(self.graph_path)
        elapsed = time.time() - start_time
        print(f"完成! 更新 {total_nodes} 个节点耗时 {elapsed:.2f} 秒")
        
        # 返回新图路径和时间戳
        return backup_path, timestamp

    def _load_images_for_batch(self, batch_ids):
        images = []
        for node_id in batch_ids:
            img_data = self.topo_graph.nodes[node_id].image
            
            if isinstance(img_data, str):
                if os.path.exists(img_data):
                    image = cv2.imread(img_data)
                    if image is not None:
                        images.append(image)
                        continue
                images.append(np.zeros((128, 128, 3), dtype=np.uint8))
            elif isinstance(img_data, list):
                try:
                    images.append(np.array(img_data, dtype=np.uint8))
                except:
                    images.append(np.zeros((128, 128, 3), dtype=np.uint8))
            elif isinstance(img_data, np.ndarray):
                images.append(img_data)
            else:
                images.append(np.zeros((128, 128, 3), dtype=np.uint8))
        return images

    def _encode_text(self, landmarks):
        with torch.no_grad():
            print("编码地标文本...")
            text_tokens = clip.tokenize(landmarks).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _process_image_batch(self, images, text_features, landmarks):
        image_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                # 处理数据类型
                if img.dtype == np.int32:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    img = img.astype(np.uint8)
                
                if img.dtype != np.uint8:
                    try:
                        img = img.astype(np.uint8)
                    except:
                        img = np.zeros((128, 128, 3), dtype=np.uint8)
                
                # 确保正确的颜色通道
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            
            try:
                pil_img = Image.fromarray(img)
                image_tensor = self.preprocess(pil_img).to(self.device)
                image_tensors.append(image_tensor)
            except Exception as e:
                print(f"图像处理错误: {e}, 使用占位图像")
                image_tensor = self.preprocess(Image.new('RGB', (128, 128))).to(self.device)
                image_tensors.append(image_tensor)
        
        image_tensors = torch.stack(image_tensors)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T) * 100
            probs = similarity.softmax(dim=-1)
        
        return probs.cpu().numpy()

def main():
    GRAPH_PATH = "/home/rosnoetic/maps/office_map/office_map.graph"
    LANDMARKS = ["wall", "black cube"]
    
    print("启动地标匹配处理...")
    try:
        matcher = VNMLandmarkMatcher(GRAPH_PATH)
        if len(matcher.topo_graph.nodes) == 0:
            print("错误: 拓扑图中没有节点，无法处理地标")
            print("请确保您有有效的拓扑图文件")
            return
            
        backup_path, timestamp = matcher.process_landmarks(LANDMARKS)
        
        if backup_path is None:
            print("地标处理未完成")
            return
            
        print("\n加载并检查更新后的拓扑图...")
        updated_graph = TopoGraph()
        try:
            updated_graph.load(backup_path)
            # 打印每个节点的地标概率分布
            updated_graph.print_landmark_probs(limit=10)  # 最多显示10个节点
        except Exception as e:
            print(f"加载更新后的拓扑图失败: {e}")
        
        print("\n地标匹配处理完成!")
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()