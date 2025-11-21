# KGE-HAKE 代码深度解析

## 目录
1. [整体架构概览](#整体架构概览)
2. [数据流分析](#数据流分析)
3. [核心模块详解](#核心模块详解)
4. [模型实现深度剖析](#模型实现深度剖析)
5. [训练流程详解](#训练流程详解)
6. [可扩展性分析](#可扩展性分析)
7. [联邦学习集成方案](#联邦学习集成方案)
8. [数据集更换指南](#数据集更换指南)

---

## 整体架构概览

KGE-HAKE采用模块化设计，主要包含三个核心模块：

```
KGE-HAKE/
├── codes/
│   ├── data.py      # 数据处理模块（数据读取、负采样、数据集类）
│   ├── models.py    # 模型定义模块（KGEModel基类、HAKE、ModE）
│   └── runs.py      # 训练/测试流程控制模块
└── data/            # 数据集目录
    ├── entities.dict
    ├── relations.dict
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

### 设计模式
- **抽象基类模式**: `KGEModel`作为基类，定义统一接口
- **策略模式**: 通过`BatchType`枚举控制不同的批处理策略
- **迭代器模式**: `BidirectionalOneShotIterator`实现双向采样

---

## 数据流分析

### 1. 数据加载流程

```
文件系统 → DataReader → TrainDataset/TestDataset → DataLoader → 训练/测试
```

#### 阶段1: 数据读取 (`DataReader`)

**关键代码位置**: `data.py:21-59`

```python
class DataReader:
    def __init__(self, data_path: str):
        # 读取实体和关系的字典映射
        self.entity_dict = self.read_dict(entity_dict_path)  # {entity_name: id}
        self.relation_dict = self.read_dict(relation_dict_path)  # {relation_name: id}
        
        # 读取三元组数据并转换为ID
        self.train_data = self.read_data(...)  # [(h_id, r_id, t_id), ...]
```

**数据格式要求**:
- `entities.dict`: `id\tentity_name`
- `relations.dict`: `id\trelation_name`
- `train/valid/test.txt`: `head\trelation\ttail`

**可扩展性要点**:
- ✅ 字典映射机制使得实体/关系名称可以是任意字符串
- ✅ 支持自定义数据格式（只需修改`read_data`方法）
- ⚠️ 当前硬编码为tab分隔，可改为配置化

#### 阶段2: 训练数据准备 (`TrainDataset`)

**关键代码位置**: `data.py:62-170`

**核心功能**:
1. **负采样策略** (`__getitem__`):
   - 根据`BatchType`决定是替换head还是tail
   - 使用`hr_map`和`tr_map`避免采样到正样本（过滤策略）

2. **子采样权重** (`subsampling_weight`):
   ```python
   subsampling_weight = sqrt(1 / (hr_freq[(h,r)] + tr_freq[(t,r)]))
   ```
   - 降低高频三元组的影响，防止模型过度拟合常见模式

3. **两元组统计** (`two_tuple_count`):
   - `hr_map`: {(h, r): [t1, t2, ...]} - 用于tail负采样过滤
   - `tr_map`: {(t, r): [h1, h2, ...]} - 用于head负采样过滤
   - `hr_freq/tr_freq`: 频率统计，用于子采样权重

**数据流示例**:
```
输入: (h=0, r=1, t=2), BatchType=HEAD_BATCH, neg_size=1024
  ↓
1. 计算subsampling_weight: sqrt(1/(hr_freq[(0,1)] + tr_freq[(2,1)]))
  ↓
2. 生成负样本候选: random.randint(0, num_entity, size=2048)
  ↓
3. 过滤: 移除tr_map[(2,1)]中的实体（避免负样本是正样本）
  ↓
4. 输出: (pos_triple, neg_triples, subsampling_weight, batch_type)
```

#### 阶段3: 测试数据准备 (`TestDataset`)

**关键代码位置**: `data.py:173-220`

**与训练数据的区别**:
- 不需要负采样（使用所有实体作为候选）
- 使用`filter_bias`标记：0表示有效负样本，-1表示需要过滤的正样本
- `triple_set`包含train+valid+test，用于过滤已存在的三元组

**过滤机制**:
```python
# HEAD_BATCH: 预测 (?, r, t)
tmp = [(0, rand_head) if (rand_head, r, t) not in triple_set 
       else (-1, head) for rand_head in range(num_entity)]
# 0: 有效负样本，-1: 需要过滤（是正样本）
```

---

## 核心模块详解

### 1. BatchType 枚举系统

**设计目的**: 支持不同的批处理模式

```python
class BatchType(Enum):
    HEAD_BATCH = 0   # 训练时替换head: (?, r, t)
    TAIL_BATCH = 1   # 训练时替换tail: (h, r, ?)
    SINGLE = 2       # 测试时: 完整三元组
```

**为什么需要双向采样？**
- 知识图谱链接预测是双向的：既需要预测tail，也需要预测head
- 交替使用HEAD_BATCH和TAIL_BATCH可以提高模型对两个方向的预测能力

### 2. BidirectionalOneShotIterator

**关键代码位置**: `data.py:223-244`

```python
class BidirectionalOneShotIterator:
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            return next(self.iterator_head)  # HEAD_BATCH
        else:
            return next(self.iterator_tail)  # TAIL_BATCH
```

**设计亮点**:
- 实现无限迭代器（`one_shot_iterator`内部有`while True`）
- 自动交替使用两种批类型
- 无需手动管理epoch

---

## 模型实现深度剖析

### 1. KGEModel 基类设计

**关键代码位置**: `models.py:14-256`

#### 抽象接口设计

```python
class KGEModel(nn.Module, ABC):
    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """子类必须实现此方法，定义评分函数"""
        ...
```

**为什么使用抽象基类？**
- 统一接口：所有KGE模型都遵循相同的调用模式
- 代码复用：`forward`、`train_step`、`test_step`可以共享
- 易于扩展：添加新模型只需实现`func`方法

#### forward 方法：嵌入查找与维度变换

**关键代码位置**: `models.py:43-127`

**三种批类型的张量形状**:

| BatchType | head shape | relation shape | tail shape |
|-----------|------------|----------------|------------|
| SINGLE | [B, 1, D] | [B, 1, D] | [B, 1, D] |
| HEAD_BATCH | [B, N, D] | [B, 1, D] | [B, 1, D] |
| TAIL_BATCH | [B, 1, D] | [B, 1, D] | [B, N, D] |

其中：
- B: batch_size
- N: negative_sample_size
- D: hidden_dim (或hidden_dim*2/3，取决于模型)

**实现技巧**:
```python
# HEAD_BATCH: 需要为每个负样本head查找嵌入
head = torch.index_select(
    self.entity_embedding,
    dim=0,
    index=head_part.view(-1)  # 展平: [B*N]
).view(batch_size, negative_sample_size, -1)  # 重塑: [B, N, D]
```

#### train_step: 训练步骤实现

**关键代码位置**: `models.py:130-171`

**损失函数设计**:

```python
# 1. 正样本得分
positive_score = model(positive_sample)  # [B, 1]
positive_score = F.logsigmoid(positive_score).squeeze(dim=1)  # [B]

# 2. 负样本得分（使用自对抗负采样）
negative_score = model((positive_sample, negative_sample), batch_type)  # [B, N]
# 自对抗权重
weights = F.softmax(negative_score * adversarial_temperature, dim=1).detach()
negative_score = (weights * F.logsigmoid(-negative_score)).sum(dim=1)  # [B]

# 3. 加权损失
loss = -((subsampling_weight * positive_score).sum() + 
         (subsampling_weight * negative_score).sum()) / (2 * subsampling_weight.sum())
```

**关键设计点**:
1. **自对抗负采样**: 使用`softmax(score * temperature)`作为权重，让模型更关注难负样本
2. **子采样权重**: 降低高频三元组的影响
3. **双向损失**: 同时优化正样本和负样本的得分

#### test_step: 评估步骤实现

**关键代码位置**: `models.py:174-256`

**评估流程**:
1. 为每个测试三元组生成所有候选实体
2. 计算所有候选的得分
3. 使用`filter_bias`过滤已存在的三元组
4. 排序并计算排名指标（MRR, HITS@K）

**过滤机制的重要性**:
```python
score += filter_bias  # -1的候选会被推到最低分
```
这确保评估时不会将训练/验证/测试集中已存在的三元组视为负样本。

### 2. HAKE 模型实现

**关键代码位置**: `models.py:295-365`

#### 嵌入维度设计

```python
# 实体嵌入: [num_entity, hidden_dim * 2]
# - 前hidden_dim维: phase部分（相位）
# - 后hidden_dim维: modulus部分（模长）

# 关系嵌入: [num_relation, hidden_dim * 3]
# - 前hidden_dim维: phase部分
# - 中间hidden_dim维: modulus部分
# - 后hidden_dim维: bias部分（偏置）
```

**为什么这样设计？**
- HAKE的核心思想是同时建模层次结构的两种模式：
  1. **模长部分（modulus）**: 捕获层次深度（如：国家 > 城市 > 街道）
  2. **相位部分（phase）**: 捕获同一层次内的语义关系（如：兄弟关系）

#### 评分函数实现

```python
def func(self, head, rel, tail, batch_type):
    # 1. 分离嵌入
    phase_head, mod_head = torch.chunk(head, 2, dim=2)
    phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
    phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)
    
    # 2. 归一化相位到[-π, π]
    phase_head = phase_head / (self.embedding_range.item() / self.pi)
    
    # 3. 计算相位得分（旋转）
    if batch_type == BatchType.HEAD_BATCH:
        phase_score = phase_head + (phase_relation - phase_tail)
    else:
        phase_score = (phase_head + phase_relation) - phase_tail
    phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
    
    # 4. 计算模长得分（层次）
    mod_relation = torch.abs(mod_relation)  # 确保非负
    bias_relation = torch.clamp(bias_relation, max=1)
    # 处理bias_relation < -mod_relation的情况
    indicator = (bias_relation < -mod_relation)
    bias_relation[indicator] = -mod_relation[indicator]
    
    r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
    r_score = torch.norm(r_score, dim=2) * self.modulus_weight
    
    # 5. 综合得分
    return self.gamma.item() - (phase_score + r_score)
```

**数学原理**:
- **相位部分**: `sin((h_p + r_p - t_p) / 2)` 捕获旋转关系
- **模长部分**: `||h_m * (r_m + b_r) - t_m * (1 - b_r)||` 捕获层次关系
- **bias_relation**: 控制关系是"放大"还是"缩小"层次距离

### 3. ModE 模型实现（基线）

**关键代码位置**: `models.py:259-292`

```python
def func(self, head, rel, tail, batch_type):
    return self.gamma.item() - torch.norm(head * rel - tail, p=1, dim=2)
```

**设计对比**:
- ModE: 简单的元素级乘法 `h * r ≈ t`
- HAKE: 复杂的双部分设计，能同时建模层次和语义关系

---

## 训练流程详解

### 主训练循环 (`runs.py:230-271`)

```python
for step in range(init_step, args.max_steps):
    # 1. 训练一步
    log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
    
    # 2. 学习率衰减（warm-up后）
    if step >= warm_up_steps:
        current_learning_rate = current_learning_rate / 10
        optimizer = torch.optim.Adam(..., lr=current_learning_rate)
        warm_up_steps = warm_up_steps * 3  # 延迟下次衰减
    
    # 3. 保存检查点
    if step % args.save_checkpoint_steps == 0:
        save_model(...)
    
    # 4. 验证
    if args.do_valid and step % args.valid_steps == 0:
        metrics = kge_model.test_step(...)
```

**关键设计**:
- **Warm-up策略**: 前50%步数使用初始学习率，之后衰减
- **检查点保存**: 定期保存模型状态和嵌入
- **验证频率**: 可配置的验证间隔

### 模型保存机制

**关键代码位置**: `runs.py:68-95`

保存内容：
1. `config.json`: 所有超参数
2. `checkpoint`: 模型状态、优化器状态、训练步数
3. `entity_embedding.npy`: 实体嵌入（便于后续使用）
4. `relation_embedding.npy`: 关系嵌入

---

## 可扩展性分析

### 1. 数据接口扩展点

#### 当前限制
- 硬编码文件格式（tab分隔）
- 固定文件命名（entities.dict, relations.dict等）
- 仅支持单机数据加载

#### 扩展方案

**方案A: 支持多种数据格式**
```python
class DataReader:
    def __init__(self, data_path, format='standard'):
        if format == 'standard':
            self.read_standard_format(data_path)
        elif format == 'rdf':
            self.read_rdf_format(data_path)
        elif format == 'json':
            self.read_json_format(data_path)
```

**方案B: 支持流式数据加载**
```python
class StreamingDataReader(DataReader):
    def __init__(self, data_source):
        # 支持从数据库、API等流式读取
        self.data_stream = self.create_stream(data_source)
```

**方案C: 支持分布式数据加载**
```python
class DistributedDataReader(DataReader):
    def __init__(self, data_path, rank, world_size):
        # 每个进程只加载部分数据
        self.train_data = self.load_shard(data_path, rank, world_size)
```

### 2. 模型扩展点

#### 当前架构优势
- ✅ 抽象基类设计，易于添加新模型
- ✅ `func`方法接口清晰
- ✅ 嵌入查找逻辑可复用

#### 扩展示例：添加新模型

```python
class TransE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma):
        super().__init__()
        self.entity_embedding = nn.Parameter(...)
        self.relation_embedding = nn.Parameter(...)
    
    def func(self, head, rel, tail, batch_type):
        return self.gamma.item() - torch.norm(head + rel - tail, p=2, dim=2)
```

### 3. 训练流程扩展点

#### 当前限制
- 固定训练循环结构
- 单一优化器（Adam）
- 简单的学习率衰减策略

#### 扩展方案

**方案A: 支持多种优化器**
```python
def create_optimizer(model, args):
    if args.optimizer == 'adam':
        return torch.optim.Adam(...)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(...)
    elif args.optimizer == 'adagrad':
        return torch.optim.Adagrad(...)
```

**方案B: 支持学习率调度器**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.max_steps
)
```

---

## 联邦学习集成方案

### 1. 架构设计

联邦学习需要将KGE-HAKE改造为支持分布式训练：

```
联邦服务器 (Aggregator)
    ↕
客户端1 (Client 1) ←→ 本地数据1
客户端2 (Client 2) ←→ 本地数据2
客户端3 (Client 3) ←→ 本地数据3
```

### 2. 关键改造点

#### 2.1 数据分片策略

**问题**: 不同客户端可能有不同的实体/关系集合

**解决方案**:
```python
class FederatedDataReader(DataReader):
    def __init__(self, data_path, client_id, num_clients):
        super().__init__(data_path)
        # 每个客户端只看到部分数据
        self.train_data = self.partition_data(
            self.train_data, client_id, num_clients
        )
        
        # 但需要全局的实体/关系字典（或通过服务器同步）
        self.global_entity_dict = self.load_global_dict()
        self.local_entity_set = set([h for h, _, _ in self.train_data] + 
                                    [t for _, _, t in self.train_data])
```

#### 2.2 模型参数同步

**关键挑战**:
- 实体嵌入矩阵大小取决于全局实体数量
- 不同客户端可能只训练部分实体

**解决方案A: 全局嵌入矩阵**
```python
class FederatedHAKE(HAKE):
    def __init__(self, global_entity_dict, ...):
        # 所有客户端共享相同的嵌入矩阵大小
        num_entity = len(global_entity_dict)
        super().__init__(num_entity, ...)
        
    def get_local_gradients(self, local_data):
        # 只计算本地数据涉及的实体的梯度
        loss = self.compute_loss(local_data)
        loss.backward()
        # 返回梯度字典 {entity_id: gradient, ...}
        return self.extract_local_gradients()
```

**解决方案B: 参数掩码**
```python
def train_step_federated(model, local_data, local_entity_mask):
    # local_entity_mask: [num_entity] bool tensor
    # 只更新本地实体对应的参数
    loss = compute_loss(model, local_data)
    loss.backward()
    
    # 掩码梯度
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            if 'entity_embedding' in param_name:
                param.grad[~local_entity_mask] = 0
```

#### 2.3 联邦聚合策略

**FedAvg算法适配**:
```python
class FederatedAggregator:
    def aggregate(self, client_models, client_weights):
        """
        client_models: List[Dict] - 每个客户端的模型参数
        client_weights: List[float] - 每个客户端的权重（通常为数据量）
        """
        aggregated_state = {}
        
        for key in client_models[0].keys():
            if 'entity_embedding' in key:
                # 实体嵌入：加权平均（只聚合本地实体对应的部分）
                aggregated_state[key] = self.aggregate_entity_embeddings(
                    [m[key] for m in client_models], client_weights
                )
            else:
                # 其他参数：标准FedAvg
                aggregated_state[key] = sum(
                    w * m[key] for w, m in zip(client_weights, client_models)
                ) / sum(client_weights)
        
        return aggregated_state
```

#### 2.4 实体对齐问题

**问题**: 不同客户端可能有不同的实体ID映射

**解决方案**:
```python
class EntityAlignment:
    def __init__(self):
        self.global_to_local = {}  # {global_id: local_id}
        self.local_to_global = {}  # {local_id: global_id}
    
    def align_embeddings(self, local_embeddings, alignment_map):
        """
        将本地嵌入对齐到全局空间
        """
        global_embeddings = torch.zeros(
            self.global_entity_count, 
            local_embeddings.shape[1]
        )
        for local_id, global_id in alignment_map.items():
            global_embeddings[global_id] = local_embeddings[local_id]
        return global_embeddings
```

### 3. 实现示例框架

```python
# federated_train.py
class FederatedKGETrainer:
    def __init__(self, args, client_id, num_clients):
        self.client_id = client_id
        self.num_clients = num_clients
        
        # 初始化数据（分片）
        self.data_reader = FederatedDataReader(
            args.data_path, client_id, num_clients
        )
        
        # 初始化模型（全局大小）
        self.model = HAKE(
            num_entity=len(self.data_reader.global_entity_dict),
            ...
        )
        
        # 实体掩码（标识本地实体）
        self.local_entity_mask = self.create_local_mask()
    
    def local_train(self, num_steps):
        """本地训练"""
        for step in range(num_steps):
            loss = self.model.train_step(...)
            # 只更新本地实体的梯度
            self.mask_gradients()
            self.optimizer.step()
        
        return self.model.state_dict()
    
    def aggregate_and_update(self, aggregated_state):
        """接收聚合后的参数并更新"""
        self.model.load_state_dict(aggregated_state)
```

### 4. 隐私保护考虑

**差分隐私**:
```python
def add_noise_to_gradients(gradients, noise_scale):
    """添加拉普拉斯噪声"""
    noise = torch.randn_like(gradients) * noise_scale
    return gradients + noise
```

**安全聚合**:
- 使用同态加密保护梯度传输
- 使用安全多方计算进行聚合

---

## 数据集更换指南

### 1. 数据格式要求

#### 标准格式
```
data/
├── entities.dict      # id\tentity_name
├── relations.dict     # id\trelation_name
├── train.txt         # head\trelation\ttail
├── valid.txt
└── test.txt
```

#### 格式转换工具

```python
# convert_dataset.py
class DatasetConverter:
    @staticmethod
    def from_rdf(rdf_file, output_dir):
        """从RDF格式转换"""
        entities = set()
        relations = set()
        triples = []
        
        # 解析RDF文件（使用rdflib等）
        # ... 解析逻辑 ...
        
        # 生成字典
        entity_dict = {e: i for i, e in enumerate(sorted(entities))}
        relation_dict = {r: i for i, r in enumerate(sorted(relations))}
        
        # 保存
        save_dict(entity_dict, f"{output_dir}/entities.dict")
        save_dict(relation_dict, f"{output_dir}/relations.dict")
        save_triples(triples, entity_dict, relation_dict, f"{output_dir}/train.txt")
    
    @staticmethod
    def from_json(json_file, output_dir):
        """从JSON格式转换"""
        # 实现JSON到标准格式的转换
        pass
    
    @staticmethod
    def from_neo4j(neo4j_uri, output_dir):
        """从Neo4j图数据库导出"""
        # 使用neo4j driver查询并转换
        pass
```

### 2. 数据预处理扩展

#### 支持动态实体/关系发现

```python
class FlexibleDataReader(DataReader):
    def __init__(self, data_path, auto_discover=True):
        if auto_discover:
            # 自动从train/valid/test中发现所有实体和关系
            self.entity_dict, self.relation_dict = self.discover_vocab(data_path)
        else:
            # 使用预定义的字典
            super().__init__(data_path)
    
    def discover_vocab(self, data_path):
        """从数据文件中自动发现词汇表"""
        entities = set()
        relations = set()
        
        for filename in ['train.txt', 'valid.txt', 'test.txt']:
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    for line in f:
                        h, r, t = line.strip().split('\t')
                        entities.add(h)
                        entities.add(t)
                        relations.add(r)
        
        entity_dict = {e: i for i, e in enumerate(sorted(entities))}
        relation_dict = {r: i for i, r in enumerate(sorted(relations))}
        
        return entity_dict, relation_dict
```

### 3. 数据集适配器模式

```python
class DatasetAdapter:
    """适配器模式：统一不同数据源的接口"""
    
    def __init__(self, data_source, source_type):
        if source_type == 'file':
            self.reader = DataReader(data_source)
        elif source_type == 'database':
            self.reader = DatabaseReader(data_source)
        elif source_type == 'api':
            self.reader = APIReader(data_source)
    
    def get_train_data(self):
        return self.reader.train_data
    
    def get_entity_dict(self):
        return self.reader.entity_dict
```

### 4. 大数据集优化

#### 内存映射

```python
class MemoryMappedDataset(TrainDataset):
    """使用内存映射处理超大数据集"""
    def __init__(self, ...):
        # 将嵌入矩阵存储在磁盘上，使用mmap访问
        self.entity_embedding = np.memmap(
            'entity_embeddings.npy',
            dtype='float32',
            mode='r+',
            shape=(num_entity, hidden_dim)
        )
```

#### 流式处理

```python
class StreamingTrainDataset:
    """流式数据集，不一次性加载所有数据"""
    def __init__(self, data_file, ...):
        self.data_file = data_file
        self.file_pointer = 0
    
    def __getitem__(self, idx):
        # 从文件流中读取指定位置的数据
        with open(self.data_file, 'r') as f:
            f.seek(self.get_line_position(idx))
            line = f.readline()
            return self.parse_line(line)
```

---

## 总结与最佳实践

### 代码设计优点
1. ✅ **模块化**: 数据、模型、训练流程清晰分离
2. ✅ **可扩展**: 抽象基类设计便于添加新模型
3. ✅ **高效**: 负采样和过滤机制优化了训练效率
4. ✅ **灵活**: 支持多种批处理模式和评估方式

### 改进建议
1. **配置管理**: 使用YAML/JSON配置文件替代命令行参数
2. **日志系统**: 集成TensorBoard或W&B进行可视化
3. **数据验证**: 添加数据格式验证和完整性检查
4. **单元测试**: 为核心模块添加测试用例
5. **文档**: 添加API文档和使用示例

### 集成建议
- **联邦学习**: 重点关注实体对齐和参数聚合策略
- **新数据集**: 实现数据适配器，统一接口
- **分布式训练**: 使用PyTorch DDP或Horovod
- **模型服务**: 添加模型导出和推理服务接口

