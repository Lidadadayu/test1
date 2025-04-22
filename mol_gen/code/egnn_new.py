from torch import nn
import torch
import math

class GCL(nn.Module):
    def __init__(self, input_node_nf, output_node_nf, hidden_nf,
                  normalization_factor, aggregation_method,
                 edges_extra_d=0, nodes_extra_d=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge_nf = input_node_nf * 2  #输入边特征的维度为节点特征的两倍
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(    #利用sequential构建一个边多层感知机
            nn.Linear(input_edge_nf + edges_extra_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(     #利用sequential构建一个节点多层感知机
            nn.Linear(hidden_nf + input_node_nf + nodes_extra_d, hidden_nf), #这里把隐藏层的维度也加上了（不知为何）
            act_fn,
            nn.Linear(hidden_nf, output_node_nf))
             
         #这里由于有个条件语句，所以没加act_fn
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source_node, target_node, edge_attr, edge_mask): #边的特征与节点特征关联，所以这里直接拼接
        if edge_attr is None:  # Unused.
            out = torch.cat([source_node, target_node], dim=1) #dim=1表示在第二维拼接
        else:
            out = torch.cat([source_node, target_node, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr): #节点属性更新需要考虑所连接的各个边的属性
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),#这里edge_attr是边自带属性特征
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method) #对边特征进行分组求和再归一化
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out,agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        #参数取值为None时，表示在传参时可以不传入该参数，自由选择
        row, col = edge_index

         #这里的h[row]和h[col]分别表示源节点数组和目标节点数组的特征
        edge_attr, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_attr, node_attr) 
        #h是经过残差+（聚合边信息经过node_mlp）之后的最终节点特征
        h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module): #节点坐标更新模块
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_attr=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge_nf = hidden_nf * 2 + edges_attr  # 输入边特征的维度为节点特征的两倍加上边特征
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf), #coord_mlp的输出维度为hidden_nf
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):#coord_diff是节点坐标差值
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)  # 拼接源节点、目标节点和边特征
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range 
            #coords_range是一个超参数，控制坐标更新的范围，
            # 使用tanh控制坐标更新范围，所以这里乘以一个范围值
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method) #还是对边特征进行分组求和再归一化
        coord = coord + agg
        return coord  #这里的coord是更新后的节点坐标

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord 
    #乘以mask是为了屏蔽掉一些节点的影响，因为有些节点可能是padding的节点，或者是没有用的节点，它们的mask值为0


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_attr=2, device='cpu', #这里device用cpu还是gpu好?
                 act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, 
                 norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        
        #先添加n层GCL层,再添加EquivariantUpdate层
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, 
                            GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_attr=edge_attr,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", 
                        EquivariantUpdate(hidden_nf, edge_attr=edge_attr, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        #distances是节点之间的距离平方和，coord_diff是节点坐标差值(distances的平方根再归一化之后的值)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1) #这里边特征等于拼接了距离平方和和原有的边特征
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
         #这里两个self._modules调用，都会调用forward函数，进行节点特征和坐标的更新

        if node_mask is not None:
            h = h * node_mask
        return h, x
    #返回更新后的节点特征h和更新后的节点坐标x


class EGNN(nn.Module): #会使用EquivariantBlock实现对节点的更新
    def __init__(self, in_node_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), 
                 n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, 
                 inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        #self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_attr = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_attr = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers): 
            self.add_module("equiv_block_%d" % i, EquivariantBlock
                            (hidden_nf, edge_attr=edge_attr, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["equiv_block_%d" % i](h, x, edge_index, node_mask=node_mask, 
                                                   edge_mask=edge_mask, edge_attr=distances)
            #调用n个EquivariantBlock实现对节点特征和坐标的更新,这里还实现sinusoidal embedding
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


 
"""
    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, 
                                               node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h
"""

class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        #下面的操作会生成0~n_frequencies-1的整数序列，再乘以2*π*div_factor^i/max_res,生成一个不同幂次的序列
        # **是幂运算符，表示div_factor的i次方
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):  #x为距离平方
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


#下面两个是全局函数
def coord2diff(x, edge_index, norm_constant=1): #x为节点坐标张量
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1) 
    # 计算每一行所有元素平方和，即距离平方，并在指定在第二维扩展维度
    norm = torch.sqrt(radial + 1e-8) #加上一个小常数，避免除0错误
    coord_diff = coord_diff/(norm + norm_constant) # 归一化
    return radial, coord_diff  #返回距离平方和归一化后的坐标差值


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
   
   #data按照索引分组，再同组求和，后面进行归一化
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # 初始化一个全0的张量
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1)) #expand进行维度扩张，-1表示不变，
    result.scatter_add_(0, segment_ids, data) #对data按照索引分组并同组求和
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape) #只能初始化为全0
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape)) #创建和data一样大小的全1张量
        norm[norm == 0] = 1 #这样不会有norm=0的情况，避免除0错误
        result = result / norm 
    return result