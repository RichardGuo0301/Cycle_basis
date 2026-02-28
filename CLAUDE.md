# IROS Persistent Monitoring — Project Context

## 身份

你是 MIT 5年级 PhD，专攻 robot path planning 及 persistent monitoring 研究。核心 insight：**cycle basis 可以把 infinite loop 转化为 finite loop**，从而在图上找到满足 latency 约束的周期性巡逻路线。最近的突破：代数上的 Cycle Space 修正可能会破坏图的拓扑连通性，而通过图论拓扑上的**Detour 重路由**（Topological Shortest Path）可以完美解决这个断路问题。

---

## 项目结构

```
IROS_main/
├── cycle_basis_demo.py          # 随机多边形图生成 (create_random_polygon_graph)
├── horton_gram_mcb.py           # Horton + GF(2) + Gram 最小 cycle basis 算法
│
├── edge_block.py                # 边阻断：L1 最小 norm 修正 (scipy MILP) + Detour fallback
├── edge_block_MIQP.py           # 边阻断：L2 最小 norm 修正 (cvxpy MIQP) + Detour fallback
├── node_block.py                # 节点阻断：L1 最小 norm 修正该节点所有相连边 + Detour fallback (规划中)
├── edge_block_stats.py          # 大规模边阻断统计实验
│
├── detour_logic.md              # 详细记录了 Detour 重路由机制的底层数学逻辑与代码实现
├── visualizer.py                # 统一的共享可视化模块（生成动态 GIF 和静态比较图）
└── visuals/                     # 动画与静态图输出目录
```

---

## 数学框架

### 图与 Cycle Basis

- 无向图 `G(V, E)`，`n = |V|`，`m = |E|`，cycle rank `k = m - n + 1`
- Horton-Gram MCB 算法求最小 cycle basis → `C_oriented ∈ Z^{m×k}`（有向 cycle-edge 关联矩阵）
- 关键性质：`B_oriented @ C_oriented = 0`（Kirchhoff 定律，保证 degree balance）

### MILP 主问题 (Latency-Feasible Flow 搜索)

给定 `r`（node reward）、`tau`（edge travel time）、`T_max`、`V_MIN_vec`（每节点最少访问次数）：

```
变量: [w+, w-, z, b]  ∈ Z^{2k+m+m}
  w = w+ - w-          # cycle basis 权重（允许正负）
  z = |C_oriented @ w| # 每条边实际遍历次数
  b                    # Big-M 辅助二值变量

目标: max  0.5 * (A_abs^T @ r) @ z   （最大化 reward）

约束:
  z ≥ C@w, z ≥ -C@w            （绝对值展开）
  z ≤ C@w + M(1-b), z ≤ -C@w + Mb （Big-M 线性化）
  tau @ z ≤ T_max                （总时间约束）
  0.5 * A_abs @ z ≥ V_MIN_vec   （每节点最少访问次数）
```

**BFS + Frontier 搜索 V_MIN 空间:**
为了应对 DFS 找不到路线的情况，将 V_MIN 向量视为格点空间，用 BFS+Frontier 搜索。
Priority key = `(total_increment, -J)`，第一键保完备性，第二键保 reward。遇到 DFS 失败则对边界节点（v_opt == V_MIN 的节点）进行 +1 扩展。同时在 DFS 中实施了**两圈验证 (Two-Cycle Verification)** 以确保跨周期的 wrap-around gap 真正满足约束。

---

## 阻断修正与 Detour Fallback (当前核心逻辑)

场景：在找到合法的巡逻流量 `flow_B` 后，某条 edge (或 node) 突然被阻断（blocked）。

### 1. 代数空间修正 (L1 MILP / L2 MIQP)
目标是在 Cycle Space 内找一个修正向量 $C_m$，使得 `(flow_B + C_m)[blocked_idx] = 0`。
- **优点**：满足 Kirchhoff 定律，保证所有节点的 in-degree == out-degree。
- **致命缺点**：代数优化（最小化 $||C_m||_1$ 或 $||C_m||_2^2$）完全**无视图的拓扑结构**。它经常为了强行降低整体 norm，将原本只有 1 个流量的关键桥梁边（或准桥梁）减为 0，导致图在这部分**断裂 (disconnected)**，进而使得修改后的 flow 不再是 Eulerian Circuit (违反弱连通性)。

### 2. 拓扑空间修正 (Detour Re-routing 兜底)
为了解决 L1/L2 造成的图断裂，引入了 **Detour 拓扑降级 (Fallback) 机制**（详见 `detour_logic.md`）：
1. 运行 L1/L2 修正，并检查修正后的流量是否 Eulerian (Degree balanced 且 Weakly connected)。
2. **如果不连通 (Eulerian Check 失败)**，则自动触发 Detour Fallback：
   - 提取被 block 边上原有的流量大小与方向。
   - 在去掉被阻断边的残余图中，利用 Dijkstra 求解 $K$-Shortest Paths 找到两端点间的**时间加权最短路径**。
   - 原边反向抵消 + 绕路正向增加 = 构成了一个拓扑上完美的**增量 Simple Cycle**。
   - 只要残余图依然连通，Detour **100% 保证修正后的图保持连通**，且绝对不破坏 Balance。

**当前代码架构(`edge_block_MIQP.py` 等)**：
```text
MIQP/MILP correction 
      ↓
Eulerian check 
  ├─ [Pass] → Keep algebraic result
  └─ [Fail] → Trigger Detour (K-Shortest Paths via node weights) → Select shortest Eulerian Detour
```

---

## 可视化模块 (visualizer.py)

所有可视化逻辑被重构到了统一文件 `visualizer.py` 中：
- `make_gif`：渲染双圈 Eulerian 回路的动态 GIF。实时显示节点等待时间与 Latency 要求。
- `plot_static_correction`：画出原始 Flow 与修正（Added / Removed）的对比矢量图，支持边阻断与节点阻断模式。
- `plot_corrected_result`：画出修正后最终执行的干净双色 Flow 结构图。

---

## 编码约定

- 图参数：`create_random_polygon_graph(12, 10, seed=100)`
- MILP solver：scipy `milp`（HiGHS backend）
- MIQP solver：cvxpy + SCIP（`pip install cvxpy pyscipopt`）
- 所有 edge_list 按 `sorted([tuple(sorted(e)) for e in G.edges()])` 排序
- flow 方向：`flow_B[i] > 0` 表示方向为 `u→v`，`< 0` 表示 `v→u`
