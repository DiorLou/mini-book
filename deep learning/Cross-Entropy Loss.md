# 🧮 交叉熵损失函数 (Cross-Entropy Loss) 的由来

> **核心逻辑：** 最小化负对数似然 (NLL) 等价于最大化观测数据的似然概率。对于分类问题，当标签采用独热编码（One-Hot Encoding）时，负对数似然演变为交叉熵公式。

---

## 1. 从似然概率出发 (Likelihood)

假设对于单个样本，模型输出的预测概率向量为 $\hat{\mathbf{y}}$，真实的标签为 $y$（假设属于第 $k$ 类）。
根据最大似然估计，我们希望模型对真实类别的预测概率 $\hat{y}_k$ 越大越好。

对于整个数据集，假设样本之间相互独立，其总似然概率为所有样本预测概率的乘积：
$$P(\mathbf{Y} | \mathbf{X}) = \prod_{i=1}^{n} P(\mathbf{y}^{(i)} | \mathbf{x}^{(i)})$$

---

## 2. 引入对数与负号 (Negative Log-Likelihood)

为了将**乘法变为加法**（计算更稳定）并将**最大化问题转为最小化问题**（优化器习惯），我们对似然函数取负对数：

$$-\log P(\mathbf{Y} | \mathbf{X}) = \sum_{i=1}^{n} -\log P(\mathbf{y}^{(i)} | \mathbf{x}^{(i)})$$

对于**单个样本**，损失项为：
$$l = -\log P(y | \mathbf{x})$$

---

## 3. 结合独热编码 (One-Hot Encoding) 的推导

在多分类任务中，真实标签 $\mathbf{y}$ 通常表示为一个长度为 $q$ 的独热向量：
* 如果样本属于第 $k$ 类，则 $y_k = 1$，其余元素 $y_j = 0$ ($j \neq k$)。

我们可以利用这个特性，将单个类别的概率提取逻辑写成**求和形式**：
1. **逻辑转换：** $\log P(y | \mathbf{x})$ 实际上就是 $\log \hat{y}_k$。
2. **通项表达：** 观察 $\sum_{j=1}^{q} y_j \log \hat{y}_j$ 这个式子：
    * 当 $j \neq k$ 时，$y_j = 0$，该项消失。
    * 当 $j = k$ 时，$y_j = 1$，结果正好等于 $1 \cdot \log \hat{y}_k$。

因此，红框中的公式诞生了：
### $$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^{q} y_j \log \hat{y}_j$$

---

## 4. 总结：公式背后的物理意义


| 成分 | 物理意义 |
| :--- | :--- |
| **$y_j$** | **真实分布**：指示函数，只在正确答案处为 1。 |
| **$\log \hat{y}_j$** | **信息量**：预测概率越低，其对数的绝对值越大（惩罚越重）。 |
| **$\sum$** | **期望/加权**：在独热编码下，它起到了“过滤”作用，只保留正确类别的损失。 |

---
*推导参考：最大似然估计 (MLE) $\rightarrow$ 负对数似然 (NLL) $\rightarrow$ 交叉熵 (Cross Entropy)*


<br>
<br>
<br>

# 🚀 PyTorch 中的 CrossEntropyLoss 深度拆解

> **核心公式：** `nn.CrossEntropyLoss` = `nn.LogSoftmax` + `nn.NLLLoss`

---

## 1. 为什么 PyTorch 要合并它们？

直接计算 $\log(\text{Softmax}(x))$ 在数学上容易遇到**数值溢出（Numerical Instability）**。
* 如果 $x$ 很大，$\exp(x)$ 会爆炸。
* 如果 $x$ 很小，$\text{Softmax}$ 结果趋近于 0，取 $\log(0)$ 会得到 $-\infty$。

PyTorch 使用了 **Log-Sum-Exp** 技巧来保证计算的稳定性。

---

## 2. 内部逻辑流程图



1. **输入 (Raw Logits):** 模型最后一层的未归一化输出。
2. **LogSoftmax:** 将 Logits 转换为对数概率。
3. **NLLLoss (Negative Log-Likelihood):** 根据真实标签 $y$ 的索引，提取对应的负对数概率。

---

## 3. 代码演示：手动实现 vs 官方函数

我们可以通过以下代码验证 `nn.CrossEntropyLoss` 的内部行为。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 模拟模型输出 (Logits) 和 真实标签
# 假设 2 个样本，3 个类别
logits = torch.tensor([[2.0, 1.0, 0.1], 
                       [1.0, 5.0, 0.2]])
target = torch.tensor([0, 1]) # 样本1属于第0类，样本2属于第1类

# --- 方法 A: 使用官方 nn.CrossEntropyLoss ---
criterion = nn.CrossEntropyLoss()
loss_official = criterion(logits, target)

# --- 方法 B: 手动分步拆解 ---
# 第一步：计算 Softmax 后的对数概率
log_probs = F.log_softmax(logits, dim=1)

# 第二步：根据标签提取对应的负值 (Negative Log-Likelihood)
# 这一步就是你之前运行的 y_hat[[0, 1], y] 的对数版本
loss_manual = F.nll_loss(log_probs, target)

print(f"官方函数 Loss: {loss_official.item():.4f}")
print(f"手动拆解 Loss: {loss_manual.item():.4f}")
print(f"两者是否相等: {torch.allclose(loss_official, loss_manual)}")
```

<br>
<br>
<br>

# 梯度推导：Softmax 交叉熵损失对 Logits 的偏导

> **核心结论：** $\frac{\partial l}{\partial o_j} = \hat{y}_j - y_j$。即：梯度 = 预测值 - 真实值。

---

## 1. 准备工作：定义与公式复盘

首先明确各变量关系：
* **未规范化预测 (Logits):** $\mathbf{o} = [o_1, o_2, \dots, o_q]$
* **Softmax 输出 (预测概率):** $\hat{y}_j = \text{softmax}(o)_j = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)}$
* **损失函数 (Loss):** 根据图 3.4.9，代入 Softmax 定义后得到：
    $$l(\mathbf{y}, \hat{\mathbf{y}}) = \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j$$

---

## 2. 逐步求导过程

我们要计算损失 $l$ 相对于某一个输入 $o_j$ 的偏导数 $\frac{\partial l}{\partial o_j}$。我们将公式拆解为左右两部分分别求导：

### 第一部分：对数项求导 $\frac{\partial}{\partial o_j} \left( \log \sum_{k=1}^q \exp(o_k) \right)$
根据链式法则 $\frac{d \log(u)}{dx} = \frac{1}{u} \frac{du}{dx}$：
1.  外层对 $\log$ 求导：$\frac{1}{\sum_{k=1}^q \exp(o_k)}$
2.  内层对 $\sum \exp(o_k)$ 求导：由于我们要对 $o_j$ 求导，求和项中只有 $\exp(o_j)$ 含有 $o_j$，其他项求导均为 0。因此内层导数为 $\exp(o_j)$。
3.  **合并结果：** $$\frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} = \hat{y}_j$$
    *(这正好就是 Softmax 的输出值！)*

### 第二部分：求和项求导 $\frac{\partial}{\partial o_j} \left( \sum_{j=1}^q y_j o_j \right)$
这是一个关于 $o$ 的线性组合：
* 在求和式 $y_1 o_1 + y_2 o_2 + \dots + y_q o_q$ 中，只有 $y_j o_j$ 这一项包含变量 $o_j$。
* **合并结果：** $$\frac{\partial}{\partial o_j} (y_j o_j) = y_j$$

---

## 3. 最终组合 (图 3.4.10 的由来)

将上述两部分相减，即可得到最终梯度公式：

$$\frac{\partial l}{\partial o_j} = \hat{y}_j - y_j$$

---

## 4. 直观理解这个结果

这个简洁的结果在工程上非常迷人：
* **如果预测很准：** $\hat{y}_j \approx 1$ 且 $y_j = 1$，则梯度趋近于 0，模型不再大幅调整。
* **如果预测很差：** 模型认为概率只有 $\hat{y}_j = 0.1$，但真实标签 $y_j = 1$，则梯度为 $-0.9$。这个较大的负梯度会引导模型在下次迭代时显著**增加** $o_j$ 的值。

---
*推导要点：利用 $\log \sum \exp$ 的特殊性质抵消了复杂的商求导法则。*

# 🔄 深度学习的核心：正向传播与反向传播

> **核心逻辑**：正向传播计算**损失 (Loss)**，反向传播计算**梯度 (Gradient)**。

---

## 1. 正向传播 vs 反向传播

### 🔹 正向传播 (Forward Propagation)
* **过程**：数据从输入层经过隐藏层，最后到达输出层。
* **目的**：根据当前的权重 $\mathbf{w}$ 和偏置 $\mathbf{b}$，计算出预测值 $\hat{y}$。
* **关键产物**：损失函数 $L$。它衡量了预测值与真实值之间的差距。

### 🔸 反向传播 (Backward Propagation)
* **过程**：从输出层开始，利用**链式法则 (Chain Rule)** 将损失 $L$ 对每个参数的偏导数从后往前传递。
* **目的**：计算损失函数相对于每一个模型参数（$\mathbf{w}$ 和 $\mathbf{b}$）的梯度。
* **关键产物**：梯度。它指明了参数应该往哪个方向调整才能使损失减小。



---

## 2. 核心指令：y.backward()

在 PyTorch 中，当你调用 `y.backward()`（通常是 `loss.backward()`）时，系统会自动完成以下操作：

1. **触发计算图**：PyTorch 会沿着正向传播时建立的**动态计算图 (Computation Graph)** 逆向遍历。
2. **应用链式法则**：自动计算 $\frac{\partial \text{loss}}{\partial w}$ 等所有偏导数。
3. **梯度填充**：将计算出的梯度数值存放到每个对应参数（Tensor）的 `.grad` 属性中。

> **注意**：默认情况下，`backward()` 会**累加**梯度。这就是为什么在每个 Batch 训练前需要调用 `optimizer.zero_grad()` 的原因。

---

## 3. 核心属性：x.grad

`x.grad` 是一个存储梯度数值的“容器”。

* **定义**：如果 $y = f(x)$，那么执行 `y.backward()` 后，`x.grad` 中存储的就是 $\frac{dy}{dx}$ 的数值。
* **更新时机**：只有在调用了 `backward()` 之后，这个属性才会有值。
* **使用方式**：优化器（如 SGD）会读取 `x.grad` 中的数值，并按照公式 $w = w - \eta \cdot w.grad$ 来更新参数。

---

## 4. 流程总结

| 阶段 | 关键操作 | 结果 |
| :--- | :--- | :--- |
| **正向** | `y_hat = net(X)` | 得到预测结果 |
| **计算损耗** | `l = loss(y_hat, y)` | 得到标量 Loss |
| **反向** | `l.backward()` | **x.grad** 被填充数值 |
| **更新** | `optimizer.step()` | 参数根据 **x.grad** 进行调整 |

---
*推导提示：在 PyTorch 中，只有 `requires_grad=True` 的张量（通常是权重和偏置）才会计算并存储 grad。*

# 🧬 深度拆解：反向传播与梯度计算的底层逻辑

> **核心定义**：反向传播是**链式法则 (Chain Rule)** 的一种高效算法实现，其目的是计算损失函数 $L$ 对模型中每个可学习参数 $w$ 的偏导数。

---

## 1. 数学视角：链式法则是如何工作的？

假设我们有一个极简的网络：输入 $x \rightarrow$ 线性层 $u = wx \rightarrow$ 激活函数 $y = \sigma(u) \rightarrow$ 损失计算 $L = (y - t)^2$。

在反向传播时，我们从 $L$ 开始，逆向计算偏导数：
1. **最后一层**：计算 $\frac{\partial L}{\partial y}$。
2. **倒数第二层**：根据链式法则，$\frac{\partial L}{\partial u} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial u}$。
3. **参数层**：最终得到 $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial w}$。



---

## 2. PyTorch 底层：y.backward() 到底干了什么？

当你在代码中执行 `loss.backward()` 时，PyTorch 的 **Autograd（自动求导引擎）** 会启动以下流程：

### 🔹 A. 遍历计算图 (DAG)
PyTorch 在正向传播时动态构建了一张**有向无环图 (Directed Acyclic Graph)**。图中每个节点是一个张量，每条边代表一次运算（如 `mul`, `add`, `exp`）。

### 🔹 B. 局部梯度计算
每个节点（算子）都预定义了自己的导数公式。例如，如果是乘法节点 $z = a \cdot b$，它知道 $\frac{\partial z}{\partial a} = b$。

### 🔹 C. 梯度填充与累加 (x.grad)
这是最容易被忽视的一点：
* `backward()` 计算出梯度值后，会寻找那些 `requires_grad=True` 的叶子节点（通常是权重 $w$）。
* 它将计算结果**加到（Accumulate）**该节点的 `.grad` 属性中，而不是覆盖它。
* **这就是为什么每次训练前必须调用 `optimizer.zero_grad()` 的物理原因。**



---

## 3. x.grad 的生命周期

我们可以通过以下实验观察梯度的变化：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # y = 4

# 第一次反向传播
y.backward()
print(f"第一次计算后 x.grad: {x.grad}") # dy/dx = 2x = 4.0

# 第二次反向传播（如果不清零）
y_new = x ** 2
y_new.backward()
print(f"第二次计算后 x.grad: {x.grad}") # 4.0 + 4.0 = 8.0 (梯度累加了！)