# installation
> the version of Behavior is from [IliaLarchenko](https://github.com/IliaLarchenko/behavior-1k-solution)
> I made some modification with the help from AI and now it can easily be installed in conda.

## 快速安装

```bash
# 【optional】指定缓存目录到空间更大的磁盘
export PIP_CACHE_DIR=/path/to/a/larger/disk/.pip-cache
export TMPDIR=/path/to/a/larger/disk/.tmp

bash ./setup.sh --new-env --eval --omnigibson --bddl --joylo --accept-nvidia-eula --accept-conda-tos

# 【optional】重新下载数据集（目前最新版本为 3.7.0rc23）
bash ./setup.sh --dataset
```

## 这个 fork 所做的修改

### 1. 降级 PyTorch 至 2.5.1

将 torch 从 `2.6.0` 降级为 `2.5.1`：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}
```

**原因**：torch 2.6.0 改用 numpy 2.x ABI 编译，而 Isaac Sim 4.5.0 内部依赖 numpy 1.x ABI（如 `numpy==1.26.4`），两者不兼容。2.5.1 是最后一个基于 numpy 1.x ABI 构建的 torch 版本。安装后额外执行 `pip install "numpy>=1.23.5,<2.0"` 防止其他包将 numpy 升级到 2.x。

### 2. 修复 `--eval` 安装时 torch-cluster 失败

OmniGibson 的 `[eval]` 额外依赖会拉取 lerobot，而 lerobot 会将 torch 升级到最新版本，导致后续 `torch-cluster` 与 torch 版本不匹配而安装失败。

**修复**：在安装 `torch-cluster` 之前重新 pin 回 torch 2.5.1，并将安装命令改为：

```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu${CUDA_VER_SHORT}.html
```

同时将 `conda install av` 替换为 `pip install av`，避免 conda 环境污染。

### 3. 限制 joylo 的 numpy 版本

`joylo/setup.py` 中原来的依赖为裸 `numpy`，安装时可能拉取 numpy 2.x。

**修复**：改为 `numpy>=1.23.5,<2.0`，在 joylo 安装后追加 `pip install "numpy>=1.23.5,<2.0"` 进行 re-pin。

### 4. 修复 bddl editable install 的导入优先级

bddl 以 editable 方式安装后，`__editable___bddl_*_finder.py` 使用 `sys.meta_path.append(...)` 追加 `_EditableFinder`，优先级低于默认的 `PathFinder`。当工作目录下存在 `bddl/` 子目录时，Python 会将其识别为 namespace package，导致 `import bddl` 实际加载的是源码目录的空 namespace 而非 editable 安装的包。

**修复**：`setup.sh` 安装完 bddl 后，自动将 finder 文件中的：

```python
sys.meta_path.append(_EditableFinder)
```

替换为：

```python
sys.meta_path.insert(0, _EditableFinder)
```

使其在 `PathFinder` 之前执行。

### 5. 从本地目录安装 Isaac Sim wheel 包

原脚本通过 `curl` 从 `pypi.nvidia.com` 在线下载 Isaac Sim 的 wheel 文件。由于网络限制，改为从本地目录加载：

```
/mnt/public/mjwei/download_models/isaac_packages/
```

该目录需提前存放对应版本的 `.whl` 文件（`manylinux_2_34` 或 `manylinux_2_31` 均可）。脚本会自动检测 GLIBC 版本，优先使用 `manylinux_2_31` 的包。

## 评估

修改 `eval.b1k.sh` 中的 `task.name` 和 `eval_instance_ids` 后执行：

```bash
bash eval.b1k.sh
```
