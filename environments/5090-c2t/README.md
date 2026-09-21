# 5090 的 c2t 环境：完整包版本、安装细节与建议

本说明来自对实际 5090 服务器的只读采集，时间为 **2026-09-21 15:57:54 UTC / 23:57:54 Asia/Shanghai**。没有修改服务器环境、安装包或启动训练。这里区分“当前已安装状态”和“新机器的建议安装步骤”；后者尚未在全新环境中完整重建验证。

本仓库此前的真实 CityFlow 短训练使用该环境；它不是仅根据 `pyproject.toml` 推测出来的依赖列表。当前 `c2t` 是共享研究环境，包含本项目不需要的 TensorFlow、LLM、Web 等包。

## 1. 实测系统与关键版本

| 项目 | 实测值 |
|---|---|
| 操作系统 / 架构 | Ubuntu 24.04.3 LTS / Linux x86_64 |
| Conda 管理器 | 26.1.1，位于 base，不等同于 c2t 的某个 Python 包 |
| Conda 环境 | `c2t`；本机解释器 `/home/chenyuyang/miniconda3/envs/c2t/bin/python` |
| Python | 3.9.25；Python 自身构建信息显示 GCC 11.2.0 |
| GPU / 显存 | NVIDIA GeForce RTX 5090 / 32607 MiB |
| NVIDIA 驱动 | 595.84 |
| nvidia-smi 的 CUDA Version | 13.2，表示驱动报告的 CUDA 支持级别，不是 PyTorch 构建版本 |
| PyTorch | distribution metadata 为 `2.8.0`；运行时 `torch.__version__` 为 `2.8.0+cu128` |
| PyTorch CUDA / GPU capability | 12.8 / `(12,0)`，即 sm_120；当前 wheel 的 arch list 包含 sm_120 |
| cuDNN | runtime 整数 91002；`nvidia-cudnn-cu12==9.10.2.21` |
| torchvision / torchaudio | 0.23.0 / 2.8.0；当前 baseline 不要求安装这两个包 |
| NumPy / OmegaConf / pytest | 1.26.2 / 2.3.0 / 8.4.2 |
| pip / setuptools / wheel | 26.0.1 / 80.9.0 / 0.45.1 |
| CityFlow | distribution `CityFlow==0.1`，本地 C++ 编译扩展 |
| 当前 PATH 的 GCC / G++ | 13.3.0 |
| 当前 PATH 的 CMake / GNU Make | 3.28.3 / 4.3 |
| glibc | 2.39，系统包版本 `2.39-0ubuntu8.8` |
| nvcc | 本次 SSH 执行环境的 PATH 中未找到；不能据此断定整台机器未安装 toolkit |

`torch.cuda.is_available()` 本次返回 True。上述 GPU 支持是对这组实际版本的观测，不是对任意旧 PyTorch wheel 的保证。CityFlow 在 CPU 上模拟交通；GPU 用于本仓库的神经网络。当前运行预编译 PyTorch wheel 和构建 CityFlow 的路径不需要额外编译 CUDA 扩展。

官方提供 PyTorch 2.8.0 的 CUDA 12.8 wheel 安装入口，建议明确指定 `cu128`，避免无意更换后端。参见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v280)。

## 2. 全量清单与文件用途

| 文件 | 内容 | 使用方式 |
|---|---|---|
| [conda-list.json](conda-list.json) | 201 项包名、版本、build、channel | 完整 Conda 视图；包括 34 项 Conda 管理包和 167 项 PyPI 记录 |
| [pip-list.json](pip-list.json) | 170 个 Python distribution 的版本 | 与 Conda 视图重叠，不能相加；editable 路径已替换为说明性占位符 |
| [pip-versions.txt](pip-versions.txt) | 上述全部 170 项 `name==version` | 版本库存；推荐用作 `-c` 约束，不要直接全量 `-r` 安装 |
| [conda-explicit-linux-64.txt](conda-explicit-linux-64.txt) | 34 个 Conda 包的具体下载 URL/build | 同平台创建 Conda 基础环境；不包含那 167 个 PyPI 包 |
| [environment.snapshot.yml](environment.snapshot.yml) | `conda env export` 全量记录 | 现状归档；去掉机器 prefix、更换环境名，保留现存冲突，因此不是承诺可一键重建的安装文件 |
| [requirements-baseline.txt](requirements-baseline.txt) | 本仓库 baseline 的 NumPy、OmegaConf、pytest 固定版本 | 与 PyTorch、CityFlow 分开安装 |
| [runtime.json](runtime.json) | 系统、GPU、CityFlow hash、源码来源、pip check 等摘要 | 结果溯源；不含 SSH 地址、凭据或环境变量转储 |

Conda explicit 文件适用于对应平台，不是跨 macOS/Windows 的通用锁文件。它只记录 Conda 包，仍须分别处理 pip 包和源码扩展。参见 [Conda 官方 explicit spec 说明](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#explicit-spec-files)。

本次没有直接提交原始 `pip freeze --all`，因为其中包含 CityFlow 的本地 `file://` 路径、旧工程的 `-e` 路径，以及 Conda 构建 pip 时留下的构建机路径。版本全部保留在清单中，相关来源在下文说明。

## 3. 当前环境确实存在的问题

### 3.1 xformers 与 PyTorch 版本不匹配

本次 `python -m pip check` 返回码为 1，输出为：

```text
xformers 0.0.27.post2 has requirement torch==2.4.0, but you have torch 2.8.0.
```

当前仓库的 `src/`、`tests/` 没有导入 xformers，已有 baseline 运行成功也不能证明这个共享环境的所有包都可用。新建 baseline 环境时不安装 xformers；不要为满足这条旧依赖而把已经在 5090 上验证的 PyTorch 降到 2.4.0。完整恢复旧环境会同时恢复这个冲突，不能把它称为依赖无冲突的重建。

### 3.2 旧工程 editable 安装与模块名重叠

环境中存在历史 `cityflow-tsc==0.1.0` editable 安装，还检测到同名同版本的另一份 distribution metadata。它们不是当前发布的 `rl-trafficlight` distribution。新旧工程都使用 Python 模块名 `cityflow_tsc`，因此 distribution 列表正常并不能保证导入了正确源码。

此前 5090 运行通过 `PYTHONPATH=<当前快照>/src` 指向本仓库，没有在共享环境中替换旧安装。继续使用 c2t 时也显式指定该路径并打印 `cityflow_tsc.__file__`；新环境只安装本仓库即可。本说明不要求卸载或改动共享 c2t 的历史包。

### 3.3 CityFlow 的版本号不足以锁定实现

`CityFlow==0.1` 只表示 Python distribution 版本，不能替代源码 revision、子模块版本和已安装扩展 hash。不要直接用 `pip install cityflow==0.1` 代替下面的源码安装过程。

## 4. CityFlow 来源、构建细节

已安装扩展名：`cityflow.cpython-39-x86_64-linux-gnu.so`，大小 573712 bytes，SHA256：

```text
8d7068117efd7efc4b3bd41ed642dfe06675e5620a4f5f8fbf855b7fb43b183a
```

安装 metadata 指向服务器本地 CityFlow 源码目录；本次看到该目录的 origin 为 [cityflow-project/CityFlow](https://github.com/cityflow-project/CityFlow)。当前源码信息：

| 来源 | 固定 revision |
|---|---|
| CityFlow | `81ee0f47659ca66177a71f81676691c58ee89184` |
| extern/pybind11 | `a8ee79d08e9705e2903d5c20327d343ad1b70870`；头文件版本 2.3.0 |
| extern/rapidjson | `7b3d971b261ab624f745e9450b1620157e39cce5` |

主仓库跟踪文件无差异，只有未跟踪的 `CityFlow.egg-info/`。这说明当前源码状态；缺少历史构建日志时，仍不能证明已安装 `.so` 一定由这个 HEAD 编译，所以同时保留源码 revision 和二进制 hash。重新编译后的二进制 hash 可能因工具链、路径等变化而不同。

读取源码可确认：`setup.py` 调用 CMake，传入当前 `sys.executable`，使用 Release 构建及 `-j2`；C++ 标准为 C++11。pybind11 和 rapidjson 是 Git 子模块，单独 `pip install pybind11` 不会替换 CityFlow 实际使用的 vendored 头文件。实测 Python 环境已有 `Python.h`。

`ldd` 显示当前扩展依赖系统的 libstdc++、libgcc、libm、libc，未显示 CUDA 依赖。若出现 `GLIBCXX_* not found`，应先核对实际加载的库路径与编译工具链，不要直接给整个共享环境追加全局 `LD_LIBRARY_PATH`。

官方源码安装入口要求 C++ 构建工具和 CMake，详见 [CityFlow Installation Guide](https://cityflow.readthedocs.io/en/latest/install.html#build-from-source)。下面进一步固定为本次观测到的源码版本；它仍属于建议复建步骤，本轮没有执行源码重编译。

## 5. 建议安装：独立 baseline 环境

以下命令假设在 Linux x86_64、已有可用 NVIDIA 驱动和 Conda 的机器上执行。当前测试环境是 Ubuntu 24.04.3 / 驱动 595.84，不建议照抄某个驱动安装命令覆盖其他机器的现有驱动。

### 5.1 准备仓库、挂载盘和缓存

在 5090 上，环境、源码构建、缓存、日志和训练输出统一放入 `/mnt/pan`；目录名是新环境建议值，不表示已创建。首次 clone 时目标目录应不存在；已有仓库则将 `RL_REPO` 指向实际路径。

```bash
set -e
mountpoint -q /mnt/pan
test -w /mnt/pan

RL_ROOT=/mnt/pan/rl-trafficlight
RL_REPO="$RL_ROOT/trafficlight-general"
RL_ENV="$RL_ROOT/conda-envs/baselines"
RL_CITYFLOW="$RL_ROOT/vendor/CityFlow"

mkdir -p "$RL_ROOT/cache/conda" "$RL_ROOT/cache/pip" "$RL_ROOT/tmp" "$RL_ROOT/vendor"
export CONDA_PKGS_DIRS="$RL_ROOT/cache/conda"
export PIP_CACHE_DIR="$RL_ROOT/cache/pip"
export TMPDIR="$RL_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1

git clone https://github.com/Oltremarer/trafficlight-general.git "$RL_REPO"
```

`mountpoint` 或可写检查失败时停止，不回退到系统盘。非这台服务器可调整根路径，但需要自行确认空间与权限。

### 5.2 创建 Python 基础环境并安装 baseline 依赖

使用完整环境中 34 项 Conda 管理包的 explicit 清单构建基础环境；这样保留 Python 3.9.25、pip/setuptools/wheel 等版本，同时不自动引入历史 Python 扩展包。

```bash
conda create --prefix "$RL_ENV" \
  --file "$RL_REPO/environments/5090-c2t/conda-explicit-linux-64.txt" -y
conda activate "$RL_ENV"

python -m pip install \
  -c "$RL_REPO/environments/5090-c2t/pip-versions.txt" \
  -r "$RL_REPO/environments/5090-c2t/requirements-baseline.txt"

python -m pip install "torch==2.8.0" \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c "$RL_REPO/environments/5090-c2t/pip-versions.txt"
```

`-c` 只约束实际要安装的依赖，不会把 xformers、TensorFlow 或其他全部清单项安装进来。Python package metadata 的 `2.8.0` 与 wheel 运行时的 `2.8.0+cu128` 分别记录；以 import 后的 CUDA 构建、GPU capability 和实际运算检查为准。

当前仓库不需要 torchvision、torchaudio、TensorFlow/Keras、transformers 或 vLLM 才能运行这些 PyTorch baseline。原始上游 TensorFlow 实现应另建环境，不把旧版 TensorFlow/Keras 依赖强行混入这里。

### 5.3 编译 CityFlow

在有管理员权限的新机器上准备系统工具；当前 5090 已具备这些工具，无需重复执行：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
```

建议首先沿用实测的 Python 3.9 与 CMake 3.28.x，避免在复建时同时升级 Python、绑定头文件和构建系统。当前版本的 CityFlow 使用较旧的 CMake 最低版本声明和 pybind11 2.3.0；换用新工具链后的兼容性需要另行验证。

```bash
git clone https://github.com/cityflow-project/CityFlow.git "$RL_CITYFLOW"
git -C "$RL_CITYFLOW" checkout 81ee0f47659ca66177a71f81676691c58ee89184
git -C "$RL_CITYFLOW" submodule update --init --recursive
git -C "$RL_CITYFLOW" submodule status

python -m pip install --no-build-isolation --no-deps "$RL_CITYFLOW"
```

使用当前环境的 `python -m pip` 和已固定的构建工具，避免调用系统 Python 的 pip。核对子模块状态是否与第 4 节一致；上面的安装命令不会自动下载或安装 CityFlow 到 GPU。

### 5.4 安装当前仓库并检查导入

```bash
python -m pip install --no-build-isolation --no-deps -e "$RL_REPO"

python - <<'PY'
import sys
import numpy
import torch
import cityflow
import cityflow_tsc
print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CityFlow:", cityflow.__file__)
print("Project:", cityflow_tsc.__file__)
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
x = torch.ones((16, 16), device="cuda")
assert (x @ x).sum().item() == 4096
print("CUDA tensor check passed")
PY

python -m cityflow_tsc.train_baseline list
python -m pip check
cd "$RL_REPO"
python -m pytest -q
```

这里的 CUDA 运算、单元测试和 baseline 列表分别验证不同层面；它们不等于重新完成真实 CityFlow 训练。真实训练和 checkpoint 评估入口见 [baseline 使用说明](../../docs/baselines.md)。本次环境文档工作没有执行上述新环境安装或额外训练。

## 6. 继续使用现有 c2t 的方式

不要在共享 c2t 中自动升级/卸载包。以前运行使用的是指定解释器和当前源码 `PYTHONPATH`，例如：

```bash
RL_PYTHON=/home/chenyuyang/miniconda3/envs/c2t/bin/python
RL_SOURCE=/mnt/pan/rl-trafficlight/source_snapshots/20260921T150511Z

PYTHONPATH="$RL_SOURCE/src" "$RL_PYTHON" -c \
  'import cityflow_tsc; print(cityflow_tsc.__file__)'
PYTHONPATH="$RL_SOURCE/src" "$RL_PYTHON" -m cityflow_tsc.train_baseline list
```

`RL_SOURCE` 是此前验证使用的快照；更新代码后应指向新快照，并在实验 manifest 中记录源码 hash。不要只根据 pip 里的旧 `cityflow-tsc==0.1.0` 判断代码版本。

每次正式运行先确认 `/mnt/pan` 挂载并可写，创建、报告实际绝对 run 目录，再运行任务。建议环境变量如下，其中 `RL_RUN` 必须是新建的实际运行目录：

```bash
mkdir -p "$RL_RUN/tmp" "$RL_RUN/cache/xdg" "$RL_RUN/cache/torch" \
  "$RL_RUN/cache/cuda" "$RL_RUN/cache/triton" "$RL_RUN/cache/pip"
export TMPDIR="$RL_RUN/tmp"
export XDG_CACHE_HOME="$RL_RUN/cache/xdg"
export TORCH_HOME="$RL_RUN/cache/torch"
export CUDA_CACHE_PATH="$RL_RUN/cache/cuda"
export TRITON_CACHE_DIR="$RL_RUN/cache/triton"
export PIP_CACHE_DIR="$RL_RUN/cache/pip"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

固定版本和这些变量不能独自保证跨不同 GPU/驱动的逐位一致；还须记录训练 seed、评估 seed、数据 hash、源码 hash、CityFlow binary hash 和具体配置。

## 7. 完整包版本表

以下两张表由本次远程输出生成，不能简单相加。Conda 视图包含底层库，pip 视图只覆盖可见的 Python distributions；Python 列表的名字大小写保留原输出。

<details>
<summary>全部 201 项 Conda 视图记录（包名 / 版本 / build / channel）</summary>

| 包名 | 版本 | build | channel |
|---|---|---|---|
| `_libgcc_mutex` | `0.1` | `main` | `pkgs/main` |
| `_openmp_mutex` | `5.1` | `1_gnu` | `pkgs/main` |
| `absl-py` | `2.3.1` | `pypi_0` | `pypi` |
| `accelerate` | `1.10.1` | `pypi_0` | `pypi` |
| `aiohappyeyeballs` | `2.6.1` | `pypi_0` | `pypi` |
| `aiohttp` | `3.13.3` | `pypi_0` | `pypi` |
| `aiosignal` | `1.4.0` | `pypi_0` | `pypi` |
| `annotated-doc` | `0.0.4` | `pypi_0` | `pypi` |
| `annotated-types` | `0.7.0` | `pypi_0` | `pypi` |
| `antlr4-python3-runtime` | `4.9.3` | `pypi_0` | `pypi` |
| `anyio` | `4.12.1` | `pypi_0` | `pypi` |
| `astunparse` | `1.6.3` | `pypi_0` | `pypi` |
| `async-timeout` | `5.0.1` | `pypi_0` | `pypi` |
| `attrs` | `26.1.0` | `pypi_0` | `pypi` |
| `bitsandbytes` | `0.48.2` | `pypi_0` | `pypi` |
| `bzip2` | `1.0.8` | `h5eee18b_6` | `pkgs/main` |
| `ca-certificates` | `2025.12.2` | `h06a4308_0` | `pkgs/main` |
| `certifi` | `2026.2.25` | `pypi_0` | `pypi` |
| `charset-normalizer` | `3.4.6` | `pypi_0` | `pypi` |
| `cityflow` | `0.1` | `pypi_0` | `pypi` |
| `cityflow-tsc` | `0.1.0` | `pypi_0` | `pypi` |
| `click` | `8.1.8` | `pypi_0` | `pypi` |
| `cloudpickle` | `3.1.2` | `pypi_0` | `pypi` |
| `datasets` | `4.5.0` | `pypi_0` | `pypi` |
| `dill` | `0.4.0` | `pypi_0` | `pypi` |
| `diskcache` | `5.6.3` | `pypi_0` | `pypi` |
| `distro` | `1.9.0` | `pypi_0` | `pypi` |
| `einops` | `0.8.2` | `pypi_0` | `pypi` |
| `eval-type-backport` | `0.3.1` | `pypi_0` | `pypi` |
| `exceptiongroup` | `1.3.1` | `pypi_0` | `pypi` |
| `expat` | `2.7.4` | `h7354ed3_0` | `pkgs/main` |
| `fastapi` | `0.128.8` | `pypi_0` | `pypi` |
| `filelock` | `3.19.1` | `pypi_0` | `pypi` |
| `fire` | `0.7.1` | `pypi_0` | `pypi` |
| `flatbuffers` | `25.12.19` | `pypi_0` | `pypi` |
| `frozenlist` | `1.8.0` | `pypi_0` | `pypi` |
| `fsspec` | `2025.10.0` | `pypi_0` | `pypi` |
| `gast` | `0.7.0` | `pypi_0` | `pypi` |
| `gguf` | `0.10.0` | `pypi_0` | `pypi` |
| `gitdb` | `4.0.12` | `pypi_0` | `pypi` |
| `gitpython` | `3.1.46` | `pypi_0` | `pypi` |
| `google-pasta` | `0.2.0` | `pypi_0` | `pypi` |
| `grpcio` | `1.78.0` | `pypi_0` | `pypi` |
| `h11` | `0.16.0` | `pypi_0` | `pypi` |
| `h5py` | `3.14.0` | `pypi_0` | `pypi` |
| `hf-xet` | `1.4.2` | `pypi_0` | `pypi` |
| `httpcore` | `1.0.9` | `pypi_0` | `pypi` |
| `httptools` | `0.7.1` | `pypi_0` | `pypi` |
| `httpx` | `0.28.1` | `pypi_0` | `pypi` |
| `huggingface-hub` | `0.36.2` | `pypi_0` | `pypi` |
| `idna` | `3.11` | `pypi_0` | `pypi` |
| `importlib-metadata` | `8.7.1` | `pypi_0` | `pypi` |
| `iniconfig` | `2.1.0` | `pypi_0` | `pypi` |
| `interegular` | `0.3.3` | `pypi_0` | `pypi` |
| `jinja2` | `3.1.6` | `pypi_0` | `pypi` |
| `jiter` | `0.13.0` | `pypi_0` | `pypi` |
| `joblib` | `1.5.3` | `pypi_0` | `pypi` |
| `jsonschema` | `4.25.1` | `pypi_0` | `pypi` |
| `jsonschema-specifications` | `2025.9.1` | `pypi_0` | `pypi` |
| `keras` | `3.10.0` | `pypi_0` | `pypi` |
| `lark` | `1.3.1` | `pypi_0` | `pypi` |
| `ld_impl_linux-64` | `2.44` | `h9e0c5a2_3` | `pkgs/main` |
| `libclang` | `18.1.1` | `pypi_0` | `pypi` |
| `libexpat` | `2.7.4` | `h7354ed3_0` | `pkgs/main` |
| `libffi` | `3.4.4` | `h6a678d5_1` | `pkgs/main` |
| `libgcc` | `15.2.0` | `h69a1729_7` | `pkgs/main` |
| `libgcc-ng` | `15.2.0` | `h166f726_7` | `pkgs/main` |
| `libgomp` | `15.2.0` | `h4751f2c_7` | `pkgs/main` |
| `libnsl` | `2.0.0` | `h5eee18b_0` | `pkgs/main` |
| `libstdcxx` | `15.2.0` | `h39759b7_7` | `pkgs/main` |
| `libstdcxx-ng` | `15.2.0` | `hc03a8fd_7` | `pkgs/main` |
| `libuuid` | `1.41.5` | `h5eee18b_0` | `pkgs/main` |
| `libxcb` | `1.17.0` | `h9b100fa_0` | `pkgs/main` |
| `libzlib` | `1.3.1` | `hb25bd0a_0` | `pkgs/main` |
| `llvmlite` | `0.43.0` | `pypi_0` | `pypi` |
| `lm-format-enforcer` | `0.10.6` | `pypi_0` | `pypi` |
| `markdown` | `3.9` | `pypi_0` | `pypi` |
| `markdown-it-py` | `3.0.0` | `pypi_0` | `pypi` |
| `markupsafe` | `3.0.2` | `pypi_0` | `pypi` |
| `mdurl` | `0.1.2` | `pypi_0` | `pypi` |
| `mistral-common` | `1.8.5` | `pypi_0` | `pypi` |
| `ml-dtypes` | `0.5.4` | `pypi_0` | `pypi` |
| `mpmath` | `1.3.0` | `pypi_0` | `pypi` |
| `msgpack` | `1.1.2` | `pypi_0` | `pypi` |
| `msgspec` | `0.20.0` | `pypi_0` | `pypi` |
| `multidict` | `6.7.1` | `pypi_0` | `pypi` |
| `multiprocess` | `0.70.18` | `pypi_0` | `pypi` |
| `namex` | `0.1.0` | `pypi_0` | `pypi` |
| `ncurses` | `6.5` | `h7934f7d_0` | `pkgs/main` |
| `nest-asyncio` | `1.6.0` | `pypi_0` | `pypi` |
| `networkx` | `3.2.1` | `pypi_0` | `pypi` |
| `numba` | `0.60.0` | `pypi_0` | `pypi` |
| `numpy` | `1.26.2` | `pypi_0` | `pypi` |
| `nvidia-cublas-cu12` | `12.8.4.1` | `pypi_0` | `pypi` |
| `nvidia-cuda-cupti-cu12` | `12.8.90` | `pypi_0` | `pypi` |
| `nvidia-cuda-nvrtc-cu12` | `12.8.93` | `pypi_0` | `pypi` |
| `nvidia-cuda-runtime-cu12` | `12.8.90` | `pypi_0` | `pypi` |
| `nvidia-cudnn-cu12` | `9.10.2.21` | `pypi_0` | `pypi` |
| `nvidia-cufft-cu12` | `11.3.3.83` | `pypi_0` | `pypi` |
| `nvidia-cufile-cu12` | `1.13.1.3` | `pypi_0` | `pypi` |
| `nvidia-curand-cu12` | `10.3.9.90` | `pypi_0` | `pypi` |
| `nvidia-cusolver-cu12` | `11.7.3.90` | `pypi_0` | `pypi` |
| `nvidia-cusparse-cu12` | `12.5.8.93` | `pypi_0` | `pypi` |
| `nvidia-cusparselt-cu12` | `0.7.1` | `pypi_0` | `pypi` |
| `nvidia-ml-py` | `13.595.45` | `pypi_0` | `pypi` |
| `nvidia-nccl-cu12` | `2.27.3` | `pypi_0` | `pypi` |
| `nvidia-nvjitlink-cu12` | `12.8.93` | `pypi_0` | `pypi` |
| `nvidia-nvtx-cu12` | `12.8.90` | `pypi_0` | `pypi` |
| `omegaconf` | `2.3.0` | `pypi_0` | `pypi` |
| `openai` | `2.29.0` | `pypi_0` | `pypi` |
| `openssl` | `3.5.5` | `h1b28b03_0` | `pkgs/main` |
| `opt-einsum` | `3.4.0` | `pypi_0` | `pypi` |
| `optree` | `0.19.0` | `pypi_0` | `pypi` |
| `outlines` | `0.0.46` | `pypi_0` | `pypi` |
| `packaging` | `26.0` | `pypi_0` | `pypi` |
| `pandas` | `1.5.0` | `pypi_0` | `pypi` |
| `partial-json-parser` | `0.2.1.1.post7` | `pypi_0` | `pypi` |
| `peft` | `0.17.1` | `pypi_0` | `pypi` |
| `pillow` | `11.3.0` | `pypi_0` | `pypi` |
| `pip` | `26.0.1` | `pyhc872135_0` | `pkgs/main` |
| `platformdirs` | `4.4.0` | `pypi_0` | `pypi` |
| `pluggy` | `1.6.0` | `pypi_0` | `pypi` |
| `prometheus-client` | `0.24.1` | `pypi_0` | `pypi` |
| `prometheus-fastapi-instrumentator` | `7.1.0` | `pypi_0` | `pypi` |
| `propcache` | `0.4.1` | `pypi_0` | `pypi` |
| `protobuf` | `6.33.6` | `pypi_0` | `pypi` |
| `psutil` | `7.2.2` | `pypi_0` | `pypi` |
| `pthread-stubs` | `0.3` | `h0ce48e5_1` | `pkgs/main` |
| `py-cpuinfo` | `9.0.0` | `pypi_0` | `pypi` |
| `pyairports` | `0.0.1` | `pypi_0` | `pypi` |
| `pyarrow` | `21.0.0` | `pypi_0` | `pypi` |
| `pycountry` | `24.6.1` | `pypi_0` | `pypi` |
| `pydantic` | `2.12.5` | `pypi_0` | `pypi` |
| `pydantic-core` | `2.41.5` | `pypi_0` | `pypi` |
| `pydantic-extra-types` | `2.11.1` | `pypi_0` | `pypi` |
| `pygments` | `2.19.2` | `pypi_0` | `pypi` |
| `pytest` | `8.4.2` | `pypi_0` | `pypi` |
| `python` | `3.9.25` | `h0dcde21_1` | `pkgs/main` |
| `python-dateutil` | `2.9.0.post0` | `pypi_0` | `pypi` |
| `python-dotenv` | `1.2.1` | `pypi_0` | `pypi` |
| `pytz` | `2026.1.post1` | `pypi_0` | `pypi` |
| `pyyaml` | `6.0.3` | `pypi_0` | `pypi` |
| `pyzmq` | `27.1.0` | `pypi_0` | `pypi` |
| `ray` | `2.51.2` | `pypi_0` | `pypi` |
| `readline` | `8.3` | `hc2a1206_0` | `pkgs/main` |
| `referencing` | `0.36.2` | `pypi_0` | `pypi` |
| `regex` | `2026.1.15` | `pypi_0` | `pypi` |
| `requests` | `2.32.5` | `pypi_0` | `pypi` |
| `rich` | `14.3.3` | `pypi_0` | `pypi` |
| `rpds-py` | `0.27.1` | `pypi_0` | `pypi` |
| `safetensors` | `0.7.0` | `pypi_0` | `pypi` |
| `scikit-learn` | `1.6.1` | `pypi_0` | `pypi` |
| `scipy` | `1.13.1` | `pypi_0` | `pypi` |
| `sentencepiece` | `0.2.1` | `pypi_0` | `pypi` |
| `sentry-sdk` | `2.55.0` | `pypi_0` | `pypi` |
| `setuptools` | `80.9.0` | `py39h06a4308_0` | `pkgs/main` |
| `six` | `1.17.0` | `pypi_0` | `pypi` |
| `smmap` | `5.0.3` | `pypi_0` | `pypi` |
| `sniffio` | `1.3.1` | `pypi_0` | `pypi` |
| `sqlite` | `3.51.2` | `h3e8d24a_0` | `pkgs/main` |
| `starlette` | `0.49.3` | `pypi_0` | `pypi` |
| `sympy` | `1.14.0` | `pypi_0` | `pypi` |
| `tensorboard` | `2.20.0` | `pypi_0` | `pypi` |
| `tensorboard-data-server` | `0.7.2` | `pypi_0` | `pypi` |
| `tensorflow` | `2.20.0` | `pypi_0` | `pypi` |
| `termcolor` | `3.1.0` | `pypi_0` | `pypi` |
| `tf-keras` | `2.20.1` | `pypi_0` | `pypi` |
| `threadpoolctl` | `3.6.0` | `pypi_0` | `pypi` |
| `tiktoken` | `0.12.0` | `pypi_0` | `pypi` |
| `tk` | `8.6.15` | `h54e0aa7_0` | `pkgs/main` |
| `tokenizers` | `0.22.2` | `pypi_0` | `pypi` |
| `tomli` | `2.4.1` | `pypi_0` | `pypi` |
| `torch` | `2.8.0` | `pypi_0` | `pypi` |
| `torchaudio` | `2.8.0` | `pypi_0` | `pypi` |
| `torchvision` | `0.23.0` | `pypi_0` | `pypi` |
| `tqdm` | `4.67.3` | `pypi_0` | `pypi` |
| `transformers` | `4.57.6` | `pypi_0` | `pypi` |
| `triton` | `3.4.0` | `pypi_0` | `pypi` |
| `trl` | `0.24.0` | `pypi_0` | `pypi` |
| `typing-extensions` | `4.15.0` | `pypi_0` | `pypi` |
| `typing-inspection` | `0.4.2` | `pypi_0` | `pypi` |
| `tzdata` | `2026a` | `he532380_0` | `pkgs/main` |
| `urllib3` | `2.6.3` | `pypi_0` | `pypi` |
| `uvicorn` | `0.39.0` | `pypi_0` | `pypi` |
| `uvloop` | `0.22.1` | `pypi_0` | `pypi` |
| `wandb` | `0.25.1` | `pypi_0` | `pypi` |
| `watchfiles` | `1.1.1` | `pypi_0` | `pypi` |
| `websockets` | `15.0.1` | `pypi_0` | `pypi` |
| `werkzeug` | `3.1.7` | `pypi_0` | `pypi` |
| `wheel` | `0.45.1` | `py39h06a4308_0` | `pkgs/main` |
| `wrapt` | `2.1.2` | `pypi_0` | `pypi` |
| `xformers` | `0.0.27.post2` | `pypi_0` | `pypi` |
| `xorg-libx11` | `1.8.12` | `h9b100fa_1` | `pkgs/main` |
| `xorg-libxau` | `1.0.12` | `h9b100fa_0` | `pkgs/main` |
| `xorg-libxdmcp` | `1.1.5` | `h9b100fa_0` | `pkgs/main` |
| `xorg-xorgproto` | `2024.1` | `h5eee18b_1` | `pkgs/main` |
| `xxhash` | `3.6.0` | `pypi_0` | `pypi` |
| `xz` | `5.8.2` | `h448239c_0` | `pkgs/main` |
| `yarl` | `1.22.0` | `pypi_0` | `pypi` |
| `zipp` | `3.23.0` | `pypi_0` | `pypi` |
| `zlib` | `1.3.1` | `hb25bd0a_0` | `pkgs/main` |

</details>

<details>
<summary>全部 170 项 Python distribution 版本</summary>

| 包名 | 版本 | 说明 |
|---|---|---|
| `absl-py` | `2.3.1` |  |
| `accelerate` | `1.10.1` |  |
| `aiohappyeyeballs` | `2.6.1` |  |
| `aiohttp` | `3.13.3` |  |
| `aiosignal` | `1.4.0` |  |
| `annotated-doc` | `0.0.4` |  |
| `annotated-types` | `0.7.0` |  |
| `antlr4-python3-runtime` | `4.9.3` |  |
| `anyio` | `4.12.1` |  |
| `astunparse` | `1.6.3` |  |
| `async-timeout` | `5.0.1` |  |
| `attrs` | `26.1.0` |  |
| `bitsandbytes` | `0.48.2` |  |
| `certifi` | `2026.2.25` |  |
| `charset-normalizer` | `3.4.6` |  |
| `CityFlow` | `0.1` | 本地源码编译；见第 4 节 |
| `cityflow-tsc` | `0.1.0` | 旧工程 editable / metadata；不是当前 rl-trafficlight |
| `click` | `8.1.8` |  |
| `cloudpickle` | `3.1.2` |  |
| `datasets` | `4.5.0` |  |
| `dill` | `0.4.0` |  |
| `diskcache` | `5.6.3` |  |
| `distro` | `1.9.0` |  |
| `einops` | `0.8.2` |  |
| `eval_type_backport` | `0.3.1` |  |
| `exceptiongroup` | `1.3.1` |  |
| `fastapi` | `0.128.8` |  |
| `filelock` | `3.19.1` |  |
| `fire` | `0.7.1` |  |
| `flatbuffers` | `25.12.19` |  |
| `frozenlist` | `1.8.0` |  |
| `fsspec` | `2025.10.0` |  |
| `gast` | `0.7.0` |  |
| `gguf` | `0.10.0` |  |
| `gitdb` | `4.0.12` |  |
| `GitPython` | `3.1.46` |  |
| `google-pasta` | `0.2.0` |  |
| `grpcio` | `1.78.0` |  |
| `h11` | `0.16.0` |  |
| `h5py` | `3.14.0` |  |
| `hf-xet` | `1.4.2` |  |
| `httpcore` | `1.0.9` |  |
| `httptools` | `0.7.1` |  |
| `httpx` | `0.28.1` |  |
| `huggingface_hub` | `0.36.2` |  |
| `idna` | `3.11` |  |
| `importlib_metadata` | `8.7.1` |  |
| `iniconfig` | `2.1.0` |  |
| `interegular` | `0.3.3` |  |
| `Jinja2` | `3.1.6` |  |
| `jiter` | `0.13.0` |  |
| `joblib` | `1.5.3` |  |
| `jsonschema` | `4.25.1` |  |
| `jsonschema-specifications` | `2025.9.1` |  |
| `keras` | `3.10.0` |  |
| `lark` | `1.3.1` |  |
| `libclang` | `18.1.1` |  |
| `llvmlite` | `0.43.0` |  |
| `lm-format-enforcer` | `0.10.6` |  |
| `Markdown` | `3.9` |  |
| `markdown-it-py` | `3.0.0` |  |
| `MarkupSafe` | `3.0.2` |  |
| `mdurl` | `0.1.2` |  |
| `mistral_common` | `1.8.5` |  |
| `ml_dtypes` | `0.5.4` |  |
| `mpmath` | `1.3.0` |  |
| `msgpack` | `1.1.2` |  |
| `msgspec` | `0.20.0` |  |
| `multidict` | `6.7.1` |  |
| `multiprocess` | `0.70.18` |  |
| `namex` | `0.1.0` |  |
| `nest-asyncio` | `1.6.0` |  |
| `networkx` | `3.2.1` |  |
| `numba` | `0.60.0` |  |
| `numpy` | `1.26.2` |  |
| `nvidia-cublas-cu12` | `12.8.4.1` |  |
| `nvidia-cuda-cupti-cu12` | `12.8.90` |  |
| `nvidia-cuda-nvrtc-cu12` | `12.8.93` |  |
| `nvidia-cuda-runtime-cu12` | `12.8.90` |  |
| `nvidia-cudnn-cu12` | `9.10.2.21` |  |
| `nvidia-cufft-cu12` | `11.3.3.83` |  |
| `nvidia-cufile-cu12` | `1.13.1.3` |  |
| `nvidia-curand-cu12` | `10.3.9.90` |  |
| `nvidia-cusolver-cu12` | `11.7.3.90` |  |
| `nvidia-cusparse-cu12` | `12.5.8.93` |  |
| `nvidia-cusparselt-cu12` | `0.7.1` |  |
| `nvidia-ml-py` | `13.595.45` |  |
| `nvidia-nccl-cu12` | `2.27.3` |  |
| `nvidia-nvjitlink-cu12` | `12.8.93` |  |
| `nvidia-nvtx-cu12` | `12.8.90` |  |
| `omegaconf` | `2.3.0` |  |
| `openai` | `2.29.0` |  |
| `opt_einsum` | `3.4.0` |  |
| `optree` | `0.19.0` |  |
| `outlines` | `0.0.46` |  |
| `packaging` | `26.0` |  |
| `pandas` | `1.5.0` |  |
| `partial-json-parser` | `0.2.1.1.post7` |  |
| `peft` | `0.17.1` |  |
| `pillow` | `11.3.0` |  |
| `pip` | `26.0.1` |  |
| `platformdirs` | `4.4.0` |  |
| `pluggy` | `1.6.0` |  |
| `prometheus-fastapi-instrumentator` | `7.1.0` |  |
| `prometheus_client` | `0.24.1` |  |
| `propcache` | `0.4.1` |  |
| `protobuf` | `6.33.6` |  |
| `psutil` | `7.2.2` |  |
| `py-cpuinfo` | `9.0.0` |  |
| `pyairports` | `0.0.1` |  |
| `pyarrow` | `21.0.0` |  |
| `pycountry` | `24.6.1` |  |
| `pydantic` | `2.12.5` |  |
| `pydantic-extra-types` | `2.11.1` |  |
| `pydantic_core` | `2.41.5` |  |
| `Pygments` | `2.19.2` |  |
| `pytest` | `8.4.2` |  |
| `python-dateutil` | `2.9.0.post0` |  |
| `python-dotenv` | `1.2.1` |  |
| `pytz` | `2026.1.post1` |  |
| `PyYAML` | `6.0.3` |  |
| `pyzmq` | `27.1.0` |  |
| `ray` | `2.51.2` |  |
| `referencing` | `0.36.2` |  |
| `regex` | `2026.1.15` |  |
| `requests` | `2.32.5` |  |
| `rich` | `14.3.3` |  |
| `rpds-py` | `0.27.1` |  |
| `safetensors` | `0.7.0` |  |
| `scikit-learn` | `1.6.1` |  |
| `scipy` | `1.13.1` |  |
| `sentencepiece` | `0.2.1` |  |
| `sentry-sdk` | `2.55.0` |  |
| `setuptools` | `80.9.0` |  |
| `six` | `1.17.0` |  |
| `smmap` | `5.0.3` |  |
| `sniffio` | `1.3.1` |  |
| `starlette` | `0.49.3` |  |
| `sympy` | `1.14.0` |  |
| `tensorboard` | `2.20.0` |  |
| `tensorboard-data-server` | `0.7.2` |  |
| `tensorflow` | `2.20.0` |  |
| `termcolor` | `3.1.0` |  |
| `tf_keras` | `2.20.1` |  |
| `threadpoolctl` | `3.6.0` |  |
| `tiktoken` | `0.12.0` |  |
| `tokenizers` | `0.22.2` |  |
| `tomli` | `2.4.1` |  |
| `torch` | `2.8.0` | metadata 2.8.0；runtime 2.8.0+cu128 |
| `torchaudio` | `2.8.0` |  |
| `torchvision` | `0.23.0` |  |
| `tqdm` | `4.67.3` |  |
| `transformers` | `4.57.6` |  |
| `triton` | `3.4.0` |  |
| `trl` | `0.24.0` |  |
| `typing-inspection` | `0.4.2` |  |
| `typing_extensions` | `4.15.0` |  |
| `urllib3` | `2.6.3` |  |
| `uvicorn` | `0.39.0` |  |
| `uvloop` | `0.22.1` |  |
| `wandb` | `0.25.1` |  |
| `watchfiles` | `1.1.1` |  |
| `websockets` | `15.0.1` |  |
| `Werkzeug` | `3.1.7` |  |
| `wheel` | `0.45.1` |  |
| `wrapt` | `2.1.2` |  |
| `xformers` | `0.0.27.post2` | 与当前 torch 存在依赖冲突 |
| `xxhash` | `3.6.0` |  |
| `yarl` | `1.22.0` |  |
| `zipp` | `3.23.0` |  |

</details>
