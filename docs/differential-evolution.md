# optibo::de

`optibo` crate 中通过 `de` feature 启用的差分进化优化模块。参考
[SciPy differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
公开的算法与参数语义，独立实现；不保证与 SciPy 随机轨迹或完整 API 等价。

## 快速使用

从外部项目添加本地路径依赖（替换为实际 checkout 位置）：

```toml
[dependencies]
optibo = { path = "/path/to/optibo", default-features = false, features = ["de"] }
```

```rust
use optibo::de::{minimize, Config};

fn main() -> Result<(), optibo::de::DeError> {
    let config = Config {
        population_size: 60,
        max_generations: 500,
        max_evaluations: 30_060,
        seed: 42,
        atol: 1e-12,
        ..Config::default()
    };
    let result = minimize(&[(-5.0, 5.0); 2], &config, |x| {
        (x[0] - 0.3).powi(2) + (x[1] + 0.7).powi(2)
    })?;
    println!("{:?}: {:?}, cost={}", result.termination, result.x, result.fun);
    Ok(())
}
```

默认只启用 `de`，没有第三方依赖。需要并行时，额外启用 `parallel` feature，
并设置 `Config::parallel = true`；该 feature 自动包含 `de`。
未开启 `parallel` 却请求标量并行会返回配置错误。
使用 `default-features = false` 且不选择任何 feature 时，不编译 DE 模块。
自有代码禁止 unsafe；可选 Rayon 及其传递依赖内部有 unsafe 实现。

## 算法与配置

所有数值使用 `f64`，非固定变量在 `[0,1]` 内搜索，调用目标函数时还原为物理量。
边界必须有限，`lower == upper` 表示固定参数。

| 字段 | 默认值与含义 |
| --- | --- |
| `strategy` | `Best1Bin`；也支持 `Rand1Bin` |
| `population_size` | **实际个体数**，默认 40，至少 4 |
| `mutation` | 每代抽取一次 `Dither { min: 0.5, max: 1.0 }`；也支持 `Fixed(F)` |
| `crossover` | 0.7；范围 `[0,1]`，至少一个自由坐标来自变异向量 |
| `init` | `LatinHypercube`；也支持 `Random` 或物理坐标 `Population(rows)` |
| `initial_guess` | 可选，替换初始种群第一行；必须有限且在边界内 |
| `seed` | 0；SplitMix64 随机流，非密码用途 |
| `max_generations` | 1000 个完整进化代，不包含初始化 |
| `max_evaluations` | `usize::MAX`；包含初始化和无效候选的评估 |
| `tol` / `atol` | 0.01 / 0；目标值离散程度的相对与绝对阈值 |

`F` 必须在 `[0,2)`；dither 要求 `0 <= min < max < 2`，每代从 `[min,max)` 采样。
变异使用互不重复且不包含目标个体的 donor。整代试验向量都由上一代种群生成；
越界试验坐标在 `[0,1)` 重新采样，不作截断。

自定义初始种群至少 4 行，每行包括所有固定和自由参数，行数覆盖 `population_size`。
其中有限坐标按边界裁剪，NaN/无穷大或错误形状返回错误。
所有变量固定时只评估唯一点一次；配置和用户输入仍需有效。

## 预算、终止与失败

- 预算必须足以评估完整初始种群；所有参数固定时仅需 1 次。
- 最后一代可以只评估剩余预算允许的前几个试验个体，按种群索引顺序选择。
- `generations` 只记录完整代；`evaluations` 记录候选数量，不是批量调用次数。
- 初始化后及每次选择后检查 `std(costs) <= atol + tol * abs(mean(costs))`。
- 仅当全部种群目标值有限时检查收敛；采用数值缩放避免均值与方差溢出。
- NaN、正负无穷大都表示无效候选，其报告目标值为正无穷。无效 trial 不替换 parent。
- 初始种群没有任何有限目标值时返回 `DeError::NoFiniteObjective`。
- `minimize_fallible` 接收 `Result<f64, EvaluationError>`，模型或数据错误会中止求解。
- 目标函数 panic 不会转换为 `DeError`。

结果保留最优解、目标值、完整物理坐标种群及各成员目标值。
`Termination` 区分 `Converged`、`MaxGenerations`、`MaxEvaluations`、`Cancelled`。
`success()` 仅在收敛时为真；它不是全局最优或参数可辨识性的证明。
目标存在较大常数偏置时，需相应设置 `tol`/`atol`，避免过早达到离散程度阈值。

## 批量接口与取消

```rust
use optibo::de::{BatchObjective, EvaluationError};

struct BatchSphere;
impl BatchObjective for BatchSphere {
    fn evaluate_batch(
        &self,
        candidates: &[f64],
        dimension: usize,
        costs: &mut [f64],
    ) -> Result<(), EvaluationError> {
        for (x, cost) in candidates.chunks_exact(dimension).zip(costs) {
            *cost = x.iter().map(|v| v * v).sum();
        }
        Ok(())
    }
}
```

通过 `minimize_batch(bounds, &config, &BatchSphere)` 调用。
候选使用行优先展平数组，`dimension` 是完整物理参数维数，`costs.len()` 是当前批次个体数。
批量实现负责自身并行或设备调度，忽略 `Config::parallel`。
每次必须填写全部输出；未填写的 NaN 槽位被当作无效候选。

三个入口都有 `_with_callback` 版本，最后一个参数是
`FnMut(&Progress<'_>) -> Control`。返回 `Control::Stop` 取消。
回调在初始化后以及每个完整或部分代后运行，包含 `x`、`fun`、`generations`、`evaluations`。
取消优先于同一边界上的收敛或预算终止；不能中断正在进行的目标评估。

随机试验向量串行生成，目标函数可并行计算，再按固定索引选择。
在确定性目标和相同数值环境下，同一种子不因 Rayon 线程调度改变结果。
跨平台数学库、编译浮点行为，以及目标函数自己的并行归约仍可能带来差异。

## 与 SciPy 的范围差异

- `population_size` 是实际数量，不是乘以自由维数的倍数。
- 只支持 `deferred`，不支持 `immediate` 更新。
- 使用自己的随机流，同 seed 不表示与 NumPy 同序列。
- 保留种群原始索引，不把最优成员交换到第 0 行。
- 初始化后即检查收敛并调用回调；与 SciPy 公共接口的 `maxiter=0` 行为不同。
- 尚无一般非线性约束、整数变量、Sobol/Halton 初始化或自动 `polish`。
- 尚无 Python 绑定；本 crate 可直接作为未来 PyO3 核心依赖。

局部精修和物理惯性约束求解应作为独立模块接入。

## 验证

```bash
cargo test --locked
cargo test --no-default-features --features de --locked
cargo test --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo fmt --all -- --check
cargo run --release --example planar_calibration --features de
```

- 算子单元测试：已知 RNG 向量、无偏索引采样、donor 唯一性、变异公式、强制交叉、LHS 分层、越界修复。
- 数值测试：极端有限边界、次正规数、固定参数、收敛阈值边界和非有限目标值。
- 接口测试：配置验证、初始化、严格预算、部分代、取消、错误传播、批量与标量等价、串并行复现。
- 优化测试：Sphere、Rosenbrock、Rastrigin、不同物理量级、边界最优解。
- 标定测试：无噪声平面双关节机械臂的零偏恢复，以及未参与优化的样本验证。
- SciPy 对照：单代更新、初始化裁剪、固定维度、收敛阈值边界；参考结果来自 SciPy 1.15.3。

普通 Rust 测试不需要 Python。可选参考生成器位于
[`tests/reference/generate_scipy_fixtures.py`](../tests/reference/generate_scipy_fixtures.py)，
用于在安装 SciPy 1.15.3 的环境中复现保存的参考数据。

这些测试验证实现与选定问题上的结果，不代表任意高维或含噪标定问题都能达到同样精度。
