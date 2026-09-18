<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

<h1>Optibo</h1>

<p><strong>A family of optimization libraries for Rust</strong></p>

<p>
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p>
  <a href="https://github.com/xiaojie-xue/optibo/actions/workflows/rust.yml"><img alt="CI" src="https://github.com/xiaojie-xue/optibo/actions/workflows/rust.yml/badge.svg?branch=main"></a>
  <a href="https://crates.io/crates/optibo"><img alt="crates.io" src="https://img.shields.io/crates/v/optibo.svg?color=CE422B&amp;logo=rust&amp;logoColor=white"></a>
  <a href="https://github.com/xiaojie-xue/optibo/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

</div>

Optibo 是一组持续扩展的 Rust 优化库。目前的 Differential Evolution 实现为机器人标定工具
[Calibo](https://github.com/xiaojie-xue/calibo) 提供数值优化支持，用于根据测量数据辨识模型参数。
后续将扩展其他优化算法，支持更多功能与应用场景，不局限于机器人标定。

## 快速开始

在 `Cargo.toml` 中添加 crates.io 依赖，默认启用差分进化算法：

```toml
[dependencies]
optibo = "0.1.0"
```

```rust
use optibo::de::{minimize, Config};

fn main() -> Result<(), optibo::de::DeError> {
    let config = Config { seed: 42, atol: 1e-12, ..Config::default() };
    let result = minimize(&[(-5.0, 5.0); 2], &config, |x| {
        x.iter().map(|v| v * v).sum()
    })?;
    println!("{:?}: {:?}, cost={}", result.termination, result.x, result.fun);
    Ok(())
}
```

## Features

### Differential Evolution: `de`

目前提供的优化算法 feature 是 `de`，通过 `optibo::de` 使用**差分进化
（Differential Evolution，DE）**。它在给定参数边界内，通过一组候选解的迭代搜索，
寻找使目标函数尽可能小的参数，不需要计算梯度，适用于非线性参数估计。
在机器人标定中，可以将模型预测与实际测量之间的误差作为目标函数，求解待标定参数。

当前实现支持：

- `best1bin` 和 `rand1bin` 策略，以及固定随机种子的可复现求解。
- 参数上下界与固定参数，可配置评估次数和迭代次数上限。
- 标量与批量目标函数，以及进度回调和取消机制。
- 通过可选的 `parallel` feature 并行计算标量目标函数。

`de` 启用差分进化算法，默认开启，默认以串行方式运行，不引入第三方依赖。
上面的快速开始示例使用的就是这一配置。

### 可选加速：`parallel`

当单次目标函数计算较耗时时，可以启用 `parallel`，通过 Rayon 将多个候选解的
目标函数计算分配到不同 CPU 线程。它是差分进化的加速选项，不增加新的优化算法。

使用并行评估需要两步：

1. 在依赖中启用 `parallel`，它会同时启用 `de`：

   ```toml
   [dependencies]
   optibo = { version = "0.1.0", features = ["parallel"] }
   ```

2. 求解时设置 `Config::parallel = true`：

   ```rust
   let config = Config {
       parallel: true,
       ..Config::default()
   };
   ```

仅启用 Cargo feature 时，求解仍默认串行运行。此选项只影响标量目标函数；
批量目标函数由调用方自行安排并行。目标函数计算很简单时，并行的调度开销可能抵消收益。

项目自身的 Rust 代码禁止使用 `unsafe`；Rayon 等可选依赖内部可能使用 unsafe 代码。

## 文档

- [docs.rs API 文档](https://docs.rs/optibo)（首次发布到 crates.io 且文档构建成功后可用）

本地执行 `cargo doc --all-features --no-deps --open` 可构建并打开 API 文档。
docs.rs 已配置为构建全部 feature 的文档。

## 开发与验证

```bash
cargo test --locked
cargo test --no-default-features --locked
cargo test --no-default-features --features de --locked
cargo test --all-features --locked
```

测试覆盖算法算子、数值边界、错误处理、可复现性、串并行一致性、SciPy 确定性对照和
合成标定问题。GitHub Actions 检查各 feature 组合、格式与 Clippy。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
