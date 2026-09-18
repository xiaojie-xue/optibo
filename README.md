# Optibo

一个通过 Cargo feature 选择算法的 Rust 优化库。目前提供差分进化模块 `optibo::de`。
整个项目只有一个 crate：`optibo`。

## 使用

从外部项目使用本地路径依赖（将路径替换为实际 checkout 位置）：

```toml
[dependencies]
optibo = { path = "/path/to/optibo", default-features = false, features = ["de"] }
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

## Features 与 unsafe

| Feature | 含义 |
| --- | --- |
| `de` | 差分进化，默认启用；没有第三方依赖 |
| `parallel` | 显式启用 Rayon 并行能力，同时启用 `de`；运行时设置 `Config::parallel = true` |
| 无 feature | 编译库外壳，不包含 DE 模块或第三方依赖 |

没有 `unsafe` feature，也没有自有 `unsafe` 实现。库根使用 `#![forbid(unsafe_code)]`，
Cargo lint 同时禁止示例与测试引入 unsafe。该限制不约束第三方依赖和 Rust 标准库内部；
启用可选 `parallel` 后，Rayon/Crossbeam 等依赖内部存在 unsafe。

## 文档与验证

- [差分进化算法、接口与 SciPy 差异](docs/differential-evolution.md)
- [合成机械臂零偏标定示例](examples/planar_calibration.rs)

```bash
cargo test --locked
cargo test --no-default-features --locked
cargo test --no-default-features --features de --locked
cargo test --all-features --locked
cargo run --release --example planar_calibration --features de
```

测试覆盖算法算子、数值边界、错误处理、可复现性、串并行一致性、SciPy 确定性对照和
合成标定问题。GitHub Actions 检查各 feature 组合、格式与 Clippy。
