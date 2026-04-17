use burn::backend::wgpu::kernel::StaticKernelSource;

pub struct TernaryMatMulKernel;

impl StaticKernelSource for TernaryMatMulKernel {
    fn source() -> &'static str {
        include_str!("ternary_matmul.wgsl")
    }

    fn id() -> &'static str {
        "bitnet_ternary_matmul"
    }
}