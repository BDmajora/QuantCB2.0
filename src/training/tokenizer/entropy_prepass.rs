use burn::backend::wgpu::kernel::StaticKernelSource;

pub struct EntropyPrePassKernel;

impl StaticKernelSource for EntropyPrePassKernel {
    fn source() -> &'static str {
        include_str!("entropy_prepass.wgsl")
    }

    fn id() -> &'static str {
        "blt_entropy_prepass"
    }
}