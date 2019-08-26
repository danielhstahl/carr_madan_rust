#[macro_use]
extern crate criterion;
use rustfft::num_complex::Complex;
use num_complex::Complex as NComplex;
use criterion::{Criterion, ParameterizedBenchmark};


fn bench_option_pricing(c: &mut Criterion) {
    c.bench("fang_oost vs carr_madan",
        ParameterizedBenchmark::new("carr madan", |b, i|{
            b.iter(|| {
                let r=0.05;
                let sig=0.3;
                let t=1.0;
                let s0=50.0;
                let discount=(-r*t as f64).exp();
                let bscf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
                let eta=0.25;
                let alpha=1.5;
                carr_madan::call_price(
                    *i,
                    eta,
                    alpha,
                    s0,
                    discount,
                    &bscf
                );
            });
        }, vec![128, 256, 512, 1024]).with_function("fang oost",|b, i|{
            b.iter(|| {
                let r=0.05;
                let sig=0.3;
                let t=1.0;
                let s0=50.0;
                let bscf=|u:&NComplex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
                let k_array = vec![0.01, 30.0, 50.0, 70.0, 300.0];
                fang_oost_option::option_pricing::fang_oost_call_price(
                    *i,
                    s0,
                    &k_array,
                    r, t,
                    bscf
                );
            });
        })
    );
}

criterion_group!(benches, bench_option_pricing);
criterion_main!(benches);