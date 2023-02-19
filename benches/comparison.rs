use criterion::BenchmarkId;
use criterion::{criterion_group, criterion_main, Criterion};

use giflzw::Decoder;
use giflzw::LzwStatus;
use std::hint::black_box;

use weezl::decode::Decoder as WzlDecoder;
use weezl::encode::Encoder;
use weezl::BitOrder;

mod perf {
    use criterion::profiler::Profiler;

    use pprof::{ProfilerGuard, Report};
    use std::{fs::File, os::raw::c_int, path::Path};
    pub struct FlamegraphProfiler<'a> {
        frequency: c_int,
        active_profiler: Option<ProfilerGuard<'a>>,
    }

    impl<'a> FlamegraphProfiler<'a> {
        #[allow(dead_code)]
        pub fn new(frequency: c_int) -> Self {
            FlamegraphProfiler {
                frequency,
                active_profiler: None,
            }
        }
    }

    impl<'a> Profiler for FlamegraphProfiler<'a> {
        fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
            self.active_profiler = Some(ProfilerGuard::new(self.frequency).unwrap());
        }

        fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
            std::fs::create_dir_all(benchmark_dir).unwrap();
            let flamegraph_path = benchmark_dir.join("flamegraph.svg");
            let flamegraph_file = File::create(&flamegraph_path)
                .expect("File system error while creating flamegraph.svg");
            if let Some(profiler) = self.active_profiler.take() {
                profiler
                    .report()
                    .build()
                    .unwrap()
                    .flamegraph(flamegraph_file)
                    .expect("Error writing flamegraph");
            }
        }
    }
}

fn encode_data(data: Vec<u8>) -> Vec<u8> {
    let mut encoder = Encoder::new(BitOrder::Lsb, 8);
    let out_data = encoder.encode(&data).unwrap();
    out_data
}

fn gif_test_body(encoded: &[u8], holder_slice: &mut [u8]) {
    let mut decoder = Decoder::new(8);
    let mut in_idx = 0;

    loop {
        let result = decoder.decode_bytes(&encoded[in_idx..], holder_slice);
        black_box(&holder_slice);
        in_idx += result.consumed_in;

        if let LzwStatus::Done = result.status.unwrap() {
            break;
        }
    }
}

fn weezl_encode(encoded: &[u8], holder_slice: &mut [u8]) {
    let mut decoder = WzlDecoder::new(BitOrder::Lsb, 8);
    let mut in_idx = 0;

    loop {
        let result = decoder.decode_bytes(&encoded[in_idx..], holder_slice);
        black_box(&holder_slice);
        in_idx += result.consumed_in;

        if let weezl::LzwStatus::Done = result.status.unwrap() {
            break;
        }
    }
}

fn bmp_data_to_vec() -> Vec<u8> {
    use embedded_graphics::pixelcolor::*;
    use tinybmp::Bmp;

    const EYES: &'static [u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test/eyes.bmp"));
    let bmp = Bmp::<Rgb888>::from_slice(EYES).unwrap();
    bmp.pixels()
        .into_iter()
        .map(|pixel| {
            let color = pixel.1;
            [color.r(), color.g(), color.b()]
        })
        .map(|val| val.into_iter())
        .flatten()
        .collect()
}

fn bench(c: &mut Criterion) {
    let data = encode_data(bmp_data_to_vec());
    let data_slice = data.as_slice();
    let mut holder = vec![0; 500];
    c.bench_function("standalone_lzw", |b| {
        b.iter(|| gif_test_body(data_slice, holder.as_mut_slice()))
    });

    //let mut group = c.benchmark_group("Bitmaps");

    //for i in [500] {
    //    group.bench_with_input(
    //        BenchmarkId::new("Weezl", format!("{} byte buffer", i)),
    //        &i,
    //        |b, i| {
    //            let mut holder = vec![0; *i];
    //            b.iter(|| weezl_encode(data_slice, holder.as_mut_slice()))
    //        },
    //    );

    //    group.bench_with_input(
    //        BenchmarkId::new("Giflzw", format!("{} byte buffer", i)),
    //        &i,
    //        |b, i| {
    //            let mut holder = vec![0; *i];
    //            b.iter(|| gif_test_body(data_slice, holder.as_mut_slice()))
    //        },
    //    );
    //}
}

criterion_group!(name = benches; config = Criterion::default().with_profiler(perf::FlamegraphProfiler::new(10000)); targets = bench);

criterion_main!(benches);
