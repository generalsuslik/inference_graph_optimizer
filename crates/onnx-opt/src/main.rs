use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use onnx_io::Model;
use onnx_passes::pipeline::Pipeline;

#[derive(Parser)]
#[command(version)]
struct Args {
    /// Model to optimize.
    input: PathBuf,

    /// Where to write the optimized model.
    #[arg(short, long)]
    output: PathBuf,

    /// Upper bound on pipeline iterations; it stops early once nothing changes.
    #[arg(long, default_value_t = 10)]
    max_iterations: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut model = Model::load(&args.input)?;
    let report = Pipeline::standard(args.max_iterations).run(&mut model.graph);
    model.save(&args.output)?;

    println!(
        "{}: {} -> {} nodes in {} iteration(s)",
        args.input.display(),
        report.nodes_before,
        report.nodes_after,
        report.iterations
    );
    for (pass, count) in &report.rewrites {
        println!("  {pass:<20} {count}");
    }
    if let (Some(before), Some(after)) = (report.opset_before, report.opset_after)
        && before != after
    {
        println!("  default opset raised {before} -> {after}: the runtime must support opset {after}");
    }
    println!("wrote {}", args.output.display());
    Ok(())
}
