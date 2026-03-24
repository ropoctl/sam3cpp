use sam3cpp::Sam3Model;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("failed to resolve repo root")?
        .to_path_buf();

    let model = Sam3Model::with_options(repo.join("models/sam3-image-f32.gguf"), false, None::<&std::path::Path>)?;
    let pred = model.predict("/Users/rpo/Downloads/tokens.jpg", "person")?;
    println!(
        "count={} height={} width={} first_score={:.6}",
        pred.count(),
        pred.height,
        pred.width,
        pred.scores.first().copied().unwrap_or(0.0)
    );
    Ok(())
}
