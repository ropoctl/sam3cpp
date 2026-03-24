use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn emit_dir_rerun_if_changed(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                emit_dir_rerun_if_changed(&path);
            } else {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

fn run(cmd: &mut Command) {
    let status = cmd.status().expect("failed to start command");
    if !status.success() {
        panic!("command failed with status {status}");
    }
}

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir
        .parent()
        .and_then(Path::parent)
        .expect("failed to resolve repo root")
        .to_path_buf();
    let build_dir = repo_root.join("build");

    println!("cargo:rerun-if-changed={}", repo_root.join("CMakeLists.txt").display());
    emit_dir_rerun_if_changed(&repo_root.join("include"));
    emit_dir_rerun_if_changed(&repo_root.join("src"));

    run(Command::new("cmake")
        .arg("-S")
        .arg(&repo_root)
        .arg("-B")
        .arg(&build_dir));
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("-j8")
        .arg("--target")
        .arg("sam3cpp_c_static"));

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("third_party/ggml/src").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("third_party/ggml/src/ggml-metal").display()
    );

    println!("cargo:rustc-link-lib=static=sam3cpp_c");
    println!("cargo:rustc-link-lib=static=sam3cpp_static");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-metal");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=z");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
}
