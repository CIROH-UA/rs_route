use std::{env, fs::create_dir_all, path::PathBuf};

fn main() -> anyhow::Result<()> {
    let Ok(project_dir) = env::var("CARGO_MANIFEST_DIR") else {
        panic!("Failed to get CARGO_MANIFEST_DIR env var: undefined");
    };
    let Ok(out_dir) = env::var("OUT_DIR") else {
        panic!("Failed to get OUT_DIR env var: undefined");
    };
    //let Ok(out_dir) = PathBuf::from(out_dir).canonicalize() else {panic!("Failed to canonicalize")};
    let project_dir = PathBuf::from(project_dir);
    let out_dir = PathBuf::from(out_dir);
    let out_dir_str = out_dir.to_str().expect("Failed to get canonical out-dir");
    let project_dir_str = project_dir.to_str().expect("Failed to get canonical project-dir");
    let fortran_build_dir = out_dir.join("muskingum_cunge");
    let fortran_build_dir_str = fortran_build_dir.to_str().expect("Failed to get canonical fortran-build-dir");
    let fortran_src_dir = project_dir.join("src/kernel/muskingum/t-route");
    let fortran_src_dir_str = fortran_src_dir.to_str().expect("Failed to get canonical t-route src dir");
    //let fortran_build_dir = format!("{}/muskingum_cunge", out_dir.to_str().unwrap());
    create_dir_all(&fortran_build_dir)?;

    let mut fortran_src_files = String::new();
    for file in ["varPrecision.f90", "MCsingleSegStime_f2py_NOLOOP.f90"] {
        fortran_src_files.push_str(&format!(" {fortran_src_dir_str}/t-route-legacy/{file}"));
    }
    for file in ["muskingum_cunge.f90"] {
        fortran_src_files.push_str(&format!(" {fortran_src_dir_str}/{file}"));
    }
     for file in ["bind.f90"] {
        fortran_src_files.push_str(&format!(" {fortran_src_dir_str}/{file}"));
    }

    let command_string_compile = format!("gfortran -g -c -O2  -lgfortran -lgcc -static-libgfortran -static-libgcc -nodefaultlibs {fortran_src_files} -J.");

    let _output_compile =  std::process::Command::new("sh").current_dir(&fortran_build_dir)
        .args([
            "-c", 
            &command_string_compile])
        .output().expect("Fortrain compilation failed");

    //println!("{:#?}",output);
    //println!("{command_string_compile}");

    let _output_compile_c =  std::process::Command::new("sh").current_dir(&fortran_build_dir)
        .args([
            "-c", 
            &format!("gcc -g -c -O2 -fPIC -ffast-math {project_dir_str}/src/kernel/muskingum/c_mc/muskingumcunge.c")])
        .output().expect("C compilation failed");
    
    println!("{_output_compile_c:#?}");

    let mut fortran_obj_files = String::new();
    for file in ["MCsingleSegStime_f2py_NOLOOP.o", "varPrecision.o", "muskingum_cunge.o", "bind.o"] {
        fortran_obj_files.push_str(&format!(" {file}"));//&format!(" {fortran_src_dir_str}/t-route-legacy/{file}"));
    }

    let command_string_link = format!("ar rcs {out_dir_str}/libfortran_muskingum.a {fortran_build_dir_str}/muskingumcunge.o {fortran_obj_files}");

    let _output_link =  std::process::Command::new("sh").current_dir(&fortran_build_dir)
        .args([
            "-c", 
            &command_string_link])
        .output().expect("Fortrain linking failed");

    //println!("{_output_compile:#?} \n\n\n {_output_link:#?}");

    
    assert!(_output_compile.status.success());
    assert!(_output_compile_c.status.success());
    assert!(_output_link.status.success());

    //println!("\n{out_dir:?}");

    // std::process::Command::new("sh").current_dir(&fortran_build_dir)
    //     .args(["-c", "gfortran gfortran MCsingleSegStime_f2py_NOLOOP.o pyMCsingleSegStime_NoLoop.o varPrecision.o bind.o"]).output().expect("Fortrain build failed");

    println!("cargo:rustc-link-search=native={out_dir_str}");
    println!("cargo:rustc-link-lib=static=fortran_muskingum");
    //println!("cargo:rustc-link-lib=static=gfortran");
    Ok(())
}