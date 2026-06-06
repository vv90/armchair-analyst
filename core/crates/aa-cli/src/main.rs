use std::{
    io::{self, Write},
    process::ExitCode,
};

mod app;
mod effects;
mod runtime;
mod utils;

fn main() -> ExitCode {
    warn_if_rustls_provider_already_installed(install_rustls_provider(), &mut io::stderr());

    effects::main_exit_code()
}

fn install_rustls_provider() -> Result<(), ()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| ())
}

fn warn_if_rustls_provider_already_installed<Output: Write>(
    install_result: Result<(), ()>,
    output: &mut Output,
) {
    if install_result.is_err() {
        let _ = writeln!(
            output,
            "warning: rustls crypto provider was already installed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{install_rustls_provider, warn_if_rustls_provider_already_installed};

    #[test]
    fn rustls_provider_installation_selects_process_default() {
        let _ = install_rustls_provider();

        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }

    #[test]
    fn rustls_provider_installation_warns_when_default_already_exists() {
        let mut output = Vec::new();

        warn_if_rustls_provider_already_installed(Err(()), &mut output);

        assert_eq!(
            String::from_utf8(output),
            Ok("warning: rustls crypto provider was already installed\n".to_owned())
        );
    }
}
