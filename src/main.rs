//! Genesis Lang Compiler CLI
//!
//! The `glc` command is the main entry point for the Genesis Lang compiler.

use clap::{Parser, Subcommand};
use genesis::{lexer, parser, typeck};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "glc")]
#[command(author = "Genesis OS Team")]
#[command(version = genesis::VERSION)]
#[command(about = "The Genesis Lang Compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a Genesis source file
    Build {
        /// Input file to compile
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Emit tokens (for debugging)
        #[arg(long)]
        emit_tokens: bool,

        /// Emit AST (for debugging)
        #[arg(long)]
        emit_ast: bool,

        /// Emit type information (for debugging)
        #[arg(long)]
        emit_types: bool,

        /// Emit IR (for debugging)
        #[arg(long)]
        emit_ir: bool,

        /// Emit LLVM IR (for debugging)
        #[arg(long)]
        emit_llvm: bool,

        /// Compile to native executable
        #[arg(long)]
        native: bool,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "2")]
        opt_level: u8,
    },

    /// Check a file for errors without compiling
    Check {
        /// Input file to check
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Tokenize a file and print tokens
    Tokenize {
        /// Input file to tokenize
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Parse a file and print AST
    Parse {
        /// Input file to parse
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Run the REPL
    Repl,
}

fn main() -> miette::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            input,
            output,
            emit_tokens,
            emit_ast,
            emit_types,
            emit_ir,
            emit_llvm,
            native,
            opt_level,
        } => {
            let source = fs::read_to_string(&input)
                .map_err(|e| miette::miette!("Failed to read file: {}", e))?;

            println!("Compiling {}...", input.display());

            if emit_tokens {
                println!("\n=== Tokens ===");
                let (tokens, errors) = lexer::lex(&source);
                for token in &tokens {
                    println!(
                        "{:?} @ {:?} = {:?}",
                        token.kind,
                        token.span,
                        token.text(&source)
                    );
                }
                if !errors.is_empty() {
                    println!("\nLexer errors: {:?}", errors);
                }
            }

            // Parse the source
            let (mut ast, parse_errors) = parser::parse(&source);

            // Resolve external modules (mod foo;)
            let base_path = input.parent().unwrap_or(std::path::Path::new("."));
            let module_errors = parser::resolve_external_modules(&mut ast, base_path);
            let all_parse_errors: Vec<_> = parse_errors.into_iter().chain(module_errors).collect();

            if emit_ast {
                println!("\n=== AST ===");
                println!("{:#?}", ast);
                if !all_parse_errors.is_empty() {
                    println!("\nParser errors: {:?}", all_parse_errors);
                }
            }

            if !all_parse_errors.is_empty() {
                for err in &all_parse_errors {
                    eprintln!("Parser error at {:?}: {}", err.span(), err);
                }
                return Err(miette::miette!("Found {} parse error(s)", all_parse_errors.len()));
            }

            // Type check
            let mut checker = typeck::TypeChecker::new();
            match checker.check_program(&ast) {
                Ok(typed_program) => {
                    if emit_types {
                        println!("\n=== Types ===");
                        for (span, ty) in &typed_program.expr_types {
                            println!("  {:?} : {}", span, ty);
                        }
                        println!("\n=== Symbols ===");
                        for (name, ty) in &typed_program.symbol_types {
                            println!("  {} : {}", name, ty);
                        }
                    }
                    println!("\nType checking successful!");

                    // Generate IR
                    use genesis::ir::{Lowerer, print_module, LLVMCodegen, OptLevel, compile_to_executable};
                    use inkwell::context::Context;

                    let lowerer = Lowerer::new(input.file_stem().unwrap_or_default().to_string_lossy())
                        .with_expr_types(typed_program.expr_types)
                        .with_monomorph(typed_program.monomorph)
                        .with_generic_fn_calls(typed_program.generic_fn_calls);
                    let module = lowerer.lower_program(&ast);

                    if emit_ir {
                        println!("\n=== Genesis IR ===");
                        println!("{}", print_module(&module));
                    }

                    // LLVM code generation
                    if emit_llvm || native {
                        let context = Context::create();
                        let mut codegen = LLVMCodegen::new(&context, &module.name);
                        codegen.compile_module(&module);

                        if let Err(e) = codegen.verify() {
                            eprintln!("LLVM verification error: {}", e);
                        }

                        let opt = match opt_level {
                            0 => OptLevel::None,
                            1 => OptLevel::Less,
                            2 => OptLevel::Default,
                            _ => OptLevel::Aggressive,
                        };
                        codegen.optimize(opt);

                        if emit_llvm {
                            println!("\n=== LLVM IR ===");
                            println!("{}", codegen.get_llvm_ir());
                        }

                        if native {
                            let out_path = output.unwrap_or_else(|| {
                                input.with_extension("")
                            });
                            println!("\nGenerating native executable: {}", out_path.display());

                            match compile_to_executable(&module, &out_path, opt) {
                                Ok(()) => {
                                    println!("Successfully compiled to: {}", out_path.display());
                                }
                                Err(e) => {
                                    return Err(miette::miette!("Code generation failed: {}", e));
                                }
                            }
                        }
                    }
                }
                Err(errors) => {
                    for err in &errors {
                        eprintln!("Type error at {:?}: {}", err.span, err.kind);
                    }
                    return Err(miette::miette!("Found {} type error(s)", errors.len()));
                }
            }

            println!("Compilation successful!");
            Ok(())
        }

        Commands::Check { input } => {
            let source = fs::read_to_string(&input)
                .map_err(|e| miette::miette!("Failed to read file: {}", e))?;

            println!("Checking {}...", input.display());

            let (tokens, lex_errors) = lexer::lex(&source);
            if !lex_errors.is_empty() {
                for err in lex_errors {
                    eprintln!("Lexer error: {:?}", err);
                }
            }

            let (ast, parse_errors) = parser::parse(&source);
            if !parse_errors.is_empty() {
                for err in &parse_errors {
                    eprintln!("Parser error at {:?}: {}", err.span(), err);
                }
                return Err(miette::miette!("Found {} parse error(s)", parse_errors.len()));
            }

            // Type check
            let mut checker = typeck::TypeChecker::new();
            match checker.check_program(&ast) {
                Ok(_) => {
                    println!("No errors found! ({} tokens)", tokens.len());
                }
                Err(errors) => {
                    for err in &errors {
                        eprintln!("Type error at {:?}: {}", err.span, err.kind);
                    }
                    return Err(miette::miette!("Found {} type error(s)", errors.len()));
                }
            }

            Ok(())
        }

        Commands::Tokenize { input } => {
            let source = fs::read_to_string(&input)
                .map_err(|e| miette::miette!("Failed to read file: {}", e))?;

            let (tokens, errors) = lexer::lex(&source);

            for token in &tokens {
                let text = token.text(&source);
                println!(
                    "{:>4}..{:<4} {:20} {:?}",
                    token.span.start,
                    token.span.end,
                    format!("{:?}", token.kind),
                    text
                );
            }

            if !errors.is_empty() {
                eprintln!("\nLexer errors:");
                for err in errors {
                    eprintln!("  {:?}", err);
                }
            }

            Ok(())
        }

        Commands::Parse { input } => {
            let source = fs::read_to_string(&input)
                .map_err(|e| miette::miette!("Failed to read file: {}", e))?;

            let (ast, errors) = parser::parse(&source);

            println!("{:#?}", ast);

            if !errors.is_empty() {
                eprintln!("\nParser errors:");
                for err in errors {
                    eprintln!("  {:?}", err);
                }
            }

            Ok(())
        }

        Commands::Repl => {
            println!("Genesis Lang REPL v{}", genesis::VERSION);
            println!("Type 'exit' to quit.\n");

            let stdin = std::io::stdin();
            let mut input = String::new();

            loop {
                print!("> ");
                use std::io::Write;
                let _ = std::io::stdout().flush(); // Ignore flush errors

                input.clear();
                if stdin.read_line(&mut input).is_err() {
                    break;
                }

                let input = input.trim();
                if input == "exit" || input == "quit" {
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                // Try to parse as expression first
                let wrapped = format!("fn __repl__() {{ {} }}", input);
                let (ast, errors) = parser::parse(&wrapped);

                if errors.is_empty() {
                    println!("{:#?}", ast);
                } else {
                    // Try parsing as item
                    let (ast, errors) = parser::parse(input);
                    if errors.is_empty() {
                        println!("{:#?}", ast);
                    } else {
                        for err in errors {
                            eprintln!("Error: {}", err);
                        }
                    }
                }
            }

            println!("Goodbye!");
            Ok(())
        }
    }
}
