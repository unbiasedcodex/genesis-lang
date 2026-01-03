//! IR Builder
//!
//! Helper for constructing IR instructions and basic blocks.

use super::types::{BasicBlock, BlockId, Constant, Function, Global, IrType, Module, VReg};
use super::instr::{CmpOp, Instruction, InstrKind, Terminator};

/// Builder for constructing IR
pub struct IrBuilder {
    /// Next virtual register ID
    next_vreg: u32,
    /// Next block ID
    next_block: u32,
    /// Next string constant ID
    next_string: u32,
    /// Current module being built
    module: Module,
    /// Current function being built
    current_fn: Option<Function>,
    /// Current block being built
    current_block: Option<BasicBlock>,
    /// Track if stdio functions are declared
    stdio_declared: bool,
    /// Track if math functions are declared
    math_declared: bool,
}

impl IrBuilder {
    /// Create a new IR builder
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            next_vreg: 0,
            next_block: 0,
            next_string: 0,
            module: Module::new(module_name),
            current_fn: None,
            current_block: None,
            stdio_declared: false,
            math_declared: false,
        }
    }

    /// Finish building and return the module
    pub fn finish(mut self) -> Module {
        // Finalize any remaining function
        self.finish_function();
        self.module
    }

    /// Create a fresh virtual register
    pub fn fresh_vreg(&mut self) -> VReg {
        let vreg = VReg(self.next_vreg);
        self.next_vreg += 1;
        vreg
    }

    /// Create a fresh block ID
    pub fn fresh_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    // ============ Function Building ============

    /// Start building a new function
    pub fn start_function(&mut self, name: impl Into<String>, params: Vec<IrType>, ret_type: IrType) -> Vec<VReg> {
        self.finish_function();

        // Create parameter vregs
        let param_vregs: Vec<(VReg, IrType)> = params
            .into_iter()
            .map(|ty| (self.fresh_vreg(), ty))
            .collect();
        let vregs: Vec<VReg> = param_vregs.iter().map(|(v, _)| *v).collect();

        self.current_fn = Some(Function::new(name, param_vregs, ret_type));

        // Create entry block
        let entry = self.fresh_block();
        self.current_block = Some(BasicBlock::new(entry));

        vregs
    }

    /// Finish the current function
    pub fn finish_function(&mut self) {
        if let Some(block) = self.current_block.take() {
            if let Some(ref mut func) = self.current_fn {
                func.blocks.push(block);
            }
        }
        if let Some(func) = self.current_fn.take() {
            self.module.functions.push(func);
        }
    }

    /// Declare an external function
    pub fn declare_external(&mut self, name: impl Into<String>, params: Vec<IrType>, ret_type: IrType) {
        let mut func = Function::new(name, vec![], ret_type);
        func.is_external = true;
        // Store param types without vregs for external functions
        func.params = params.into_iter().map(|ty| (VReg(0), ty)).collect();
        self.module.functions.push(func);
    }

    /// Declare an external variadic function
    pub fn declare_external_vararg(&mut self, name: impl Into<String>, params: Vec<IrType>, ret_type: IrType) {
        let mut func = Function::new(name, vec![], ret_type);
        func.is_external = true;
        func.is_vararg = true;
        func.params = params.into_iter().map(|ty| (VReg(0), ty)).collect();
        self.module.functions.push(func);
    }

    /// Declare stdio functions (puts, printf, fputs, fprintf, etc.)
    pub fn declare_stdio(&mut self) {
        if self.stdio_declared {
            return;
        }
        self.stdio_declared = true;

        // int puts(const char *s)
        self.declare_external("puts", vec![IrType::Ptr(Box::new(IrType::I8))], IrType::I32);

        // int printf(const char *format, ...) - variadic function
        self.declare_external_vararg("printf", vec![IrType::Ptr(Box::new(IrType::I8))], IrType::I32);

        // int fputs(const char *s, FILE *stream)
        self.declare_external("fputs", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // string
            IrType::Ptr(Box::new(IrType::I8)),  // FILE* (opaque)
        ], IrType::I32);

        // int fprintf(FILE *stream, const char *format, ...) - variadic function
        self.declare_external_vararg("fprintf", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // FILE* (opaque)
            IrType::Ptr(Box::new(IrType::I8)),  // format string
        ], IrType::I32);

        // char *fgets(char *s, int size, FILE *stream)
        self.declare_external("fgets", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // buffer
            IrType::I32,                         // size
            IrType::Ptr(Box::new(IrType::I8)),  // FILE* (opaque)
        ], IrType::Ptr(Box::new(IrType::I8)));

        // size_t strlen(const char *s)
        self.declare_external("strlen", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // string
        ], IrType::I64);

        // int sprintf(char *str, const char *format, ...) - variadic function
        self.declare_external_vararg("sprintf", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // buffer
            IrType::Ptr(Box::new(IrType::I8)),  // format string
        ], IrType::I32);

        // long strtol(const char *nptr, char **endptr, int base)
        self.declare_external("strtol", vec![
            IrType::Ptr(Box::new(IrType::I8)),                      // nptr (string)
            IrType::Ptr(Box::new(IrType::Ptr(Box::new(IrType::I8)))), // endptr (char**)
            IrType::I32,                                             // base
        ], IrType::I64);

        // double strtod(const char *nptr, char **endptr)
        self.declare_external("strtod", vec![
            IrType::Ptr(Box::new(IrType::I8)),                      // nptr (string)
            IrType::Ptr(Box::new(IrType::Ptr(Box::new(IrType::I8)))), // endptr (char**)
        ], IrType::F64);

        // float strtof(const char *nptr, char **endptr)
        self.declare_external("strtof", vec![
            IrType::Ptr(Box::new(IrType::I8)),                      // nptr (string)
            IrType::Ptr(Box::new(IrType::Ptr(Box::new(IrType::I8)))), // endptr (char**)
        ], IrType::F32);

        // ============ File I/O ============

        // FILE *fopen(const char *path, const char *mode)
        self.declare_external("fopen", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // path
            IrType::Ptr(Box::new(IrType::I8)),  // mode
        ], IrType::Ptr(Box::new(IrType::I8)));  // FILE* (opaque)

        // int fclose(FILE *stream)
        self.declare_external("fclose", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // FILE*
        ], IrType::I32);

        // size_t fread(void *ptr, size_t size, size_t count, FILE *stream)
        self.declare_external("fread", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // buffer
            IrType::I64,                         // size
            IrType::I64,                         // count
            IrType::Ptr(Box::new(IrType::I8)),  // FILE*
        ], IrType::I64);

        // size_t fwrite(const void *ptr, size_t size, size_t count, FILE *stream)
        self.declare_external("fwrite", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // buffer
            IrType::I64,                         // size
            IrType::I64,                         // count
            IrType::Ptr(Box::new(IrType::I8)),  // FILE*
        ], IrType::I64);

        // int fseek(FILE *stream, long offset, int whence)
        self.declare_external("fseek", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // FILE*
            IrType::I64,                         // offset
            IrType::I32,                         // whence (SEEK_SET=0, SEEK_CUR=1, SEEK_END=2)
        ], IrType::I32);

        // long ftell(FILE *stream)
        self.declare_external("ftell", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // FILE*
        ], IrType::I64);

        // ============ Extended File System Operations ============

        // int access(const char *pathname, int mode)
        // mode: F_OK=0 (exists), R_OK=4, W_OK=2, X_OK=1
        self.declare_external("access", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // pathname
            IrType::I32,                         // mode
        ], IrType::I32);

        // int stat(const char *pathname, struct stat *statbuf)
        // Returns 0 on success, -1 on error
        self.declare_external("stat", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // pathname
            IrType::Ptr(Box::new(IrType::I8)),  // stat buffer (opaque, ~144 bytes on Linux x86_64)
        ], IrType::I32);

        // int remove(const char *pathname)
        // Removes a file or empty directory
        self.declare_external("remove", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // pathname
        ], IrType::I32);

        // int mkdir(const char *pathname, mode_t mode)
        self.declare_external("mkdir", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // pathname
            IrType::I32,                         // mode (e.g., 0755)
        ], IrType::I32);

        // int rmdir(const char *pathname)
        self.declare_external("rmdir", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // pathname
        ], IrType::I32);

        // DIR *opendir(const char *name)
        self.declare_external("opendir", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // pathname
        ], IrType::Ptr(Box::new(IrType::I8)));  // DIR* (opaque)

        // struct dirent *readdir(DIR *dirp)
        self.declare_external("readdir", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // DIR*
        ], IrType::Ptr(Box::new(IrType::I8)));  // struct dirent* (opaque)

        // int closedir(DIR *dirp)
        self.declare_external("closedir", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // DIR*
        ], IrType::I32);

        // char *strrchr(const char *s, int c)
        // Find last occurrence of character
        self.declare_external("strrchr", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // string
            IrType::I32,                         // character
        ], IrType::Ptr(Box::new(IrType::I8)));

        // void *memcpy(void *dest, const void *src, size_t n)
        self.declare_external("memcpy", vec![
            IrType::Ptr(Box::new(IrType::I8)),  // dest
            IrType::Ptr(Box::new(IrType::I8)),  // src
            IrType::I64,                         // n
        ], IrType::Ptr(Box::new(IrType::I8)));

        // ============ Epoll I/O (for async reactor) ============

        // int epoll_create1(int flags)
        self.declare_external("epoll_create1", vec![IrType::I32], IrType::I32);

        // int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event)
        // epoll_event struct is { u32 events, u64 data } = 12 bytes, passed as ptr
        self.declare_external("epoll_ctl", vec![
            IrType::I32,                         // epfd
            IrType::I32,                         // op (EPOLL_CTL_ADD=1, MOD=3, DEL=2)
            IrType::I32,                         // fd
            IrType::Ptr(Box::new(IrType::I8)),   // epoll_event* (opaque)
        ], IrType::I32);

        // int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout)
        self.declare_external("epoll_wait", vec![
            IrType::I32,                         // epfd
            IrType::Ptr(Box::new(IrType::I8)),   // events array
            IrType::I32,                         // maxevents
            IrType::I32,                         // timeout (-1 = infinite)
        ], IrType::I32);

        // ============ Low-level I/O ============

        // ssize_t read(int fd, void *buf, size_t count)
        self.declare_external("read", vec![
            IrType::I32,                         // fd
            IrType::Ptr(Box::new(IrType::I8)),   // buf
            IrType::I64,                         // count
        ], IrType::I64);

        // ssize_t write(int fd, const void *buf, size_t count)
        self.declare_external("write", vec![
            IrType::I32,                         // fd
            IrType::Ptr(Box::new(IrType::I8)),   // buf
            IrType::I64,                         // count
        ], IrType::I64);

        // int close(int fd)
        self.declare_external("close", vec![IrType::I32], IrType::I32);

        // int fcntl(int fd, int cmd, ... /* arg */)
        self.declare_external_vararg("fcntl", vec![
            IrType::I32,                         // fd
            IrType::I32,                         // cmd
        ], IrType::I32);

        // ============ Socket I/O ============

        // int socket(int domain, int type, int protocol)
        self.declare_external("socket", vec![
            IrType::I32,                         // domain (AF_INET=2)
            IrType::I32,                         // type (SOCK_STREAM=1)
            IrType::I32,                         // protocol
        ], IrType::I32);

        // int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
        self.declare_external("bind", vec![
            IrType::I32,                         // sockfd
            IrType::Ptr(Box::new(IrType::I8)),   // addr
            IrType::I32,                         // addrlen
        ], IrType::I32);

        // int listen(int sockfd, int backlog)
        self.declare_external("listen", vec![
            IrType::I32,                         // sockfd
            IrType::I32,                         // backlog
        ], IrType::I32);

        // int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
        self.declare_external("accept", vec![
            IrType::I32,                         // sockfd
            IrType::Ptr(Box::new(IrType::I8)),   // addr (can be null)
            IrType::Ptr(Box::new(IrType::I32)),  // addrlen (can be null)
        ], IrType::I32);

        // int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
        self.declare_external("connect", vec![
            IrType::I32,                         // sockfd
            IrType::Ptr(Box::new(IrType::I8)),   // addr
            IrType::I32,                         // addrlen
        ], IrType::I32);
    }

    /// Declare math functions from libm for use in Genesis programs.
    ///
    /// This function declares external C math library functions that are linked
    /// with `-lm` during the final linking phase. Functions are organized into:
    ///
    /// - **Trigonometric**: sin, cos, tan, asin, acos, atan, atan2
    /// - **Hyperbolic**: sinh, cosh, tanh
    /// - **Exponential/Log**: exp, exp2, log, log2, log10
    /// - **Power**: pow, sqrt, cbrt, hypot
    /// - **Rounding**: floor, ceil, round, trunc
    /// - **Absolute value**: fabs, fabsf, labs, abs
    /// - **Min/Max**: fmin, fmax, fminf, fmaxf
    ///
    /// Called lazily on first use of any math function to avoid unnecessary
    /// declarations in programs that don't use math operations.
    ///
    /// See `docs/math.md` for complete API documentation.
    pub fn declare_math(&mut self) {
        if self.math_declared {
            return;
        }
        self.math_declared = true;

        // ============ Trigonometric Functions ============

        // double sin(double x)
        self.declare_external("sin", vec![IrType::F64], IrType::F64);

        // double cos(double x)
        self.declare_external("cos", vec![IrType::F64], IrType::F64);

        // double tan(double x)
        self.declare_external("tan", vec![IrType::F64], IrType::F64);

        // double asin(double x)
        self.declare_external("asin", vec![IrType::F64], IrType::F64);

        // double acos(double x)
        self.declare_external("acos", vec![IrType::F64], IrType::F64);

        // double atan(double x)
        self.declare_external("atan", vec![IrType::F64], IrType::F64);

        // double atan2(double y, double x)
        self.declare_external("atan2", vec![IrType::F64, IrType::F64], IrType::F64);

        // ============ Hyperbolic Functions ============

        // double sinh(double x)
        self.declare_external("sinh", vec![IrType::F64], IrType::F64);

        // double cosh(double x)
        self.declare_external("cosh", vec![IrType::F64], IrType::F64);

        // double tanh(double x)
        self.declare_external("tanh", vec![IrType::F64], IrType::F64);

        // ============ Exponential and Logarithmic Functions ============

        // double exp(double x)
        self.declare_external("exp", vec![IrType::F64], IrType::F64);

        // double exp2(double x)
        self.declare_external("exp2", vec![IrType::F64], IrType::F64);

        // double log(double x) - natural logarithm
        self.declare_external("log", vec![IrType::F64], IrType::F64);

        // double log2(double x)
        self.declare_external("log2", vec![IrType::F64], IrType::F64);

        // double log10(double x)
        self.declare_external("log10", vec![IrType::F64], IrType::F64);

        // ============ Power Functions ============

        // double pow(double base, double exp)
        self.declare_external("pow", vec![IrType::F64, IrType::F64], IrType::F64);

        // double sqrt(double x)
        self.declare_external("sqrt", vec![IrType::F64], IrType::F64);

        // double cbrt(double x) - cube root
        self.declare_external("cbrt", vec![IrType::F64], IrType::F64);

        // double hypot(double x, double y)
        self.declare_external("hypot", vec![IrType::F64, IrType::F64], IrType::F64);

        // ============ Rounding Functions ============

        // double floor(double x)
        self.declare_external("floor", vec![IrType::F64], IrType::F64);

        // double ceil(double x)
        self.declare_external("ceil", vec![IrType::F64], IrType::F64);

        // double round(double x)
        self.declare_external("round", vec![IrType::F64], IrType::F64);

        // double trunc(double x)
        self.declare_external("trunc", vec![IrType::F64], IrType::F64);

        // ============ Absolute Value Functions ============

        // double fabs(double x)
        self.declare_external("fabs", vec![IrType::F64], IrType::F64);

        // float fabsf(float x)
        self.declare_external("fabsf", vec![IrType::F32], IrType::F32);

        // long labs(long x) - for i64
        self.declare_external("labs", vec![IrType::I64], IrType::I64);

        // int abs(int x) - for i32
        self.declare_external("abs", vec![IrType::I32], IrType::I32);

        // ============ Min/Max Functions ============

        // double fmin(double x, double y)
        self.declare_external("fmin", vec![IrType::F64, IrType::F64], IrType::F64);

        // double fmax(double x, double y)
        self.declare_external("fmax", vec![IrType::F64, IrType::F64], IrType::F64);

        // float fminf(float x, float y)
        self.declare_external("fminf", vec![IrType::F32, IrType::F32], IrType::F32);

        // float fmaxf(float x, float y)
        self.declare_external("fmaxf", vec![IrType::F32, IrType::F32], IrType::F32);

        // ============ Other Functions ============

        // double fmod(double x, double y) - floating point remainder
        self.declare_external("fmod", vec![IrType::F64, IrType::F64], IrType::F64);

        // double copysign(double x, double y)
        self.declare_external("copysign", vec![IrType::F64, IrType::F64], IrType::F64);

        // ============ Float versions (f32) ============

        // float sinf(float x)
        self.declare_external("sinf", vec![IrType::F32], IrType::F32);

        // float cosf(float x)
        self.declare_external("cosf", vec![IrType::F32], IrType::F32);

        // float tanf(float x)
        self.declare_external("tanf", vec![IrType::F32], IrType::F32);

        // float sqrtf(float x)
        self.declare_external("sqrtf", vec![IrType::F32], IrType::F32);

        // float powf(float base, float exp)
        self.declare_external("powf", vec![IrType::F32, IrType::F32], IrType::F32);

        // float expf(float x)
        self.declare_external("expf", vec![IrType::F32], IrType::F32);

        // float logf(float x)
        self.declare_external("logf", vec![IrType::F32], IrType::F32);

        // float floorf(float x)
        self.declare_external("floorf", vec![IrType::F32], IrType::F32);

        // float ceilf(float x)
        self.declare_external("ceilf", vec![IrType::F32], IrType::F32);

        // float roundf(float x)
        self.declare_external("roundf", vec![IrType::F32], IrType::F32);

        // float truncf(float x)
        self.declare_external("truncf", vec![IrType::F32], IrType::F32);
    }

    /// Get a reference to stderr (returns VReg pointing to stderr global)
    pub fn get_stderr(&mut self) -> VReg {
        self.emit_with_result(InstrKind::GlobalRef("stderr".to_string()))
    }

    /// Get a reference to stdin (returns VReg pointing to stdin global)
    pub fn get_stdin(&mut self) -> VReg {
        self.emit_with_result(InstrKind::GlobalRef("stdin".to_string()))
    }

    /// Add a global string constant and return its name
    pub fn add_string_constant(&mut self, value: &str) -> String {
        let name = format!(".str.{}", self.next_string);
        self.next_string += 1;

        self.module.globals.push(Global {
            name: name.clone(),
            ty: IrType::Array(Box::new(IrType::I8), value.len() + 1), // +1 for null terminator
            init: Some(Constant::String(value.to_string())),
            is_const: true,
        });

        name
    }

    /// Emit a global string reference (pointer to the first byte)
    pub fn global_string_ptr(&mut self, global_name: &str) -> VReg {
        self.emit_with_result(InstrKind::GlobalRef(global_name.to_string()))
    }

    /// Add a global variable to the module
    pub fn add_global(&mut self, name: impl Into<String>, ty: IrType, init: Option<Constant>, is_const: bool) {
        self.module.globals.push(Global {
            name: name.into(),
            ty,
            init,
            is_const,
        });
    }

    /// Get a reference to a global variable
    pub fn global_ref(&mut self, name: &str) -> VReg {
        self.emit_with_result(InstrKind::GlobalRef(name.to_string()))
    }

    // ============ Block Building ============

    /// Create a new block and return its ID
    pub fn create_block(&mut self) -> BlockId {
        self.fresh_block()
    }

    /// Start building a block (finishes current block first)
    pub fn start_block(&mut self, id: BlockId) {
        if let Some(block) = self.current_block.take() {
            if let Some(ref mut func) = self.current_fn {
                func.blocks.push(block);
            }
        }
        self.current_block = Some(BasicBlock::new(id));
    }

    /// Get the current block ID
    pub fn current_block_id(&self) -> Option<BlockId> {
        self.current_block.as_ref().map(|b| b.id)
    }

    // ============ Instruction Emission ============

    fn emit(&mut self, result: Option<VReg>, kind: InstrKind) -> Option<VReg> {
        if let Some(ref mut block) = self.current_block {
            block.instructions.push(Instruction::new(result, kind));
        }
        result
    }

    fn emit_with_result(&mut self, kind: InstrKind) -> VReg {
        let result = self.fresh_vreg();
        self.emit(Some(result), kind);
        result
    }

    // ============ Constants ============

    /// Emit an integer constant (i64)
    pub fn const_int(&mut self, value: i64) -> VReg {
        self.emit_with_result(InstrKind::Const(Constant::Int(value)))
    }

    /// Emit an i32 constant
    pub fn const_i32(&mut self, value: i32) -> VReg {
        let i64_val = self.emit_with_result(InstrKind::Const(Constant::Int(value as i64)));
        self.emit_with_result(InstrKind::Trunc(i64_val, IrType::I32))
    }

    /// Emit an i8 constant
    pub fn const_i8(&mut self, value: i8) -> VReg {
        let i64_val = self.emit_with_result(InstrKind::Const(Constant::Int(value as i64)));
        self.emit_with_result(InstrKind::Trunc(i64_val, IrType::I8))
    }

    /// Emit a float constant
    pub fn const_float(&mut self, value: f64) -> VReg {
        self.emit_with_result(InstrKind::Const(Constant::Float(value)))
    }

    /// Emit a boolean constant
    pub fn const_bool(&mut self, value: bool) -> VReg {
        self.emit_with_result(InstrKind::Const(Constant::Bool(value)))
    }

    /// Emit a float32 constant
    pub fn const_float32(&mut self, value: f64) -> VReg {
        // Store as f64 but mark as f32 type conceptually
        self.emit_with_result(InstrKind::Const(Constant::Float32(value as f32)))
    }

    /// Emit a null pointer constant
    pub fn const_null(&mut self) -> VReg {
        let zero = self.emit_with_result(InstrKind::Const(Constant::Int(0)));
        self.emit_with_result(InstrKind::IntToPtr(zero, IrType::Ptr(Box::new(IrType::I8))))
    }

    // ============ Arithmetic ============

    pub fn add(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::Add(a, b))
    }

    pub fn sub(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::Sub(a, b))
    }

    pub fn mul(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::Mul(a, b))
    }

    pub fn sdiv(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::SDiv(a, b))
    }

    pub fn srem(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::SRem(a, b))
    }

    pub fn urem(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::URem(a, b))
    }

    pub fn neg(&mut self, v: VReg) -> VReg {
        self.emit_with_result(InstrKind::Neg(v))
    }

    pub fn fadd(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::FAdd(a, b))
    }

    pub fn fsub(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::FSub(a, b))
    }

    pub fn fmul(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::FMul(a, b))
    }

    pub fn fdiv(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::FDiv(a, b))
    }

    pub fn fneg(&mut self, v: VReg) -> VReg {
        self.emit_with_result(InstrKind::FNeg(v))
    }

    // ============ Bitwise ============

    pub fn and(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::And(a, b))
    }

    pub fn or(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::Or(a, b))
    }

    pub fn xor(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::Xor(a, b))
    }

    pub fn shl(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::Shl(a, b))
    }

    pub fn ashr(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::AShr(a, b))
    }

    pub fn lshr(&mut self, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::LShr(a, b))
    }

    pub fn not(&mut self, v: VReg) -> VReg {
        self.emit_with_result(InstrKind::Not(v))
    }

    // ============ Comparison ============

    pub fn icmp(&mut self, op: CmpOp, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::ICmp(op, a, b))
    }

    pub fn fcmp(&mut self, op: CmpOp, a: VReg, b: VReg) -> VReg {
        self.emit_with_result(InstrKind::FCmp(op, a, b))
    }

    // ============ Conversions ============

    pub fn sext(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::SExt(v, ty))
    }

    pub fn zext(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::ZExt(v, ty))
    }

    pub fn trunc(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::Trunc(v, ty))
    }

    pub fn bitcast(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::Bitcast(v, ty))
    }

    pub fn inttoptr(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::IntToPtr(v, ty))
    }

    pub fn ptrtoint(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::PtrToInt(v, ty))
    }

    /// Signed integer to floating point
    pub fn sitofp(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::SIToFP(v, ty))
    }

    /// Floating point to signed integer
    pub fn fptosi(&mut self, v: VReg, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::FPToSI(v, ty))
    }

    // ============ Memory ============

    pub fn alloca(&mut self, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::Alloca(ty))
    }

    /// Allocate memory on the heap (malloc)
    pub fn malloc(&mut self, ty: IrType) -> VReg {
        self.emit_with_result(InstrKind::Malloc(ty))
    }

    /// Allocate array on the heap (malloc with count)
    pub fn malloc_array(&mut self, ty: IrType, count: VReg) -> VReg {
        self.emit_with_result(InstrKind::MallocArray(ty, count))
    }

    /// Free heap memory
    pub fn free(&mut self, ptr: VReg) {
        self.emit(None, InstrKind::Free(ptr));
    }

    /// Reallocate heap memory (returns new pointer)
    pub fn realloc(&mut self, ptr: VReg, new_size: VReg) -> VReg {
        self.emit_with_result(InstrKind::Realloc(ptr, new_size))
    }

    /// Allocate heap memory by byte count (for dynamic sizes)
    pub fn malloc_bytes(&mut self, size: VReg) -> VReg {
        self.emit_with_result(InstrKind::MallocBytes(size))
    }

    /// Allocate zero-initialized heap memory (calloc)
    pub fn calloc(&mut self, size: VReg) -> VReg {
        self.emit_with_result(InstrKind::Calloc(size))
    }

    /// Copy memory from src to dst (memcpy)
    pub fn memcpy(&mut self, dst: VReg, src: VReg, len: VReg) {
        self.emit(None, InstrKind::Memcpy(dst, src, len));
    }

    /// Set memory to a value (memset)
    pub fn memset(&mut self, dst: VReg, val: VReg, len: VReg) {
        self.emit(None, InstrKind::Memset(dst, val, len));
    }

    // ============ Reference Counting (HARC) ============

    /// Allocate with reference count header
    /// Returns pointer to data (after header), with refcount initialized to 1
    pub fn rc_alloc(&mut self, ty: IrType, type_id: u64) -> VReg {
        self.emit_with_result(InstrKind::RcAlloc { ty, type_id })
    }

    /// Increment reference count
    pub fn rc_retain(&mut self, ptr: VReg) {
        self.emit(None, InstrKind::RcRetain(ptr));
    }

    /// Decrement reference count, call destructor if zero
    pub fn rc_release(&mut self, ptr: VReg) {
        self.emit(None, InstrKind::RcRelease(ptr));
    }

    /// Get current reference count (for debugging)
    pub fn rc_get_count(&mut self, ptr: VReg) -> VReg {
        self.emit_with_result(InstrKind::RcGetCount(ptr))
    }

    /// Call type-specific destructor
    pub fn drop_value(&mut self, ptr: VReg, type_id: u64) {
        self.emit(None, InstrKind::Drop { ptr, type_id });
    }

    pub fn load(&mut self, ptr: VReg) -> VReg {
        self.emit_with_result(InstrKind::Load(ptr))
    }

    pub fn store(&mut self, ptr: VReg, value: VReg) {
        self.emit(None, InstrKind::Store(ptr, value));
    }

    pub fn get_field_ptr(&mut self, ptr: VReg, field_idx: u32) -> VReg {
        self.emit_with_result(InstrKind::GetFieldPtr(ptr, field_idx))
    }

    pub fn get_element_ptr(&mut self, ptr: VReg, index: VReg) -> VReg {
        self.emit_with_result(InstrKind::GetElementPtr(ptr, index))
    }

    /// Get byte pointer (ptr + byte_offset) - for i8 array access
    pub fn get_byte_ptr(&mut self, ptr: VReg, offset: VReg) -> VReg {
        self.emit_with_result(InstrKind::GetBytePtr(ptr, offset))
    }

    /// Load single byte from pointer
    pub fn load_byte(&mut self, ptr: VReg) -> VReg {
        self.emit_with_result(InstrKind::LoadByte(ptr))
    }

    // ============ Calls ============

    pub fn call(&mut self, func: impl Into<String>, args: Vec<VReg>) -> VReg {
        self.emit_with_result(InstrKind::Call {
            func: func.into(),
            args,
        })
    }

    pub fn call_void(&mut self, func: impl Into<String>, args: Vec<VReg>) {
        self.emit(None, InstrKind::Call {
            func: func.into(),
            args,
        });
    }

    /// Get a function pointer
    pub fn func_ref(&mut self, name: impl Into<String>) -> VReg {
        self.emit_with_result(InstrKind::FuncRef(name.into()))
    }

    /// Call through a function pointer
    pub fn call_ptr(&mut self, ptr: VReg, args: Vec<VReg>) -> VReg {
        self.emit_with_result(InstrKind::CallPtr { ptr, args })
    }

    /// Call an f64 math function (helper that returns f64 type result)
    pub fn call_f64(&mut self, func: impl Into<String>, args: Vec<VReg>) -> VReg {
        self.emit_with_result(InstrKind::Call {
            func: func.into(),
            args,
        })
    }

    /// Call an f32 math function (helper that returns f32 type result)
    pub fn call_f32(&mut self, func: impl Into<String>, args: Vec<VReg>) -> VReg {
        self.emit_with_result(InstrKind::Call {
            func: func.into(),
            args,
        })
    }

    // ============ Trait Objects ============

    /// Create a trait object (fat pointer) from data pointer and vtable
    ///
    /// The result is a struct containing {data_ptr, vtable_ptr}.
    /// The vtable parameter is the name of the global vtable constant.
    pub fn make_trait_object(&mut self, data_ptr: VReg, vtable: impl Into<String>) -> VReg {
        self.emit_with_result(InstrKind::MakeTraitObject {
            data_ptr,
            vtable: vtable.into(),
        })
    }

    /// Extract data pointer from trait object (field 0)
    pub fn get_data_ptr(&mut self, trait_obj: VReg) -> VReg {
        self.emit_with_result(InstrKind::GetDataPtr(trait_obj))
    }

    /// Extract vtable pointer from trait object (field 1)
    pub fn get_vtable_ptr(&mut self, trait_obj: VReg) -> VReg {
        self.emit_with_result(InstrKind::GetVTablePtr(trait_obj))
    }

    /// Call method through vtable
    ///
    /// The method_idx is the index into the vtable:
    /// - 0: drop function
    /// - 1: size
    /// - 2: align
    /// - 3+: trait methods in declaration order
    ///
    /// The data_ptr is automatically extracted and prepended as the first argument.
    pub fn vtable_call(&mut self, trait_obj: VReg, method_idx: u32, args: Vec<VReg>) -> VReg {
        self.emit_with_result(InstrKind::VTableCall {
            trait_obj,
            method_idx,
            args,
        })
    }

    /// Call method through vtable (void return)
    pub fn vtable_call_void(&mut self, trait_obj: VReg, method_idx: u32, args: Vec<VReg>) {
        self.emit(None, InstrKind::VTableCall {
            trait_obj,
            method_idx,
            args,
        });
    }

    // ============ Control Flow ============

    pub fn select(&mut self, cond: VReg, then_val: VReg, else_val: VReg) -> VReg {
        self.emit_with_result(InstrKind::Select(cond, then_val, else_val))
    }

    pub fn phi(&mut self, preds: Vec<(VReg, BlockId)>) -> VReg {
        self.emit_with_result(InstrKind::Phi(preds))
    }

    // ============ Terminators ============

    pub fn ret(&mut self, value: Option<VReg>) {
        if let Some(ref mut block) = self.current_block {
            block.terminator = Some(Terminator::Ret(value));
        }
    }

    pub fn br(&mut self, target: BlockId) {
        if let Some(ref mut block) = self.current_block {
            block.terminator = Some(Terminator::Br(target));
        }
    }

    pub fn cond_br(&mut self, cond: VReg, then_block: BlockId, else_block: BlockId) {
        if let Some(ref mut block) = self.current_block {
            block.terminator = Some(Terminator::CondBr {
                cond,
                then_block,
                else_block,
            });
        }
    }

    pub fn unreachable(&mut self) {
        if let Some(ref mut block) = self.current_block {
            block.terminator = Some(Terminator::Unreachable);
        }
    }
}
