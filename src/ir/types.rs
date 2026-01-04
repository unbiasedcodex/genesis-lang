//! IR Types
//!
//! Type representations for the Genesis IR.

use std::collections::HashMap;
use std::fmt;

/// A virtual register (SSA value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(pub u32);

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// A basic block label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// IR types (simplified from the full type system)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrType {
    /// Void/unit type
    Void,
    /// Boolean (1 bit)
    Bool,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// Pointer to another type
    Ptr(Box<IrType>),
    /// Array of elements
    Array(Box<IrType>, usize),
    /// Struct with named fields
    Struct(Vec<IrType>),
    /// Function type: (params) -> ret
    Fn {
        params: Vec<IrType>,
        ret: Box<IrType>,
    },
}

impl IrType {
    pub fn ptr(inner: IrType) -> Self {
        IrType::Ptr(Box::new(inner))
    }

    pub fn array(element: IrType, size: usize) -> Self {
        IrType::Array(Box::new(element), size)
    }

    /// Size in bytes (approximate, platform-dependent)
    pub fn size(&self) -> usize {
        match self {
            IrType::Void => 0,
            IrType::Bool | IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 | IrType::F32 => 4,
            IrType::I64 | IrType::F64 => 8,
            IrType::Ptr(_) => 8, // Assume 64-bit pointers
            IrType::Array(elem, size) => elem.size() * size,
            IrType::Struct(fields) => fields.iter().map(|f| f.size()).sum(),
            IrType::Fn { .. } => 8, // Function pointer
        }
    }

    /// Is this type a pointer?
    pub fn is_ptr(&self) -> bool {
        matches!(self, IrType::Ptr(_))
    }

    /// Is this type an integer?
    pub fn is_int(&self) -> bool {
        matches!(self, IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64)
    }

    /// Is this type a float?
    pub fn is_float(&self) -> bool {
        matches!(self, IrType::F32 | IrType::F64)
    }
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Void => write!(f, "void"),
            IrType::Bool => write!(f, "i1"),
            IrType::I8 => write!(f, "i8"),
            IrType::I16 => write!(f, "i16"),
            IrType::I32 => write!(f, "i32"),
            IrType::I64 => write!(f, "i64"),
            IrType::F32 => write!(f, "f32"),
            IrType::F64 => write!(f, "f64"),
            IrType::Ptr(inner) => write!(f, "*{}", inner),
            IrType::Array(elem, size) => write!(f, "[{} x {}]", size, elem),
            IrType::Struct(fields) => {
                write!(f, "{{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field)?;
                }
                write!(f, "}}")
            }
            IrType::Fn { params, ret } => {
                write!(f, "(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
        }
    }
}

/// VTable layout information for trait objects
#[derive(Debug, Clone)]
pub struct VTableLayout {
    /// Name of the vtable global (e.g., "__vtable_Animal_Dog")
    pub vtable_name: String,
    /// Name of the trait
    pub trait_name: String,
    /// Name of the implementing type
    pub type_name: String,
    /// Size of the concrete type in bytes
    pub size: usize,
    /// Alignment of the concrete type
    pub align: usize,
    /// Drop function name (e.g., "__drop_Dog_for_Animal")
    pub drop_fn_name: String,
    /// Method function names in vtable order (after drop, size, align)
    pub methods: Vec<String>,
}

/// A module contains functions and global definitions
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub functions: Vec<Function>,
    pub globals: Vec<Global>,
    pub structs: Vec<StructDef>,
    /// VTable layouts for trait objects
    pub vtable_layouts: HashMap<String, VTableLayout>,
}

impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            globals: Vec::new(),
            structs: Vec::new(),
            vtable_layouts: HashMap::new(),
        }
    }
}

/// A function in the IR
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<(VReg, IrType)>,
    pub ret_type: IrType,
    pub blocks: Vec<BasicBlock>,
    pub is_external: bool,
    pub is_vararg: bool,
}

impl Function {
    pub fn new(name: impl Into<String>, params: Vec<(VReg, IrType)>, ret_type: IrType) -> Self {
        Self {
            name: name.into(),
            params,
            ret_type,
            blocks: Vec::new(),
            is_external: false,
            is_vararg: false,
        }
    }

    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.first()
    }
}

/// A basic block contains a sequence of instructions
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub label: Option<String>,
    pub instructions: Vec<super::Instruction>,
    pub terminator: Option<super::Terminator>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            label: None,
            instructions: Vec::new(),
            terminator: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A global variable or constant
#[derive(Debug, Clone)]
pub struct Global {
    pub name: String,
    pub ty: IrType,
    pub init: Option<Constant>,
    pub is_const: bool,
}

/// A struct type definition
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, IrType)>,
}

/// A constant value
#[derive(Debug, Clone)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Float32(f32),
    Bool(bool),
    Null,
    String(String),
    Array(Vec<Constant>),
    Struct(Vec<Constant>),
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Int(v) => write!(f, "{}", v),
            Constant::Float(v) => write!(f, "{}", v),
            Constant::Float32(v) => write!(f, "{}f", v),
            Constant::Bool(v) => write!(f, "{}", if *v { "true" } else { "false" }),
            Constant::Null => write!(f, "null"),
            Constant::String(s) => write!(f, "{:?}", s),
            Constant::Array(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Constant::Struct(fields) => {
                write!(f, "{{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field)?;
                }
                write!(f, "}}")
            }
        }
    }
}
