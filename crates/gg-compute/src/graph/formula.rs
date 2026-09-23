//! `Formula` operator: an element-wise expression written in the description (ADR-0025).
//!
//! This is the platform's code-free "custom operator": the researcher writes an expression,
//! `softplus(a) * tanh(b) + 0.5`, over one to four inputs named `a`, `b`, `c`, `d` that broadcast
//! as for `Add`. The grammar is closed (four operations, a power with a constant exponent and a
//! finite list of functions) and bounded in length, size and depth. The executor evaluates it
//! with Burn operations: the backward pass follows by automatic differentiation, on the CPU as on
//! the GPU, with nothing more to write.
//!
//! No loops, no variables, no calls outside the list: what gets computed is what is written, and
//! the cost can be read from the expression.

/// Maximum length of an expression, in bytes.
pub const MAX_LEN: usize = 1024;
/// Maximum number of terms (numbers, variables, operations, calls).
pub const MAX_TERMS: usize = 128;
/// Maximum nesting depth.
pub const MAX_DEPTH: usize = 32;
/// Input names, in the node's order.
pub const VARIABLES: [&str; 4] = ["a", "b", "c", "d"];

/// Binary operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

/// Function from the published list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Func {
    /// `exp(x)`
    Exp,
    /// `log(x)`, natural logarithm.
    Log,
    /// `log1p(x)` = `log(1 + x)`, accurate near zero.
    Log1p,
    /// `sqrt(x)`
    Sqrt,
    /// `rsqrt(x)` = `1 / sqrt(x)`
    Rsqrt,
    /// `abs(x)`
    Abs,
    /// `tanh(x)`
    Tanh,
    /// `sigmoid(x)`
    Sigmoid,
    /// `relu(x)` = `max(x, 0)`
    Relu,
    /// `sin(x)`
    Sin,
    /// `cos(x)`
    Cos,
    /// `erf(x)`
    Erf,
    /// `softplus(x)` = `log(1 + exp(x))`, computed in the stable form `relu(x) + log1p(exp(−|x|))`.
    Softplus,
    /// `max(x, y)`
    Max,
    /// `min(x, y)`
    Min,
    /// `tan(x)`
    Tan,
    /// `asin(x)`, defined on [−1, 1].
    Asin,
    /// `acos(x)`, defined on [−1, 1].
    Acos,
    /// `atan(x)`
    Atan,
    /// `sinh(x)`
    Sinh,
    /// `cosh(x)`
    Cosh,
    /// `asinh(x)`
    Asinh,
    /// `acosh(x)`, defined on [1, +∞).
    Acosh,
    /// `atanh(x)`, defined on (−1, 1).
    Atanh,
    /// `atan2(y, x)`: angle of the point `(x, y)`, in (−π, π], as `torch.atan2(y, x)`.
    Atan2,
    /// `floor(x)`. Piecewise constant: its derivative is zero wherever it exists (and undefined at
    /// the integers); automatic differentiation returns zero, as PyTorch does.
    Floor,
    /// `ceil(x)`. Piecewise constant, derivative zero (see `floor`).
    Ceil,
    /// `round(x)`, half to even (`round(2.5) = 2`), as PyTorch. Piecewise constant, derivative
    /// zero (see `floor`).
    Round,
    /// `trunc(x)`, rounding toward zero. Piecewise constant, derivative zero (see `floor`).
    Trunc,
    /// `sign(x)`: −1, 0 or 1. Piecewise constant, derivative zero (see `floor`).
    Sign,
}

impl Func {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "exp" => Func::Exp,
            "log" => Func::Log,
            "log1p" => Func::Log1p,
            "sqrt" => Func::Sqrt,
            "rsqrt" => Func::Rsqrt,
            "abs" => Func::Abs,
            "tanh" => Func::Tanh,
            "sigmoid" => Func::Sigmoid,
            "relu" => Func::Relu,
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "erf" => Func::Erf,
            "softplus" => Func::Softplus,
            "max" => Func::Max,
            "min" => Func::Min,
            "tan" => Func::Tan,
            "asin" => Func::Asin,
            "acos" => Func::Acos,
            "atan" => Func::Atan,
            "sinh" => Func::Sinh,
            "cosh" => Func::Cosh,
            "asinh" => Func::Asinh,
            "acosh" => Func::Acosh,
            "atanh" => Func::Atanh,
            "atan2" => Func::Atan2,
            "floor" => Func::Floor,
            "ceil" => Func::Ceil,
            "round" => Func::Round,
            "trunc" => Func::Trunc,
            "sign" => Func::Sign,
            _ => return None,
        })
    }

    /// Number of arguments.
    pub fn arity(self) -> usize {
        match self {
            Func::Max | Func::Min | Func::Atan2 => 2,
            _ => 1,
        }
    }

    /// True for the functions added with `gg-ops/7`: a worker of an earlier profile does not
    /// know them.
    pub fn since_profile_7(self) -> bool {
        matches!(
            self,
            Func::Tan
                | Func::Asin
                | Func::Acos
                | Func::Atan
                | Func::Sinh
                | Func::Cosh
                | Func::Asinh
                | Func::Acosh
                | Func::Atanh
                | Func::Atan2
                | Func::Floor
                | Func::Ceil
                | Func::Round
                | Func::Trunc
                | Func::Sign
        )
    }

    /// True for the piecewise-constant functions: their derivative is zero wherever it exists.
    pub fn is_piecewise_constant(self) -> bool {
        matches!(
            self,
            Func::Floor | Func::Ceil | Func::Round | Func::Trunc | Func::Sign
        )
    }

    /// Cost per element, in floating-point operations, for estimation.
    fn flops(self) -> u64 {
        match self {
            Func::Abs
            | Func::Relu
            | Func::Max
            | Func::Min
            | Func::Floor
            | Func::Ceil
            | Func::Trunc
            | Func::Sign => 1,
            // Half-to-even rounding is written with floor, comparisons and selections.
            Func::Round => 6,
            Func::Softplus => 6,
            _ => 4,
        }
    }

    /// Value on numbers, for constant subexpressions.
    fn eval(self, x: f64, y: f64) -> f64 {
        match self {
            Func::Exp => x.exp(),
            Func::Log => x.ln(),
            Func::Log1p => x.ln_1p(),
            Func::Sqrt => x.sqrt(),
            Func::Rsqrt => 1.0 / x.sqrt(),
            Func::Abs => x.abs(),
            Func::Tanh => x.tanh(),
            Func::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Func::Relu => x.max(0.0),
            Func::Sin => x.sin(),
            Func::Cos => x.cos(),
            Func::Erf => erf(x),
            Func::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
            Func::Max => x.max(y),
            Func::Min => x.min(y),
            Func::Tan => x.tan(),
            Func::Asin => x.asin(),
            Func::Acos => x.acos(),
            Func::Atan => x.atan(),
            Func::Sinh => x.sinh(),
            Func::Cosh => x.cosh(),
            Func::Asinh => x.asinh(),
            Func::Acosh => x.acosh(),
            Func::Atanh => x.atanh(),
            Func::Atan2 => x.atan2(y),
            Func::Floor => x.floor(),
            Func::Ceil => x.ceil(),
            Func::Round => x.round_ties_even(),
            Func::Trunc => x.trunc(),
            Func::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    x
                }
            }
        }
    }
}

/// Function names, for messages and documentation.
pub const FUNCTIONS: &[&str] = &[
    "exp", "log", "log1p", "sqrt", "rsqrt", "abs", "tanh", "sigmoid", "relu", "sin", "cos", "erf",
    "softplus", "max", "min", "tan", "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh",
    "atanh", "atan2", "floor", "ceil", "round", "trunc", "sign",
];

/// Parsed expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Finite number.
    Num(f64),
    /// Input `a`, `b`, `c` or `d` (index into the node's inputs).
    Var(usize),
    /// Negation.
    Neg(Box<Expr>),
    /// Binary operation.
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// Power with a constant exponent.
    Pow(Box<Expr>, f64),
    /// Call to a function from the list.
    Call(Func, Vec<Expr>),
}

impl Expr {
    /// True if the expression depends on no input.
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Num(_) => true,
            Expr::Var(_) => false,
            Expr::Neg(e) | Expr::Pow(e, _) => e.is_constant(),
            Expr::Bin(_, l, r) => l.is_constant() && r.is_constant(),
            Expr::Call(_, args) => args.iter().all(Expr::is_constant),
        }
    }

    /// Value of a constant expression.
    pub fn constant_value(&self) -> Option<f64> {
        Some(match self {
            Expr::Num(v) => *v,
            Expr::Var(_) => return None,
            Expr::Neg(e) => -e.constant_value()?,
            Expr::Pow(e, p) => e.constant_value()?.powf(*p),
            Expr::Bin(op, l, r) => {
                let (x, y) = (l.constant_value()?, r.constant_value()?);
                match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => x / y,
                }
            }
            Expr::Call(f, args) => {
                let x = args.first()?.constant_value()?;
                let y = match args.get(1) {
                    Some(a) => a.constant_value()?,
                    None => 0.0,
                };
                f.eval(x, y)
            }
        })
    }

    fn visit(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Num(_) | Expr::Var(_) => {}
            Expr::Neg(e) | Expr::Pow(e, _) => e.visit(f),
            Expr::Bin(_, l, r) => {
                l.visit(f);
                r.visit(f);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| a.visit(f)),
        }
    }

    /// Floating-point operations per output element.
    pub fn flops_per_element(&self) -> u64 {
        let mut total = 0u64;
        self.visit(&mut |e| {
            if e.is_constant() {
                return;
            }
            total += match e {
                Expr::Num(_) | Expr::Var(_) => 0,
                Expr::Neg(_) | Expr::Bin(..) => 1,
                Expr::Pow(_, p) if p.fract() == 0.0 && p.abs() <= 8.0 => p.abs().max(1.0) as u64,
                Expr::Pow(..) => 4,
                Expr::Call(func, _) => func.flops(),
            };
        });
        total
    }

    /// Number of intermediate tensors the evaluation produces, hence that automatic
    /// differentiation may keep for the backward pass. Used for memory estimation.
    pub fn intermediates(&self) -> u64 {
        let mut total = 0u64;
        self.visit(&mut |e| {
            if e.is_constant() {
                return;
            }
            // Tensors the executor creates for this term (see `exec::formula_eval`).
            total += match e {
                Expr::Num(_) | Expr::Var(_) => 0,
                Expr::Pow(_, p) if p.fract() == 0.0 && p.abs() <= 8.0 => {
                    (p.abs() as u64).saturating_sub(1).max(1) + u64::from(*p < 0.0)
                }
                Expr::Call(Func::Softplus, _) => 6,
                Expr::Call(Func::Round, _) => 6,
                Expr::Call(Func::Rsqrt, _) => 2,
                _ => 1,
            };
        });
        total.max(1)
    }

    /// True if the expression calls a function added with `gg-ops/7`.
    pub fn needs_profile_7(&self) -> bool {
        let mut found = false;
        self.visit(&mut |e| {
            if let Expr::Call(f, _) = e {
                found |= f.since_profile_7();
            }
        });
        found
    }

    /// Referenced inputs, by index.
    pub fn variables(&self) -> [bool; 4] {
        let mut used = [false; 4];
        self.visit(&mut |e| {
            if let Expr::Var(i) = e {
                used[*i] = true;
            }
        });
        used
    }
}

/// Parse or check error of an expression; the message says where.
pub type FormulaError = String;

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    terms: usize,
}

impl Parser<'_> {
    fn skip(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.src.get(self.pos).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn fail<T>(&self, what: &str) -> Result<T, FormulaError> {
        Err(format!("expression, position {}: {what}", self.pos))
    }

    fn term(&mut self) -> Result<(), FormulaError> {
        self.terms += 1;
        if self.terms > MAX_TERMS {
            return self.fail(&format!("more than {MAX_TERMS} terms"));
        }
        Ok(())
    }

    fn depth(&self, depth: usize) -> Result<(), FormulaError> {
        if depth > MAX_DEPTH {
            return self.fail(&format!("nesting deeper than {MAX_DEPTH} levels"));
        }
        Ok(())
    }

    // sum := product (('+' | '-') product)*
    fn sum(&mut self, depth: usize) -> Result<Expr, FormulaError> {
        self.depth(depth)?;
        let mut left = self.product(depth + 1)?;
        loop {
            let op = if self.eat(b'+') {
                BinOp::Add
            } else if self.eat(b'-') {
                BinOp::Sub
            } else {
                return Ok(left);
            };
            self.term()?;
            let right = self.product(depth + 1)?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
    }

    // product := unary (('*' | '/') unary)*
    fn product(&mut self, depth: usize) -> Result<Expr, FormulaError> {
        self.depth(depth)?;
        let mut left = self.unary(depth + 1)?;
        loop {
            let op = if self.eat(b'*') {
                BinOp::Mul
            } else if self.eat(b'/') {
                BinOp::Div
            } else {
                return Ok(left);
            };
            self.term()?;
            let right = self.unary(depth + 1)?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
    }

    // unary := '-' unary | power
    fn unary(&mut self, depth: usize) -> Result<Expr, FormulaError> {
        self.depth(depth)?;
        if self.eat(b'-') {
            self.term()?;
            return Ok(Expr::Neg(Box::new(self.unary(depth + 1)?)));
        }
        self.power(depth + 1)
    }

    // power := atom ('^' ['-'] number)?
    fn power(&mut self, depth: usize) -> Result<Expr, FormulaError> {
        let base = self.atom(depth)?;
        if !self.eat(b'^') {
            return Ok(base);
        }
        self.term()?;
        let negative = self.eat(b'-');
        let p = match self.number()? {
            Some(v) => v,
            None => return self.fail("the exponent of a power must be a literal number"),
        };
        Ok(Expr::Pow(Box::new(base), if negative { -p } else { p }))
    }

    fn number(&mut self) -> Result<Option<f64>, FormulaError> {
        self.skip();
        let start = self.pos;
        let s = self.src;
        let digits = |p: &mut usize| {
            while *p < s.len() && s[*p].is_ascii_digit() {
                *p += 1;
            }
        };
        let mut p = self.pos;
        digits(&mut p);
        if p < s.len() && s[p] == b'.' {
            p += 1;
            digits(&mut p);
        }
        if p == start || (p == start + 1 && s[start] == b'.') {
            return Ok(None);
        }
        if p < s.len() && (s[p] == b'e' || s[p] == b'E') {
            let mut q = p + 1;
            if q < s.len() && (s[q] == b'+' || s[q] == b'-') {
                q += 1;
            }
            let before = q;
            digits(&mut q);
            if q == before {
                self.pos = q;
                return self.fail("incomplete decimal exponent");
            }
            p = q;
        }
        let text = std::str::from_utf8(&s[start..p]).expect("ASCII digits");
        let v: f64 = text
            .parse()
            .map_err(|_| format!("expression: unreadable number '{text}'"))?;
        if !v.is_finite() {
            return self.fail(&format!("non-finite number '{text}'"));
        }
        self.pos = p;
        Ok(Some(v))
    }

    fn atom(&mut self, depth: usize) -> Result<Expr, FormulaError> {
        self.depth(depth)?;
        self.term()?;
        if self.eat(b'(') {
            let e = self.sum(depth + 1)?;
            if !self.eat(b')') {
                return self.fail("expected ')'");
            }
            return Ok(e);
        }
        if let Some(v) = self.number()? {
            return Ok(Expr::Num(v));
        }
        self.skip();
        let start = self.pos;
        while self.pos < self.src.len()
            && (self.src[self.pos].is_ascii_alphanumeric() || self.src[self.pos] == b'_')
        {
            self.pos += 1;
        }
        if start == self.pos {
            return match self.peek() {
                Some(c) => self.fail(&format!("unexpected character '{}'", c as char)),
                None => self.fail("incomplete expression"),
            };
        }
        let name = std::str::from_utf8(&self.src[start..self.pos]).expect("ASCII");
        if let Some(i) = VARIABLES.iter().position(|v| *v == name) {
            return Ok(Expr::Var(i));
        }
        let Some(func) = Func::from_name(name) else {
            return Err(format!(
                "expression: '{name}' is neither an input (a, b, c, d) nor a known function \
({})",
                FUNCTIONS.join(", ")
            ));
        };
        if !self.eat(b'(') {
            return self.fail(&format!("expected '(' after {name}"));
        }
        let mut args = vec![self.sum(depth + 1)?];
        while self.eat(b',') {
            args.push(self.sum(depth + 1)?);
        }
        if !self.eat(b')') {
            return self.fail("expected ')'");
        }
        if args.len() != func.arity() {
            return Err(format!(
                "expression: {name} takes {} argument(s), {} given",
                func.arity(),
                args.len()
            ));
        }
        Ok(Expr::Call(func, args))
    }
}

/// Parses an expression. The bounds (length, terms, depth) are checked here.
pub fn parse(src: &str) -> Result<Expr, FormulaError> {
    if src.len() > MAX_LEN {
        return Err(format!(
            "expression of {} bytes, the maximum is {MAX_LEN}",
            src.len()
        ));
    }
    if !src.is_ascii() {
        return Err("expression: only ASCII characters are accepted".into());
    }
    let mut p = Parser {
        src: src.as_bytes(),
        pos: 0,
        terms: 0,
    };
    let e = p.sum(0)?;
    if p.peek().is_some() {
        return p.fail("trailing text after the expression");
    }
    Ok(e)
}

/// Parses the expression of a node with `arity` inputs and checks that each one appears in it:
/// an ignored input would receive no gradient and the output shape would no longer be the one
/// declared.
pub fn check(src: &str, arity: usize) -> Result<Expr, FormulaError> {
    let e = parse(src)?;
    let used = e.variables();
    for (i, u) in used.iter().enumerate() {
        match (i < arity, *u) {
            (false, true) => {
                return Err(format!(
                    "expression: input '{}' is used but the node only has {arity}",
                    VARIABLES[i]
                ));
            }
            (true, false) => {
                return Err(format!(
                    "expression: input '{}' is provided but never used",
                    VARIABLES[i]
                ));
            }
            _ => {}
        }
    }
    Ok(e)
}

/// Error function, Abramowitz and Stegun approximation 7.1.26 (error < 1.5e-7). Only used for
/// constant subexpressions, folded before execution.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_and_associativity() {
        let e = parse("a - b - c * d ^ 2").unwrap();
        // ((a - b) - (c * (d^2)))
        let Expr::Bin(BinOp::Sub, l, r) = e else {
            panic!()
        };
        assert!(matches!(*l, Expr::Bin(BinOp::Sub, _, _)));
        assert!(matches!(*r, Expr::Bin(BinOp::Mul, _, _)));
        assert_eq!(
            parse("-a^2").unwrap(),
            Expr::Neg(Box::new(Expr::Pow(Box::new(Expr::Var(0)), 2.0)))
        );
    }

    #[test]
    fn numbers() {
        assert_eq!(parse("1.5e-3").unwrap(), Expr::Num(1.5e-3));
        assert_eq!(parse(".5").unwrap(), Expr::Num(0.5));
        assert!(parse("1e").is_err());
        assert!(parse("1e999").unwrap_err().contains("non-finite"));
    }

    #[test]
    fn rejections_say_why() {
        assert!(parse("a +").unwrap_err().contains("incomplete"));
        assert!(parse("pow(a, 2)").unwrap_err().contains("neither an input"));
        assert!(parse("max(a)").unwrap_err().contains("2 argument"));
        assert!(parse("a ^ b").unwrap_err().contains("literal number"));
        assert!(parse("(a").unwrap_err().contains("')'"));
        assert!(parse("a b").unwrap_err().contains("trailing text"));
        assert!(parse("é").unwrap_err().contains("ASCII"));
        assert!(check("a + b", 1).unwrap_err().contains("only has 1"));
        assert!(check("a * 2", 2).unwrap_err().contains("never used"));
    }

    #[test]
    fn bounds() {
        let long = vec!["a"; 200].join(" + ");
        assert!(parse(&long).unwrap_err().contains("terms"));
        let deep = format!("{}a{}", "(".repeat(40), ")".repeat(40));
        assert!(parse(&deep).unwrap_err().contains("nesting"));
        assert!(
            parse(&"1".repeat(MAX_LEN + 1))
                .unwrap_err()
                .contains("maximum")
        );
    }

    #[test]
    fn folded_constants_and_costs() {
        assert_eq!(
            parse("2 * 3 + sqrt(16)").unwrap().constant_value(),
            Some(10.0)
        );
        let e = parse("a * sigmoid(a) + 1 / 2").unwrap();
        assert_eq!(e.flops_per_element(), 1 + 1 + 4);
        assert_eq!(e.intermediates(), 3);
        assert!((erf(0.5) - 0.520_499_877_8).abs() < 2e-7);
    }
}
