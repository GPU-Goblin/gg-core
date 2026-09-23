//! Synthetic regression dataset for `mlp.train.v1`, fully determined by the protocol DRBG.
//!
//! A one-hidden-layer tanh "teacher" is drawn from `mlp.data.v1/<seed>`; inputs are uniform in
//! [-1, 1); targets are the teacher output plus approximately Gaussian noise (sum of 12 uniforms).

use gg_protocol::drbg::Drbg;
use gg_protocol::tensor_file::TensorF32;

/// Dataset generation parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticMlpConfig {
    /// Input width.
    pub d_input: u32,
    /// Output width.
    pub d_output: u32,
    /// Teacher hidden width.
    pub d_teacher: u32,
    /// Training examples.
    pub n_train: u32,
    /// Evaluation examples.
    pub n_eval: u32,
    /// Seed.
    pub seed: u64,
    /// Noise amplitude (standard deviation).
    pub noise: f32,
}

/// Materialized dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    /// `[n_train, d_input]`.
    pub train_x: TensorF32,
    /// `[n_train, d_output]`.
    pub train_y: TensorF32,
    /// `[n_eval, d_input]`.
    pub eval_x: TensorF32,
    /// `[n_eval, d_output]`.
    pub eval_y: TensorF32,
}

/// Maximum examples per split.
pub const MAX_EXAMPLES: u32 = 65_536;

struct Teacher {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d_in: usize,
    d_h: usize,
    d_out: usize,
}

impl Teacher {
    fn draw(d: &mut Drbg, d_in: usize, d_h: usize, d_out: usize) -> Self {
        let s1 = 2.0 / (d_in as f32).sqrt();
        let s2 = 1.0 / (d_h as f32).sqrt();
        let w1 = (0..d_in * d_h).map(|_| d.next_sym_f32() * s1).collect();
        let b1 = (0..d_h).map(|_| d.next_sym_f32() * 0.5).collect();
        let w2 = (0..d_h * d_out).map(|_| d.next_sym_f32() * s2).collect();
        let b2 = (0..d_out).map(|_| d.next_sym_f32() * 0.5).collect();
        Self {
            w1,
            b1,
            w2,
            b2,
            d_in,
            d_h,
            d_out,
        }
    }

    fn forward(&self, x: &[f32], out: &mut [f32]) {
        let mut h = vec![0.0f32; self.d_h];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = self.b1[j];
            for (i, xi) in x.iter().enumerate() {
                acc += xi * self.w1[i * self.d_h + j];
            }
            *hj = acc.tanh();
        }
        for (k, ok) in out.iter_mut().enumerate() {
            let mut acc = self.b2[k];
            for (j, hj) in h.iter().enumerate() {
                acc += hj * self.w2[j * self.d_out + k];
            }
            *ok = acc;
        }
    }
}

fn approx_gaussian(d: &mut Drbg) -> f32 {
    // Irwin–Hall: sum of 12 uniforms in [0,1) minus 6 has mean 0 and variance 1.
    let mut s = 0.0f32;
    for _ in 0..12 {
        s += d.next_unit_f32();
    }
    s - 6.0
}

fn split(d: &mut Drbg, teacher: &Teacher, n: usize, noise: f32) -> (TensorF32, TensorF32) {
    let mut xs = Vec::with_capacity(n * teacher.d_in);
    let mut ys = Vec::with_capacity(n * teacher.d_out);
    let mut out = vec![0.0f32; teacher.d_out];
    let mut x = vec![0.0f32; teacher.d_in];
    for _ in 0..n {
        for xi in x.iter_mut() {
            *xi = d.next_sym_f32();
        }
        teacher.forward(&x, &mut out);
        xs.extend_from_slice(&x);
        for o in &out {
            ys.push(o + noise * approx_gaussian(d));
        }
    }
    (
        TensorF32::new(vec![n as u64, teacher.d_in as u64], xs).expect("shape"),
        TensorF32::new(vec![n as u64, teacher.d_out as u64], ys).expect("shape"),
    )
}

impl SyntheticMlpConfig {
    /// Validate bounds.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_input == 0 || self.d_input > 256 || self.d_output == 0 || self.d_output > 64 {
            return Err("d_input in 1..=256 and d_output in 1..=64".into());
        }
        if self.d_teacher == 0 || self.d_teacher > 512 {
            return Err("d_teacher in 1..=512".into());
        }
        if self.n_train == 0
            || self.n_train > MAX_EXAMPLES
            || self.n_eval == 0
            || self.n_eval > MAX_EXAMPLES
        {
            return Err(format!("n_train and n_eval in 1..={MAX_EXAMPLES}"));
        }
        if !self.noise.is_finite() || self.noise < 0.0 || self.noise > 1.0 {
            return Err("noise in [0, 1]".into());
        }
        Ok(())
    }

    /// Generate the dataset deterministically.
    pub fn generate(&self) -> Result<Dataset, String> {
        self.validate()?;
        let mut d = Drbg::new(&format!("mlp.data.v1/{}", self.seed), self.seed);
        let teacher = Teacher::draw(
            &mut d,
            self.d_input as usize,
            self.d_teacher as usize,
            self.d_output as usize,
        );
        let (train_x, train_y) = split(&mut d, &teacher, self.n_train as usize, self.noise);
        let (eval_x, eval_y) = split(&mut d, &teacher, self.n_eval as usize, self.noise);
        Ok(Dataset {
            train_x,
            train_y,
            eval_x,
            eval_y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_bounded() {
        let cfg = SyntheticMlpConfig {
            d_input: 4,
            d_output: 2,
            d_teacher: 8,
            n_train: 64,
            n_eval: 16,
            seed: 3,
            noise: 0.05,
        };
        let a = cfg.generate().unwrap();
        let b = cfg.generate().unwrap();
        assert_eq!(a, b);
        assert_eq!(a.train_x.shape, vec![64, 4]);
        assert_eq!(a.train_y.shape, vec![64, 2]);
        assert_eq!(a.eval_x.shape, vec![16, 4]);
        assert!(a.train_x.data.iter().all(|v| (-1.0..1.0).contains(v)));
        assert!(a.train_y.check_finite().is_ok());
        let other = SyntheticMlpConfig {
            seed: 4,
            ..cfg.clone()
        }
        .generate()
        .unwrap();
        assert_ne!(a.train_y, other.train_y);
        assert!(SyntheticMlpConfig { n_train: 0, ..cfg }.generate().is_err());
    }
}
