pub type IOPair<const I: usize, const O: usize> = ([f64; I], [f64; O]);

pub type DataSet<const I: usize, const O: usize> = Vec<IOPair<I, O>>;

#[macro_export]
macro_rules! dataset {
    ($([$($k: expr),* $(,)?] => [$($v: expr),* $(,)?]),* $(,)?) => {{
        let mut t = $crate::dataset::DataSet::new();
        $(
            t.push((
                [$($k),*],
                [$($v),*]
            ));
        )*
        t
    }}
}
