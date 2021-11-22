//type definaions for input-output pair and dataset
//and macro for quick intialization
pub type IOPair<'a> = (&'a [f64], &'a [f64]);

pub type DataSet<'a> = Vec<IOPair<'a>>;

#[macro_export]
macro_rules! dataset {
    ($([$($k: expr),* $(,)?] => [$($v: expr),* $(,)?]),* $(,)?) => {{
        let mut t = $crate::dataset::DataSet::new();
        $(
            t.push((
                &[$($k),*],
                &[$($v),*]
            ));
        )*
        t
    }}
}
