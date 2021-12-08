//type definaions for input-output pair and dataset
//and macro for quick intialization
pub type IOPair = (Vec<f64>, Vec<f64>);

pub type DataSet = Vec<IOPair>;

#[macro_export]
macro_rules! dataset {
    ($([$($k: expr),* $(,)?] => [$($v: expr),* $(,)?]),* $(,)?) => {{
        let mut t = $crate::dataset::DataSet::new();
        $(
            t.push((
                vec![$($k),*],
                vec![$($v),*]
            ));
        )*
        t
    }}
}
