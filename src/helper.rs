//helper functions

use crate::Matrix;

pub(crate) fn transpose(v: &[&'_ [f64]]) -> Matrix {
    let columns = v.len();
    let rows = v[0].len();
    let mut res = vec![Vec::with_capacity(columns); rows];

    for r in v {
        assert_eq!(r.len(), rows);

        for (i, &c) in r.iter().enumerate() {
            res[i].push(c);
        }
    }

    res
}

pub(crate) fn unwrap(
    iter: impl Iterator<Item = (Matrix, Matrix, Vec<f64>)>,
) -> (Vec<Matrix>, Vec<Matrix>, Matrix) {
    let mut res = (Vec::new(), Vec::new(), Matrix::new());

    iter.for_each(|(a, o, y)| {
        res.0.push(a);
        res.1.push(o);
        res.2.push(y);
    });

    res
}
