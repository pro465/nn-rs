//helper functions

use crate::Matrix;

pub(crate) fn transpose<'a>(mut v: impl Iterator<Item = &'a [f64]>, columns: usize) -> Matrix {
    if columns == 0 {
        return vec![];
    }

    let first = v.next().unwrap();

    let rows = first.len();
    let mut res = vec![Vec::with_capacity(columns); rows];

    let mut transpose_inner = |x: &'a [f64]| {
        for (i, &c) in x.iter().enumerate() {
            res[i].push(c);
        }
    };

    transpose_inner(first);

    for r in v {
        assert_eq!(r.len(), rows);

        transpose_inner(r);
    }

    res
}
