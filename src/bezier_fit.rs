use crate::math::{add2, dist2, dot2, normalize2, scale2, sub2};

/// 点群を3次ベジェ曲線列にフィッティングする（Schneider 1990）。
/// 戻り値は制御点の平坦なリスト: [P0,P1,P2,P3, P0,P1,P2,P3, ...]
pub fn fit_curve(points: &[[f32; 2]], error: f32) -> Vec<[f32; 2]> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let t_hat1 = compute_left_tangent(points, 0);
    let t_hat2 = compute_right_tangent(points, points.len() - 1);
    let mut result = Vec::new();
    fit_cubic(
        points,
        0,
        points.len() - 1,
        t_hat1,
        t_hat2,
        error,
        &mut result,
    );
    result
}

/// 部分点列 [first..=last] を再帰的にベジェフィッティングする。
/// 誤差が error 以内なら採用、超えていれば最大誤差点で分割して再帰。
fn fit_cubic(
    points: &[[f32; 2]],
    first: usize,
    last: usize,
    t_hat1: [f32; 2],
    t_hat2: [f32; 2],
    error: f32,
    result: &mut Vec<[f32; 2]>,
) {
    let n_pts = last - first + 1;

    // 2点だけの場合は端点接線から制御点をヒューリスティックに決める
    if n_pts == 2 {
        let dist = dist2(points[first], points[last]) / 3.0;
        let bezier = [
            points[first],
            add2(points[first], scale2(t_hat1, dist)),
            add2(points[last], scale2(t_hat2, dist)),
            points[last],
        ];
        result.extend_from_slice(&bezier);
        return;
    }

    // コード長でパラメータ化 → 最小二乗法でベジェ生成 → 誤差計算
    let u = chord_length_parameterize(&points[first..=last]);
    let mut bezier = generate_bezier(points, first, last, &u, t_hat1, t_hat2);
    let (mut max_error, mut split_point) = compute_max_error(points, first, last, &bezier, &u);

    // 誤差が許容範囲内なら採用
    if max_error < error {
        result.extend_from_slice(&bezier);
        return;
    }

    // 誤差が中程度なら再パラメータ化を最大4回試みて精度改善
    let iteration_error = error * 4.0;
    if max_error < iteration_error {
        for _ in 0..4 {
            let u_prime = reparameterize(points, first, last, &u, &bezier);
            bezier = generate_bezier(points, first, last, &u_prime, t_hat1, t_hat2);
            let (err, sp) = compute_max_error(points, first, last, &bezier, &u_prime);
            max_error = err;
            split_point = sp;
            if max_error < error {
                result.extend_from_slice(&bezier);
                return;
            }
        }
    }

    // それでも誤差が大きければ最大誤差点で分割して再帰
    let t_hat_center = compute_center_tangent(points, split_point);
    fit_cubic(
        points,
        first,
        split_point,
        t_hat1,
        t_hat_center,
        error,
        result,
    );
    let t_hat_center_neg = scale2(t_hat_center, -1.0);
    fit_cubic(
        points,
        split_point,
        last,
        t_hat_center_neg,
        t_hat2,
        error,
        result,
    );
}

/// 始点側の接線ベクトル（始点→次点方向）
fn compute_left_tangent(points: &[[f32; 2]], end: usize) -> [f32; 2] {
    normalize2(sub2(points[end + 1], points[end]))
}

/// 終点側の接線ベクトル（終点→前点方向）
fn compute_right_tangent(points: &[[f32; 2]], end: usize) -> [f32; 2] {
    normalize2(sub2(points[end - 1], points[end]))
}

/// 分割点での接線ベクトル（前後の差分の平均）
fn compute_center_tangent(points: &[[f32; 2]], center: usize) -> [f32; 2] {
    let v1 = sub2(points[center - 1], points[center]);
    let v2 = sub2(points[center], points[center + 1]);
    normalize2(add2(v1, v2))
}

/// 点列をコード長比率でt=0~1にパラメータ化する
fn chord_length_parameterize(points: &[[f32; 2]]) -> Vec<f32> {
    let mut u = vec![0.0f32; points.len()];
    for i in 1..points.len() {
        let dx = points[i][0] - points[i - 1][0];
        let dy = points[i][1] - points[i - 1][1];
        u[i] = u[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let total = u[points.len() - 1];
    for i in 1..points.len() {
        u[i] /= total;
    }
    u
}

/// 最小二乗法で3次ベジェの内側2制御点を求める
fn generate_bezier(
    points: &[[f32; 2]],
    first: usize,
    last: usize,
    u: &[f32],
    t_hat1: [f32; 2],
    t_hat2: [f32; 2],
) -> [[f32; 2]; 4] {
    let n = last - first + 1;

    // 各点でのベジェ基底関数による接線方向の寄与を計算
    let mut a = vec![[[0.0f32; 2]; 2]; n];
    for i in 0..n {
        a[i][0] = scale2(t_hat1, b1(u[i]));
        a[i][1] = scale2(t_hat2, b2(u[i]));
    }

    // 連立方程式の係数行列 C と右辺ベクトル X を構築
    let mut c = [[0.0f32; 2]; 2];
    let mut x = [0.0f32; 2];

    for i in 0..n {
        c[0][0] += dot2(a[i][0], a[i][0]);
        c[0][1] += dot2(a[i][0], a[i][1]);
        c[1][0] = c[0][1];
        c[1][1] += dot2(a[i][1], a[i][1]);

        let p = points[first + i];
        let tmp = sub2(
            p,
            add2(
                scale2(points[first], b0(u[i])),
                add2(
                    scale2(points[first], b1(u[i])),
                    add2(
                        scale2(points[last], b2(u[i])),
                        scale2(points[last], b3(u[i])),
                    ),
                ),
            ),
        );

        x[0] += dot2(a[i][0], tmp);
        x[1] += dot2(a[i][1], tmp);
    }

    // クラメールの公式で制御点の伸び量 alpha を求める
    let det_c0_c1 = c[0][0] * c[1][1] - c[1][0] * c[0][1];
    let det_c0_x = c[0][0] * x[1] - c[1][0] * x[0];
    let det_x_c1 = x[0] * c[1][1] - x[1] * c[0][1];

    let alpha_l = if det_c0_c1 == 0.0 {
        0.0
    } else {
        det_x_c1 / det_c0_c1
    };
    let alpha_r = if det_c0_c1 == 0.0 {
        0.0
    } else {
        det_c0_x / det_c0_c1
    };

    // alpha が小さすぎる場合は端点間距離の1/3をフォールバックとして使う
    let seg_len = dist2(points[first], points[last]);
    let epsilon = 1.0e-6 * seg_len;
    if alpha_l < epsilon || alpha_r < epsilon {
        let dist = seg_len / 3.0;
        return [
            points[first],
            add2(points[first], scale2(t_hat1, dist)),
            add2(points[last], scale2(t_hat2, dist)),
            points[last],
        ];
    }

    [
        points[first],
        add2(points[first], scale2(t_hat1, alpha_l)),
        add2(points[last], scale2(t_hat2, alpha_r)),
        points[last],
    ]
}

/// ベジェ曲線と元の点群の最大二乗誤差と分割点インデックスを返す
fn compute_max_error(
    points: &[[f32; 2]],
    first: usize,
    last: usize,
    bezier: &[[f32; 2]; 4],
    u: &[f32],
) -> (f32, usize) {
    let mut max_dist = 0.0f32;
    let mut split_point = (last - first + 1) / 2 + first;

    for i in first + 1..last {
        let p = bezier_eval3(bezier, u[i - first]);
        let dx = p[0] - points[i][0];
        let dy = p[1] - points[i][1];
        let dist = dx * dx + dy * dy;
        if dist >= max_dist {
            max_dist = dist;
            split_point = i;
        }
    }
    (max_dist, split_point)
}

/// 現在のパラメータ値を Newton-Raphson 法でより良い値に改善する
fn reparameterize(
    points: &[[f32; 2]],
    first: usize,
    last: usize,
    u: &[f32],
    bezier: &[[f32; 2]; 4],
) -> Vec<f32> {
    (first..=last)
        .map(|i| newton_raphson_root_find(bezier, points[i], u[i - first]))
        .collect()
}

/// Newton-Raphson 法で点 p に最も近いベジェ上のパラメータ値を求める
fn newton_raphson_root_find(bezier: &[[f32; 2]; 4], p: [f32; 2], u: f32) -> f32 {
    // Q'（1階微分）と Q''（2階微分）の制御点を生成
    let q1 = [
        scale2(sub2(bezier[1], bezier[0]), 3.0),
        scale2(sub2(bezier[2], bezier[1]), 3.0),
        scale2(sub2(bezier[3], bezier[2]), 3.0),
    ];
    let q2 = [
        scale2(sub2(q1[1], q1[0]), 2.0),
        scale2(sub2(q1[2], q1[1]), 2.0),
    ];

    let q_u = bezier_eval3(bezier, u);
    let q1_u = bezier_eval2(&q1, u);
    let q2_u = bezier_eval1(&q2, u);

    let numerator = dot2(sub2(q_u, p), q1_u);
    let denominator = dot2(q1_u, q1_u) + dot2(sub2(q_u, p), q2_u);

    if denominator == 0.0 {
        return u;
    }
    u - numerator / denominator
}

/// 3次ベジェ曲線をパラメータ t で評価
fn bezier_eval3(bezier: &[[f32; 2]; 4], t: f32) -> [f32; 2] {
    add2(
        scale2(bezier[0], b0(t)),
        add2(
            scale2(bezier[1], b1(t)),
            add2(scale2(bezier[2], b2(t)), scale2(bezier[3], b3(t))),
        ),
    )
}

/// 2次ベジェ曲線をパラメータ t で評価
fn bezier_eval2(pts: &[[f32; 2]; 3], t: f32) -> [f32; 2] {
    add2(
        scale2(pts[0], b0_2(t)),
        add2(scale2(pts[1], b1_2(t)), scale2(pts[2], b2_2(t))),
    )
}

/// 1次ベジェ（線形補間）をパラメータ t で評価
fn bezier_eval1(pts: &[[f32; 2]; 2], t: f32) -> [f32; 2] {
    add2(scale2(pts[0], 1.0 - t), scale2(pts[1], t))
}

// 3次ベジェ基底関数
fn b0(t: f32) -> f32 {
    let s = 1.0 - t;
    s * s * s
}
fn b1(t: f32) -> f32 {
    let s = 1.0 - t;
    3.0 * t * s * s
}
fn b2(t: f32) -> f32 {
    let s = 1.0 - t;
    3.0 * t * t * s
}
fn b3(t: f32) -> f32 {
    t * t * t
}

// 2次ベジェ基底関数
fn b0_2(t: f32) -> f32 {
    (1.0 - t) * (1.0 - t)
}
fn b1_2(t: f32) -> f32 {
    2.0 * t * (1.0 - t)
}
fn b2_2(t: f32) -> f32 {
    t * t
}
