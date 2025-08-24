/// Muskingum-Cunge routing implementation matching Fortran NWM version
pub fn submuskingcunge(
    qup: f32,     // flow upstream previous timestep
    quc: f32,     // flow upstream current timestep
    qdp: f32,     // flow downstream previous timestep
    ql: f32,      // lateral inflow through reach (m^3/sec)
    dt: f32,      // routing period in seconds
    so: f32,      // channel bottom slope (as fraction, not %)
    dx: f32,      // channel length (m)
    n: f32,       // mannings coefficient
    cs: f32,      // channel side slope
    bw: f32,      // bottom width (meters)
    tw: f32,      // top width before bankfull (meters)
    tw_cc: f32,   // top width of compound (meters)
    n_cc: f32,    // mannings of compound
    depth_p: f32, // depth of flow in channel
) -> (f32, f32, f32, f32, f32, f32) {
    // Returns (qdc, velc, depthc, ck, cn, x)

    #[inline(always)]
    fn pow_2_3(x: f32) -> f32 {
        x.powf(2.0 / 3.0)
    }

    #[inline(always)]
    fn pow_5_3(x: f32) -> f32 {
        x * pow_2_3(x)
    }

    // Calculate hydraulic geometry
    fn hydraulic_geometry(
        h: f32,
        bfd: f32,
        bw: f32,
        tw_cc: f32,
        z: f32,
    ) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
        // Returns (twl, r, area, area_c, wp, wp_c, h_lt_bf, h_gt_bf)

        let twl = bw + 2.0 * z * h;

        let mut h_gt_bf = f32::max(h - bfd, 0.0);
        let mut h_lt_bf = f32::min(bfd, h);

        // Exception for NWM 3.0: if depth > bankfull but floodplain width is zero,
        // extend trapezoidal channel upwards
        if h_gt_bf > 0.0 && tw_cc <= 0.0 {
            h_gt_bf = 0.0;
            h_lt_bf = h;
        }

        let area = (bw + h_lt_bf * z) * h_lt_bf;
        let wp = bw + 2.0 * h_lt_bf * (1.0 + z * z).sqrt();
        let area_c = tw_cc * h_gt_bf;
        let wp_c = if h_gt_bf > 0.0 {
            tw_cc + 2.0 * h_gt_bf
        } else {
            0.0
        };

        let r = if (wp + wp_c) > 0.0 {
            (area + area_c) / (wp + wp_c)
        } else {
            0.0
        };

        (twl, r, area, area_c, wp, wp_c, h_lt_bf, h_gt_bf)
    }

    // Secant method helper function
    fn secant2_h(
        z: f32,
        bw: f32,
        bfd: f32,
        tw_cc: f32,
        so: f32,
        n: f32,
        n_cc: f32,
        dt: f32,
        dx: f32,
        qdp: f32,
        ql: f32,
        qup: f32,
        quc: f32,
        h: f32,
        interval: i32,
    ) -> (f32, f32, f32, f32, f32, f32) {
        // Returns (qj, c1, c2, c3, c4, x)

        let (twl, r, area, area_c, wp, wp_c, _, _) = hydraulic_geometry(h, bfd, bw, tw_cc, z);

        // Calculate kinematic celerity
        let ck = if h > bfd && tw_cc > 0.0 && n_cc > 0.0 {
            f32::max(
                0.0,
                ((so.sqrt() / n)
                    * ((5.0 / 3.0) * pow_2_3(r)
                        - (2.0 / 3.0)
                            * pow_5_3(r)
                            * (2.0 * (1.0 + z * z).sqrt() / (bw + 2.0 * bfd * z)))
                    * area
                    + (so.sqrt() / n_cc) * (5.0 / 3.0) * pow_2_3(h - bfd) * area_c)
                    / (area + area_c),
            )
        } else if h > 0.0 {
            f32::max(
                0.0,
                (so.sqrt() / n)
                    * ((5.0 / 3.0) * pow_2_3(r)
                        - (2.0 / 3.0)
                            * pow_5_3(r)
                            * (2.0 * (1.0 + z * z).sqrt() / (bw + 2.0 * h * z))),
            )
        } else {
            0.0
        };

        let km = if ck > 0.0 { f32::max(dt, dx / ck) } else { dt };

        // First calculate coefficients
        let d = km * (1.0 - 0.5) + dt / 2.0; // Use X=0.5 temporarily
        if d == 0.0 {
            panic!("FATAL ERROR: D is 0 in MUSKINGCUNGE");
        }

        // Calculate initial coefficients with X=0.5
        let mut c1 = (km * 0.5 + dt / 2.0) / d;
        let mut c2 = (dt / 2.0 - km * 0.5) / d;
        let mut c3 = (km * 0.5 - dt / 2.0) / d;
        let mut c4 = (ql * dt) / d;

        // Now calculate X based on interval and flow
        let x = if interval == 1 {
            // H0 interval - X calculation doesn't depend on coefficients
            0.0 // Will be recalculated below
        } else {
            // H interval - X depends on the flow sum using current coefficients
            let flow_sum = c1 * qup + c2 * quc + c3 * qdp + c4;
            if h > bfd && tw_cc > 0.0 && n_cc > 0.0 && ck > 0.0 {
                f32::min(
                    0.5,
                    f32::max(
                        0.25,
                        0.5 * (1.0 - (flow_sum / (2.0 * tw_cc * so * ck * dx))),
                    ),
                )
            } else if ck > 0.0 {
                f32::min(
                    0.5,
                    f32::max(0.25, 0.5 * (1.0 - (flow_sum / (2.0 * twl * so * ck * dx)))),
                )
            } else {
                0.5
            }
        };

        // For interval 1, calculate X differently (uses Qj which we haven't calculated yet)
        // So we need to iterate: calculate Qj with X=0, then recalculate X, then recalculate everything
        let x = if interval == 1 {
            // First pass: calculate Qj with current coefficients
            let qj_temp = if (wp + wp_c) > 0.0 {
                let manning_avg = ((wp * n) + (wp_c * n_cc)) / (wp + wp_c);
                (c1 * qup + c2 * quc + c3 * qdp + c4)
                    - ((1.0 / manning_avg) * (area + area_c) * pow_2_3(r) * so.sqrt())
            } else {
                0.0
            };

            // Now calculate X using qj_temp
            if h > bfd && tw_cc > 0.0 && n_cc > 0.0 && ck > 0.0 {
                f32::min(
                    0.5,
                    f32::max(0.0, 0.5 * (1.0 - (qj_temp / (2.0 * tw_cc * so * ck * dx)))),
                )
            } else if ck > 0.0 {
                f32::min(
                    0.5,
                    f32::max(0.0, 0.5 * (1.0 - (qj_temp / (2.0 * twl * so * ck * dx)))),
                )
            } else {
                0.5
            }
        } else {
            x
        };

        // Recalculate coefficients with correct X
        let d = km * (1.0 - x) + dt / 2.0;
        if d == 0.0 {
            panic!("FATAL ERROR: D is 0 in MUSKINGCUNGE");
        }

        c1 = (km * x + dt / 2.0) / d;
        c2 = (dt / 2.0 - km * x) / d;
        c3 = (km * (1.0 - x) - dt / 2.0) / d;
        c4 = (ql * dt) / d;

        // Check for negative flow in interval 2
        if interval == 2 {
            if c4 < 0.0 && c4.abs() > (c1 * qup + c2 * quc + c3 * qdp) {
                c4 = -(c1 * qup + c2 * quc + c3 * qdp);
            }
        }

        // Calculate Qj
        let qj = if (wp + wp_c) > 0.0 {
            let manning_avg = ((wp * n) + (wp_c * n_cc)) / (wp + wp_c);
            (c1 * qup + c2 * quc + c3 * qdp + c4)
                - ((1.0 / manning_avg) * (area + area_c) * pow_2_3(r) * so.sqrt())
        } else {
            0.0
        };

        (qj, c1, c2, c3, c4, x)
    }

    // Main function body
    let z = if cs == 0.0 { 1.0 } else { 1.0 / cs };

    let bfd = if bw > tw {
        bw / 0.00001
    } else if bw == tw {
        bw / (2.0 * z)
    } else {
        (tw - bw) / (2.0 * z)
    };

    if n <= 0.0 || so <= 0.0 || z <= 0.0 || bw <= 0.0 {
        panic!(
            "Error in channel coefficients -> Muskingum cunge: n={}, so={}, z={}, bw={}",
            n, so, z, bw
        );
    }

    let mut depth_c = f32::max(depth_p, 0.0);
    let mut h = (depth_c * 1.33) + 0.01;
    let mut h_0 = depth_c * 0.67;

    let qdc: f32;
    let velc: f32;
    let mut ck: f32 = 0.0;
    let cn: f32;
    let mut x: f32 = 0.0;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut c3: f32 = 0.0;
    let mut c4: f32 = 0.0;

    if ql > 0.0 || qup > 0.0 || quc > 0.0 || qdp > 0.0 {
        let mut tries = 0;
        let mut maxiter = 100;
        let mindepth = 0.01;

        'outer: loop {
            let mut iter = 0;
            let mut rerror = 1.0;
            let mut aerror = 0.01;

            while rerror > 0.01 && aerror >= mindepth && iter <= maxiter {
                // First call to secant2_h (interval 1, h_0)
                let (qj_0, _, _, _, _, _) = secant2_h(
                    z, bw, bfd, tw_cc, so, n, n_cc, dt, dx, qdp, ql, qup, quc, h_0, 1,
                );

                // Second call to secant2_h (interval 2, h) - this updates our main coefficients
                let (qj, c1_new, c2_new, c3_new, c4_new, x_new) = secant2_h(
                    z, bw, bfd, tw_cc, so, n, n_cc, dt, dx, qdp, ql, qup, quc, h, 2,
                );

                // Store the coefficients and X from interval 2
                c1 = c1_new;
                c2 = c2_new;
                c3 = c3_new;
                c4 = c4_new;
                x = x_new;

                // Update h using secant method
                let h_1 = if (qj_0 - qj) != 0.0 {
                    let h_new = h - (qj * (h_0 - h) / (qj_0 - qj));
                    if h_new < 0.0 { h } else { h_new }
                } else {
                    h
                };

                if h > 0.0 {
                    rerror = ((h_1 - h) / h).abs();
                    aerror = (h_1 - h).abs();
                } else {
                    rerror = 0.0;
                    aerror = 0.9;
                }

                h_0 = f32::max(0.0, h);
                h = f32::max(0.0, h_1);
                iter += 1;

                if h < mindepth {
                    break;
                }
            }

            if iter >= maxiter {
                tries += 1;
                if tries <= 4 {
                    h = h * 1.33;
                    h_0 = h_0 * 0.67;
                    maxiter += 25;
                    continue 'outer;
                }
                eprintln!("Musk Cunge WARNING: Failure to converge");
            }

            // Calculate final flow using the last coefficients from interval 2
            let flow_sum = c1 * qup + c2 * quc + c3 * qdp + c4;

            if flow_sum < 0.0 {
                if c4 < 0.0 && c4.abs() > (c1 * qup + c2 * quc + c3 * qdp) {
                    qdc = 0.0;
                } else {
                    qdc = f32::max(c1 * qup + c2 * quc + c4, c1 * qup + c3 * qdp + c4);
                }
            } else {
                qdc = flow_sum;
            }

            // Calculate velocity using simplified hydraulic radius (matching Fortran)
            let twl = bw + 2.0 * z * h;
            let r = (h * (bw + twl) / 2.0)
                / (bw + 2.0 * (((twl - bw) / 2.0).powi(2) + h.powi(2)).sqrt());
            velc = (1.0 / n) * pow_2_3(r) * so.sqrt();
            depth_c = h;

            break;
        }
    } else {
        qdc = 0.0;
        velc = 0.0;
        depth_c = 0.0;
    }

    // Calculate Courant number (matching Fortran courant subroutine)
    if depth_c > 0.0 {
        let (_, r, area, area_c, _, _, h_lt_bf, h_gt_bf) =
            hydraulic_geometry(depth_c, bfd, bw, tw_cc, z);

        ck = f32::max(
            0.0,
            ((so.sqrt() / n)
                * ((5.0 / 3.0) * pow_2_3(r)
                    - (2.0 / 3.0)
                        * pow_5_3(r)
                        * (2.0 * (1.0 + z * z).sqrt() / (bw + 2.0 * h_lt_bf * z)))
                * area
                + (so.sqrt() / n_cc) * (5.0 / 3.0) * pow_2_3(h_gt_bf) * area_c)
                / (area + area_c),
        );

        cn = ck * (dt / dx);
    } else {
        cn = 0.0;
    }

    (qdc, velc, depth_c, ck, cn, x)
}
