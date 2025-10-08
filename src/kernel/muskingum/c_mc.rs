use crate::kernel::muskingum::MuskingumCungeResult;

#[repr(C)]
    #[derive(Default)]
    pub struct MuskingumCungeInput {
        /// routing period in  seconds
        dt: f32,
        ///flow upstream previous timestep
        qup: f32,
        /// flow upstream current timestep
        quc: f32,
        /// flow downstream previous timestep
        qdp: f32,
        /// lateral inflow through reach (m^3/sec)
        ql: f32,
        /// channel lngth (m)
        dx: f32,
        /// bottom width (meters)
        bw: f32,
        /// top width before bankfull (meters)
        tw: f32,
        /// top width of Compund (meters)
        tw_cc: f32,
        /// mannings coefficient
        n: f32,
        /// mannings of compund
        n_cc: f32,
        /// Channel side slope slope
        cs: f32,
        /// Channel bottom slope %
        s0: f32,
        /// Velocity at previous timestep (currently unused)
        velp: f32,
        /// depth of flow in channel
        depthp: f32,
    }

    unsafe extern "C" {
        pub fn c_binding_c_mc_muskingum_cunge(
            input: *const MuskingumCungeInput,
            result: *mut MuskingumCungeResult,
        );
    }

    pub fn submuskingcunge(
        qup: f32,                // flow upstream previous timestep
        quc: f32,                // flow upstream current timestep
        qdp: f32,                // flow downstream previous timestep
        ql: f32,                 // lateral inflow through reach (m^3/sec)
        dt: f32,                 // routing period in seconds
        so: f32,                 // channel bottom slope (as fraction, not %)
        dx: f32,                 // channel length (m)
        n: f32,                  // mannings coefficient
        cs: f32,                 // channel side slope
        bw: f32,                 // bottom width (meters)
        tw: f32,                 // top width before bankfull (meters)
        tw_cc: f32,              // top width of compound (meters)
        n_cc: f32,               // mannings of compound
        depth_p: f32,            // depth of flow in channel
        _calculate_courant: bool, // whether to calculate courant number
    ) -> MuskingumCungeResult {

        // double dt,
        // double qup,
        // double quc,
        // double qdp,
        // double ql,
        // double dx,
        // double bw,
        // double tw,
        // double twcc,
        // double n,
        // double ncc,
        // double cs,
        // double s0,
        // double velp,
        // double depthp,
        let input = MuskingumCungeInput {
            dt, qup, quc, qdp, ql, dx, bw, tw, tw_cc, n, n_cc, cs, s0: so, velp: 0.0, depthp: depth_p
        };
        let mut result = MuskingumCungeResult::default();
        unsafe {
            c_binding_c_mc_muskingum_cunge(&input, &mut result);
        }
        result
    }