use std::fmt::Display;

use clap::ValueEnum;

use crate::kernel::muskingum::rs_route::*;

pub mod t_route;
pub mod rs_route {
    pub mod mc_kernel;
}
pub mod c_mc;

//     type, bind(c) :: muskingum_cunge_input_t
//         real(real32) :: dt      ! routing period in  seconds
//         real(real32) :: qup     ! flow upstream previous timestep
//         real(real32) :: quc     ! flow upstream current timestep
//         real(real32) :: qdp     ! flow downstream previous timestep
//         real(real32) :: ql      ! lateral inflow through reach (m^3/sec)
//         real(real32) :: dx      ! channel lngth (m)
//         real(real32) :: bw      ! bottom width (meters)
//         real(real32) :: tw      ! top width before bankfull (meters)
//         real(real32) :: tw_cc   ! top width of Compund (meters)
//         real(real32) :: n       ! mannings coefficient
//         real(real32) :: n_cc    ! mannings of compund
//         real(real32) :: cs      ! Channel side slope slope
//         real(real32) :: s0      ! Channel bottom slope %
//         real(real32) :: velp    ! Previous step velocity (unused)
//         real(real32) :: depthp  ! depth of flow in channel
//     end type

#[repr(C)]
#[derive(Default)]
pub struct MuskingumCungeInput {
    /// routing period in  seconds
    pub dt: f32,
    ///flow upstream previous timestep
    pub qup: f32,
    /// flow upstream current timestep
    pub quc: f32,
    /// flow downstream previous timestep
    pub qdp: f32,
    /// lateral inflow through reach (m^3/sec)
    pub ql: f32,
    /// channel lngth (m)
    pub dx: f32,
    /// bottom width (meters)
    pub bw: f32,
    /// top width before bankfull (meters)
    pub tw: f32,
    /// top width of Compund (meters)
    pub tw_cc: f32,
    /// mannings coefficient
    pub n: f32,
    /// mannings of compund
    pub n_cc: f32,
    /// Channel side slope slope
    pub cs: f32,
    /// Channel bottom slope %
    pub s0: f32,
    pub velp: f32,
    /// depth of flow in channel
    pub depthp: f32,
}

/// Universal MC Result struct
///
/// FORTRAN:
///     type, bind(c) :: muskingum_cunge_result_t
///         real(real32) :: qdc     ! flow downstream current timestep
///         real(real32) :: velc    ! channel velocity
///         real(real32) :: depthc  ! depth of flow in channel
///         real(real32) :: ck  
///         real(real32) :: cn
///         real(real32) :: X
///     end type
///
/// C_MC:
///     typedef struct QVD {
///         float qdc;
///         float velc;
///         float depthc;
///         float cn;
///         float ck;
///         float X;
///     } QVD_float;
///
#[repr(C)]
#[derive(Default)]
pub struct MuskingumCungeResult {
    /// flow downstream current timestep
    pub qdc: f32,
    /// channel velocity
    pub velc: f32,
    /// depth of flow in channel  
    pub depthc: f32,
    /// SHOULD BE; MUST CHECK Wave Celerity
    pub ck: f32,
    /// SHOULD BE; MUST CHECK Courant Number (ratio of Wave Celerity to spatio-temporal discretization in model)
    pub cn: f32,
    /// SHOULD BE; MUST CHECK (almost sure) Musk X -- the Musk X calculated dynamically within the compute kernel; almost always will be 0.5
    pub x: f32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum MuskingumCungeKernel {
    //#[value(name="route-rs-kern")]
    RouteRs,
    TRouteModernized,
    TRouteLegacy,
    CMuskingumCunge,
}
impl MuskingumCungeKernel {
    pub fn exec(
        self,
        qup: f32,
        quc: f32,
        qdp: f32,
        ql: f32,
        dt: f32,
        so: f32,
        dx: f32,
        n: f32,
        cs: f32,
        bw: f32,
        tw: f32,
        tw_cc: f32,
        n_cc: f32,
        depth_p: f32,
        calculate_courant: bool,
    ) -> MuskingumCungeResult {
        // println!("running kernel: {self}");
        match self {
            MuskingumCungeKernel::RouteRs => mc_kernel::submuskingcunge(
                qup,
                quc,
                qdp,
                ql,
                dt,
                so,
                dx,
                n,
                cs,
                bw,
                tw,
                tw_cc,
                n_cc,
                depth_p,
                calculate_courant,
            ),
            MuskingumCungeKernel::TRouteModernized => t_route::fortran_modernized::submuskingcunge(
                qup,
                quc,
                qdp,
                ql,
                dt,
                so,
                dx,
                n,
                cs,
                bw,
                tw,
                tw_cc,
                n_cc,
                depth_p,
                calculate_courant,
            ),
            MuskingumCungeKernel::TRouteLegacy => t_route::fortran_legacy::submuskingcunge(
                qup,
                quc,
                qdp,
                ql,
                dt,
                so,
                dx,
                n,
                cs,
                bw,
                tw,
                tw_cc,
                n_cc,
                depth_p,
                calculate_courant,
            ),
            MuskingumCungeKernel::CMuskingumCunge => c_mc::submuskingcunge(
                qup,
                quc,
                qdp,
                ql,
                dt,
                so,
                dx,
                n,
                cs,
                bw,
                tw,
                tw_cc,
                n_cc,
                depth_p,
                calculate_courant,
            ),
        }
    }
}
/*
    HERE BE DRAGONS

    These display values control the incredibly obscure case where `clap` displays the name of a ValueEnum variant as a default value
    if you do not do this, the user will see something like this:

    `-k, --kernel <KERNEL> [default: TRouteModernized] [possible values: route-rs-kernel, t-route-modernized, t-route-legacy, c-muskingum-cunge]`

    We do not like that. Because this will confuse the user. And we love the user.

    Brodie Alexander
    8 Oct 2025
*/
impl Display for MuskingumCungeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MuskingumCungeKernel::RouteRs => "route-rs",
                MuskingumCungeKernel::TRouteModernized => "t-route-modernized",
                MuskingumCungeKernel::TRouteLegacy => "t-route-legacy",
                MuskingumCungeKernel::CMuskingumCunge => "c-muskingum-cunge",
            }
        )
    }
}
