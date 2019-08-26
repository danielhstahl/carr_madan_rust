//! Carr-Madan approach for an option using the underlying's
//! characteristic function. Some useful characteristic functions
//! are provided in the [cf_functions](https://crates.io/crates/cf_functions) repository.
//! Carr and Madan's approach works well for computing a large number of equidistant
//! strikes. [Link to Carr-Madan paper](https://wwwf.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/CarrMadan.pdf).
//!
/*extern crate rustfft;
extern crate num;
extern crate black_scholes;
extern crate rayon;
#[macro_use]
#[cfg(test)]
extern crate approx;*/

use num::traits::{Zero};
use rustfft::algorithm::Radix4;
use rustfft::FFT;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use rayon::prelude::*;

/**
@ada the distance between nodes in the U domain
@returns the maximum value of the U domain
*/
fn get_max_k(ada: f64)->f64{
    PI/ada
}
/**
@numSteps the number of steps used to discretize the x and u domains
@returns the distance between nodes in the X domain
*/
fn get_lambda(num_steps:usize, b:f64)->f64{
    2.0*b/(num_steps as f64)
}

fn get_x(xmin:f64, dx:f64, index:usize)->f64{
    xmin+dx*(index as f64)
}

fn get_k_at_index(b:f64, lambda:f64, s0:f64, index:usize)->f64{
    s0*get_x(-b, lambda, index).exp()
}

/// Returns iterator which produces strikes
///
/// # Examples
/// ```
/// let eta = 0.04;
/// let s0 = 50.0;
/// let num_x = 256;
/// let strikes = carr_madan::get_strikes(eta, s0, num_x);
/// ```
pub fn get_strikes(
    eta:f64, s0:f64, num_x:usize
)->impl IndexedParallelIterator<Item = f64>{
    let b=get_max_k(eta);
    let lambda=get_lambda(num_x, b);
    (0..num_x).into_par_iter().map(move |index|get_k_at_index(b, lambda, s0, index))
}


fn call_aug<T>(v:&Complex<f64>, alpha:f64, cf:T)->Complex<f64>
    where T: Fn(&Complex<f64>)->Complex<f64>
{ //used for Carr-Madan approach...v is typically complex
    let aug_u=v+(alpha+1.0);
    cf(&aug_u)/(alpha*alpha+alpha+v*v+(2.0*alpha+1.0)*v)
}

fn carr_madan_g<T, S>(num_steps:usize, eta:f64, alpha:f64, s0:f64, discount:f64, m_out:S, aug_cf:T)->Vec<(f64, f64)>
    where T: Fn(&Complex<f64>, f64)->Complex<f64>+std::marker::Sync, S:Fn(f64, usize)->f64+std::marker::Sync
{
    let b=get_max_k(eta);
    let lambda=get_lambda(num_steps, b);
    let fft = Radix4::new(num_steps, false);
    let mut cmpl:  Vec<Complex<f64> > =(0..num_steps).into_par_iter().map(|index| {
        let pm=if index%2==0 {-1.0} else {1.0};
        let u=Complex::<f64>::new(0.0, (index as f64)*eta);
        let aug_u=Complex::<f64>::new(0.0, b*(index as f64)*eta);
        let f_answer=discount*aug_cf(&u, alpha)*aug_u.exp()*(3.0+pm);
        if index==0 {f_answer*0.5} else {f_answer}
    }).collect();
    let mut output:  Vec<Complex<f64> >=vec![Complex::zero(); num_steps];//why do I have to initialize this?
    fft.process(&mut cmpl, &mut output);
    get_strikes(eta, s0, num_steps)
        .zip(output.par_iter()
            .enumerate()
            .map(|(index, &x)| m_out(s0*x.re*(-alpha*get_x(-b, lambda, index)).exp()*eta/(PI*3.0), index))
        ).collect()
}
/// Returns call prices over a series of strikes
///
/// # Examples
/// ```
/// extern crate rustfft;
/// use rustfft::num_complex::Complex;
/// extern crate carr_madan;
/// # fn main() {
/// let alpha = 1.5;
/// let eta = 0.25;
/// let num_steps = 256;
/// let s0 = 50.0;
/// let r:f64 = 0.05;
/// let t:f64 = 1.5;
/// let sig = 0.4;
/// let discount = (-r*t).exp();
/// let bscf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
/// let call_prices = carr_madan::call(
///     num_steps, eta, alpha, s0,
///     discount, &bscf
/// ); //first element is strike, second is call price
/// # }
/// ```
pub fn call_price<T>(
    num_steps:usize, eta:f64, alpha:f64,
    s0:f64, discount:f64,
    cf:&T
)->Vec<(f64, f64)>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync
{
    carr_madan_g(num_steps, eta, alpha, s0, discount,
        |x, _| x,
        |&x, alpha| call_aug(&x, alpha, &cf)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    #[test]
    fn get_x_returns_correctly() {
        let x=get_x(-5.0, 1.0, 3);
        assert_eq!(x, -2.0);
    }
    #[test]
    fn get_strikes_returns_correctly() {
        let strikes=get_strikes(1.0, 1.0, 5);
        assert_eq!(strikes.len(), 5);
    }
    #[test]
    fn carr_madan_call_returns_correctly() {
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let s0=50.0;
        let discount=(-r*t as f64).exp();
        let bscf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let num_x=(2 as usize).pow(10);
        let eta=0.25;
        let alpha=1.5;
        let my_options_price=call_price(
            num_x,
            eta,
            alpha,
            s0,
            discount,
            &bscf
        );
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            let (strike, price)=my_options_price[i];
            assert_abs_diff_eq!(
                black_scholes::call(s0, strike, r, t, sig),
                price,
                epsilon=0.001

            );
        }
    }
}
