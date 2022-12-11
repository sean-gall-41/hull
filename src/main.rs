use std::ops::{Add, Sub, Mul};
use rand::Rng;
use std::f32;

#[macro_use]
extern crate approx;

extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::input::*;
use piston::event_loop::*;
use piston::window::WindowSettings;

use crate::piston::Window as pistonWindow;

pub struct App {
    hull: Hull, 
    gl: GlGraphics, // OpenGL drawing backend.
}

impl App {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        // set x_scale and y_scale as scaling factors
        let (x_scale, y_scale) = (args.window_size[0] * 0.62, args.window_size[1] * 0.62);

        // use piston2d-opengl draw convenience function
        self.gl.draw(args.viewport(), |c, gl| {
            clear([0.1; 4], gl);
            let transform = c
                .transform;

            // connecting lines: white
            for i in 0..self.hull.convex_hull.len() {
                line([1.0, 1.0, 1.0, 0.5], 1.0,
                     [x_scale * (self.hull.convex_hull[i].x as f64) + 0.31 * x_scale,
                      y_scale * (self.hull.convex_hull[i].y as f64) + 0.31 * y_scale,
                      x_scale * (self.hull.convex_hull[(i+1) % self.hull.convex_hull.len()].x as f64) + 0.31 * x_scale,
                      y_scale * (self.hull.convex_hull[(i+1) % self.hull.convex_hull.len()].y as f64) + 0.31 * y_scale],
                      transform,
                      gl);
            }

            // hull pts: green and larger than interior pts
            for coords in &self.hull.convex_hull {
                ellipse([0.0, 0.7, 0.3, 1.0],
                        [ x_scale * (coords.x  as f64) + 0.3 * x_scale,
                          y_scale * (coords.y as f64) + 0.3 * y_scale, 9.0, 9.0],
                        transform, gl); 
            }

            // interior pts: blue and smaller than hull pts
            for coords in &self.hull.interior {
                ellipse([0.0, 0.3, 0.7, 1.0],
                        [ x_scale * (coords.x  as f64) + 0.3 * x_scale,
                          y_scale * (coords.y as f64) + 0.3 * y_scale, 4.0, 4.0],
                        transform, gl); 
            }
        });
    }
}

// 2d vector, in the mathy sense
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
struct Vec2d {
    x: f32,
    y: f32,
}

impl Vec2d {
    // constructor: generate from an array slice
    fn from(in_arr: &[f32; 2]) -> Self {
        Self {
            x: in_arr[0],
            y: in_arr[1]
        }
    }

    // cross product: no need to worry about direction as
    // everything is defined in the plane
    fn cross(&self, other: &Self) -> f32 {
        self.x * other.y - self.y * other.x
    }
}

//add overload
impl Add for Vec2d {
    type Output = Self;

    fn add(self, other: Vec2d) -> Self {
        Vec2d {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

//subtraction overload
impl Sub for Vec2d {
    type Output = Self;

    fn sub(self, other: Vec2d) -> Self {
        Vec2d {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

// multiplication overload
impl Mul<f32> for Vec2d {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        Vec2d {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

// main object definition: consists
// of the number of total pts, the bounding
// box of the pts, the pts themselves,
// their convex hull and interior
#[derive(Debug)]
struct Hull {
    num_pts: usize,
    x_bounds: Vec<f32>,
    y_bounds: Vec<f32>,
    elems: Vec<Vec2d>,
    interior: Vec<Vec2d>,
    convex_hull: Vec<Vec2d>,
}

impl Hull {
    // construct a default hull
    fn new() -> Self {
        Hull {
            num_pts: 0usize,
            x_bounds: vec![0.0, 1.0],
            y_bounds: vec![0.0, 1.0],
            elems: Vec::<Vec2d>::new(),
            interior: Vec::<Vec2d>::new(),
            convex_hull: Vec::<Vec2d>::new(),
        }
    }
    
    // assuming we've created a hull already, generates the pts
    // randomly
    fn gen_rand_pts(&mut self) {
        assert!(self.num_pts > 0);
        // generates from random seed
        for _ in 0..self.num_pts {
            let x = rand::thread_rng().gen_range(self.x_bounds[0]..self.x_bounds[1]);
            let y = rand::thread_rng().gen_range(self.y_bounds[0]..self.y_bounds[1]);
            self.elems.push(Vec2d::from(&[x, y]));
        }
    }

    // construct a new hull from a given number of elements,
    // this includes calculating the convex hull and interior
    fn from_capacity(capacity: usize) -> Self {
        let mut out = Self::new();
        out.num_pts = capacity;
        out.gen_rand_pts();
        out.calc_convex_hull_graham_scan();
        out.calc_interior();
        out
    } 

    // clear all pts and regenerate them. keep
    // the same bounds and number of elements
    fn reset(&mut self) {
        self.elems.clear();
        self.interior.clear();
        self.convex_hull.clear();
        self.gen_rand_pts();
        self.calc_convex_hull_graham_scan();
        self.calc_interior();
    }

    // main algorithm of this module. use the graham scan
    // algorithm to compute the convex hull of a set of
    // pts in the plane
    fn calc_convex_hull_graham_scan(&mut self) {
        let mut temp_min = Vec2d::from(&[0.0, 1.0]);
        for elem in self.elems.iter() {
            if elem.y < temp_min.y {
                temp_min.x = elem.x;
                temp_min.y = elem.y;
            } 
        }

        let mut elems_cp = self.elems.clone();
        let index = self.elems.iter().position(|an_elem| relative_eq!(an_elem.x, temp_min.x, epsilon=f32::EPSILON) && relative_eq!(an_elem.y, temp_min.y, epsilon=f32::EPSILON)).unwrap();
        elems_cp.remove(index);
        let elems_cp_sorted = sort_rel_to(&self.elems[index], &elems_cp);
        self.convex_hull.push(self.elems[index]);
        self.convex_hull.push(elems_cp_sorted[0]);
        for elem in &elems_cp_sorted[1..] {
            // TODO: apply a collinear condition
            //if collinear(&self.convex_hull[self.convex_hull.len()-2],
            //              &self.convex_hull[self.convex_hull.len()-1],
            //              elem) {
            //    convex_null.push(*elem);
            //} else {
                while cw_turn(&self.convex_hull[self.convex_hull.len()-2],
                              &self.convex_hull[self.convex_hull.len()-1],
                              elem) {
                    self.convex_hull.pop();
                }
                self.convex_hull.push(*elem);
            //}
        }
    }

    // compute the interior by checking for elements that are not in the convex hull
    fn calc_interior(&mut self) {
        if self.convex_hull.is_empty() {
            panic!("cannot calculate interior when convex hull has not been computed.");
        }
        self.interior.clear();
        for coords in &self.elems {
            if !self.convex_hull.contains(coords) {
                self.interior.push(*coords);
            } 
        }
    }
}

// uses the cross product to determine whether a cw turn exists
// between 3 pts
fn cw_turn(vec1: &Vec2d, vec2: &Vec2d, vec3: &Vec2d) -> bool {
    let diff1 = *vec2 - *vec1;
    let diff2 = *vec3 - *vec2;
    let rot_diff = diff1.cross(&diff2);
    if rot_diff < 0.0 { true } else { false }
}

// sorts a vector of 2d vectors via an angle defined relative to
// a reference pt, assumed to have been pre-computed as the
// lowest-y pt, and the horizontal.
fn sort_rel_to(ref_pt: &Vec2d, elements: &Vec<Vec2d>) -> Vec<Vec2d> {
    let mut ordered_elems = Vec::<Vec2d>::new(); // return val
    let mut index_ang_vec = Vec::<(usize, f32)>::new();
    for (index, elem) in elements.iter().enumerate() {
        let ang = (elem.y-ref_pt.y).atan2(elem.x-ref_pt.x);
        index_ang_vec.push((index, ang));
    }
    index_ang_vec.sort_by(|a, b| a.1.total_cmp(&b.1));
    for elem in index_ang_vec {
        ordered_elems.push(elements[elem.0]);
    }
    ordered_elems
}

fn main() {
    let hull_num_pts = 30usize;
    let opengl = OpenGL::V3_2;

    // window initialization
    let mut window: Window = WindowSettings::new("Convex Hull", [800, 600])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .samples(8) // antialiasing 8x
        .build()
        .unwrap();

    let mut app = App {
        hull: Hull::from_capacity(hull_num_pts),
        gl: GlGraphics::new(opengl),
    };

    let mut events = Events::new(EventSettings::new());

    // event loop
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }
        
        // button control
        if let Some(Button::Keyboard(key)) = e.press_args() {
            match key {
                Key::R => app.hull.reset(), // reset the pts
                Key::Q => window.set_should_close(true), // close the window
                _ => ()
            }
        }
    }
}

