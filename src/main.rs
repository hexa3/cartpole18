//! Interactive viewer: an 18-link inverted pendulum balanced on a cart by a
//! single linear motor under LQR control.
//!
//! Controls:
//! ```text
//!   ← / →      push the selected link left / right (50 ms pulse)
//!   ↑ / ↓      select link (or click a joint)
//!   + / -      push strength
//!   G          toggle random gusts
//!   V          bend exaggeration (×1 / ×5 / ×20) — drawing only
//!   S          slow motion
//!   Space      pause
//!   R          reset (chain leaning 1°)
//! ```

use cartpole18::sim::{Sim, CTRL_DT, PUSH_DURATION};
use macroquad::prelude::*;
use std::collections::VecDeque;

const BG: Color = Color::new(0.07, 0.08, 0.10, 1.0);
const RAIL: Color = Color::new(0.35, 0.37, 0.42, 1.0);
const LINK: Color = Color::new(0.85, 0.87, 0.92, 1.0);
const JOINT: Color = Color::new(0.30, 0.65, 1.00, 1.0);
const SELECTED: Color = Color::new(1.00, 0.75, 0.20, 1.0);
const PUSH: Color = Color::new(1.00, 0.35, 0.30, 1.0);
const MOTOR: Color = Color::new(0.35, 0.85, 0.55, 1.0);
const TEXT: Color = Color::new(0.80, 0.82, 0.86, 1.0);
const DIM: Color = Color::new(0.50, 0.52, 0.56, 1.0);

const HISTORY_SECS: f64 = 8.0;
const HISTORY_DT: f64 = 0.01;
const GUST_PERIOD: f64 = 3.0;
const START_LEAN_DEG: f64 = 1.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "18-link cart pendulum".to_owned(),
        window_width: 1280,
        window_height: 900,
        high_dpi: true,
        ..Default::default()
    }
}

struct App {
    sim: Sim,
    selected: usize,
    push_strength: f64,
    gusts: bool,
    next_gust: f64,
    exaggeration: f64,
    slow_mo: bool,
    paused: bool,
    accumulator: f64,
    view_x: f64,
    /// (motor force, max |θ|) samples for the strip charts
    history: VecDeque<(f64, f64)>,
    next_sample: f64,
    peak_force: f64,
}

impl App {
    fn new() -> Self {
        let mut sim = Sim::new();
        sim.reset(START_LEAN_DEG.to_radians());
        let selected = sim.p.n_links - 1;
        Self {
            sim,
            selected,
            push_strength: 1e-6,
            gusts: false,
            next_gust: GUST_PERIOD,
            exaggeration: 1.0,
            slow_mo: false,
            paused: false,
            accumulator: 0.0,
            view_x: 0.0,
            history: VecDeque::new(),
            next_sample: 0.0,
            peak_force: 0.0,
        }
    }

    fn reset(&mut self) {
        self.sim.reset(START_LEAN_DEG.to_radians());
        self.history.clear();
        self.next_sample = 0.0;
        self.next_gust = GUST_PERIOD;
        self.peak_force = 0.0;
        self.accumulator = 0.0;
    }

    fn handle_input(&mut self, joints: &[Vec2]) {
        let n = self.sim.p.n_links;
        if is_key_pressed(KeyCode::R) {
            self.reset();
        }
        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if is_key_pressed(KeyCode::S) {
            self.slow_mo = !self.slow_mo;
        }
        if is_key_pressed(KeyCode::G) {
            self.gusts = !self.gusts;
            self.next_gust = self.sim.t + 0.5;
        }
        if is_key_pressed(KeyCode::V) {
            self.exaggeration = match self.exaggeration as u32 {
                1 => 5.0,
                5 => 20.0,
                _ => 1.0,
            };
        }
        if is_key_pressed(KeyCode::Up) {
            self.selected = (self.selected + 1).min(n - 1);
        }
        if is_key_pressed(KeyCode::Down) {
            self.selected = self.selected.saturating_sub(1);
        }
        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
            self.push_strength = (self.push_strength * 1.5).min(1e-3);
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) {
            self.push_strength = (self.push_strength / 1.5).max(1e-9);
        }
        if is_mouse_button_pressed(MouseButton::Left) {
            let m = Vec2::from(mouse_position());
            if let Some((i, d)) = joints
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, j)| (i - 1, j.distance(m)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                && d < 40.0
            {
                self.selected = i;
            }
        }
        if !self.sim.fallen() {
            if is_key_pressed(KeyCode::Left) {
                self.sim.apply_push(self.selected, -self.push_strength);
            }
            if is_key_pressed(KeyCode::Right) {
                self.sim.apply_push(self.selected, self.push_strength);
            }
        }
    }

    fn advance(&mut self, frame_dt: f64) {
        if self.paused || self.sim.fallen() {
            return;
        }
        let speed = if self.slow_mo { 0.2 } else { 1.0 };
        self.accumulator += frame_dt.min(0.05) * speed;
        while self.accumulator >= CTRL_DT {
            self.accumulator -= CTRL_DT;
            if self.gusts && self.sim.t >= self.next_gust && self.sim.push.is_none() {
                let link = rand::gen_range(0, self.sim.p.n_links);
                let sign = if rand::gen_range(0, 2) == 0 { -1.0 } else { 1.0 };
                let mag = self.push_strength * rand::gen_range(0.4, 1.0);
                self.sim.apply_push(link, sign * mag);
                self.next_gust = self.sim.t + GUST_PERIOD;
            }
            self.sim.step();
            self.peak_force = self.peak_force.max(self.sim.u.abs());
            if self.sim.t >= self.next_sample {
                self.next_sample += HISTORY_DT;
                self.history.push_back((self.sim.u, self.sim.max_abs_angle()));
                if self.history.len() > (HISTORY_SECS / HISTORY_DT) as usize {
                    self.history.pop_front();
                }
            }
            if self.sim.fallen() {
                break;
            }
        }
    }

    /// Screen positions of the cart pivot and every joint, plus pixels per metre.
    fn layout(&mut self) -> (Vec<Vec2>, f32, f32) {
        let p = &self.sim.p;
        let total_len: f64 = p.link_len.iter().sum();
        let ground = screen_height() - 110.0;
        let ppm = ((ground - 60.0) as f64 / (total_len * 1.03)) as f32;
        // camera follows the cart only once it wanders off-centre
        let cart_x = self.sim.s[0];
        let half_view = (screen_width() * 0.35 / ppm) as f64;
        if (cart_x - self.view_x).abs() > half_view {
            self.view_x += (cart_x - self.view_x) - half_view * (cart_x - self.view_x).signum();
        }
        let origin_x = screen_width() * 0.5;
        let to_screen = |x: f64, y: f64| {
            vec2(origin_x + ((x - self.view_x) as f32) * ppm, ground - 18.0 - (y as f32) * ppm)
        };
        let mut pts = vec![to_screen(cart_x, 0.0)];
        let (mut x, mut y) = (cart_x, 0.0);
        for i in 0..p.n_links {
            let th = self.sim.s[1 + i] * self.exaggeration;
            x += p.link_len[i] * th.sin();
            y += p.link_len[i] * th.cos();
            pts.push(to_screen(x, y));
        }
        (pts, ppm, ground)
    }

    fn draw(&self, joints: &[Vec2], ppm: f32, ground: f32) {
        clear_background(BG);
        let p = &self.sim.p;

        // rail with metre ticks
        draw_line(0.0, ground, screen_width(), ground, 3.0, RAIL);
        let origin_x = screen_width() * 0.5;
        let first = (self.view_x - (origin_x / ppm) as f64).floor() as i64;
        let last = (self.view_x + (origin_x / ppm) as f64).ceil() as i64;
        for m in first..=last {
            for k in 0..10 {
                let x = m as f64 + k as f64 * 0.1;
                let sx = origin_x + ((x - self.view_x) as f32) * ppm;
                let h = if k == 0 { 14.0 } else { 6.0 };
                draw_line(sx, ground, sx, ground + h, 1.0, RAIL);
                if k == 0 {
                    draw_text(format!("{m} m"), sx + 3.0, ground + 28.0, 18.0, DIM);
                }
            }
        }

        // cart and motor force arrow
        let cart = joints[0];
        draw_rectangle(cart.x - 45.0, cart.y - 4.0, 90.0, 26.0, Color::new(0.22, 0.25, 0.31, 1.0));
        draw_rectangle_lines(cart.x - 45.0, cart.y - 4.0, 90.0, 26.0, 2.0, RAIL);
        let arrow_len = (self.sim.u / self.sim.max_force) as f32 * 160.0;
        if arrow_len.abs() > 1.0 {
            draw_arrow(vec2(cart.x, cart.y + 9.0), vec2(cart.x + arrow_len, cart.y + 9.0), MOTOR, 4.0);
        }

        // links, thickness growing with link mass
        let m_max = p.link_mass.iter().cloned().fold(0.0, f64::max);
        for i in 0..p.n_links {
            let w = 2.0 + 5.0 * (p.link_mass[i] / m_max) as f32;
            draw_line(joints[i].x, joints[i].y, joints[i + 1].x, joints[i + 1].y, w, LINK);
        }
        draw_circle(cart.x, cart.y, 5.0, JOINT);
        for i in 0..p.n_links {
            let j = joints[i + 1];
            let r = 3.0 + 5.0 * (p.link_mass[i] / m_max).cbrt() as f32;
            let c = if i == self.selected { SELECTED } else { JOINT };
            draw_circle(j.x, j.y, r, c);
        }
        let sel = joints[self.selected + 1];
        draw_circle_lines(sel.x, sel.y, 14.0, 2.0, SELECTED);

        // active push
        if let Some((push, _)) = self.sim.push {
            let j = joints[push.link + 1];
            let dir = push.force.signum() as f32;
            draw_arrow(vec2(j.x - dir * 90.0, j.y), vec2(j.x - dir * 14.0, j.y), PUSH, 4.0);
        }

        self.draw_hud();
    }

    fn draw_hud(&self) {
        let sim = &self.sim;
        let mut y = 30.0;
        let mut line = |s: &str, c: Color| {
            draw_text(s, 20.0, y, 22.0, c);
            y += 24.0;
        };
        let status = if sim.fallen() {
            ("FALLEN - press R to reset", PUSH)
        } else if self.paused {
            ("PAUSED", SELECTED)
        } else {
            ("BALANCING", MOTOR)
        };
        line(status.0, status.1);
        line(&format!("t = {:.2} s{}", sim.t, if self.slow_mo { "   (slow motion x0.2)" } else { "" }), TEXT);
        line(&format!("cart x = {:+.3} m   v = {:+.3} m/s", sim.s[0], sim.s[sim.p.dof()]), TEXT);
        line(&format!("motor force = {:+7.2} N   (limit {:.0} N, peak {:.1} N)", sim.u, sim.max_force, self.peak_force), TEXT);
        line(&format!("max link tilt = {:.3} deg", sim.max_abs_angle().to_degrees()), TEXT);
        line(
            &format!(
                "selected link {} ({:.0} cm, {:.2} kg)   push = {} for {:.0} ms",
                self.selected + 1,
                sim.p.link_len[self.selected] * 100.0,
                sim.p.link_mass[self.selected],
                fmt_force(self.push_strength),
                PUSH_DURATION * 1000.0
            ),
            SELECTED,
        );
        if self.gusts {
            line("random gusts ON", PUSH);
        }
        if self.exaggeration > 1.0 {
            line(&format!("drawing bends exaggerated x{}", self.exaggeration), SELECTED);
        }

        let help = [
            "<- / ->  push selected link",
            "up / down or click  select link",
            "+ / -  push strength",
            "G  random gusts    V  exaggerate bends",
            "S  slow motion    Space  pause    R  reset",
        ];
        for (k, h) in help.iter().enumerate() {
            let w = measure_text(h, None, 18, 1.0).width;
            draw_text(h, screen_width() - w - 20.0, 26.0 + 22.0 * k as f32, 18.0, DIM);
        }

        // strip charts
        let w = 320.0;
        let h = 70.0;
        let x0 = screen_width() - w - 20.0;
        let y_force = 150.0;
        let y_angle = y_force + h + 30.0;
        let max_angle = self.history.iter().map(|s| s.1).fold(1e-4, f64::max);
        strip_chart(
            &self.history.iter().map(|s| s.0 / sim.max_force).collect::<Vec<_>>(),
            x0, y_force, w, h, true, MOTOR,
            &format!("motor force (±{:.0} N)", sim.max_force),
        );
        strip_chart(
            &self.history.iter().map(|s| s.1 / max_angle).collect::<Vec<_>>(),
            x0, y_angle, w, h, false, SELECTED,
            &format!("max link tilt (0 - {:.2} deg)", max_angle.to_degrees()),
        );
    }
}

fn fmt_force(f: f64) -> String {
    if f >= 1e-3 {
        format!("{:.2} mN", f * 1e3)
    } else if f >= 1e-6 {
        format!("{:.2} uN", f * 1e6)
    } else {
        format!("{:.2} nN", f * 1e9)
    }
}

fn draw_arrow(from: Vec2, to: Vec2, color: Color, thickness: f32) {
    draw_line(from.x, from.y, to.x, to.y, thickness, color);
    let dir = (to - from).normalize_or_zero();
    let side = vec2(-dir.y, dir.x);
    draw_triangle(to + dir * 6.0, to - dir * 10.0 + side * 8.0, to - dir * 10.0 - side * 8.0, color);
}

/// Values are pre-normalized to [-1, 1] (symmetric) or [0, 1].
#[allow(clippy::too_many_arguments)]
fn strip_chart(values: &[f64], x: f32, y: f32, w: f32, h: f32, symmetric: bool, color: Color, label: &str) {
    draw_rectangle(x, y, w, h, Color::new(1.0, 1.0, 1.0, 0.04));
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(1.0, 1.0, 1.0, 0.15));
    draw_text(label, x, y - 6.0, 16.0, DIM);
    let baseline = if symmetric { y + h / 2.0 } else { y + h };
    draw_line(x, baseline, x + w, baseline, 1.0, Color::new(1.0, 1.0, 1.0, 0.12));
    let cap = (HISTORY_SECS / HISTORY_DT) as f32;
    let scale = if symmetric { h / 2.0 } else { h };
    let point = |i: usize, v: f64| vec2(x + w * i as f32 / cap, baseline - (v.clamp(-1.0, 1.0) as f32) * scale);
    for (i, pair) in values.windows(2).enumerate() {
        let (a, b) = (point(i, pair[0]), point(i + 1, pair[1]));
        draw_line(a.x, a.y, b.x, b.y, 1.5, color);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app = App::new();
    loop {
        let (joints, _, _) = app.layout();
        app.handle_input(&joints);
        app.advance(get_frame_time() as f64);
        let (joints, ppm, ground) = app.layout();
        app.draw(&joints, ppm, ground);
        next_frame().await;
    }
}
