
// ---------------------------------------------------------------- engine_spike additions (scripts/fastenv/engine_spike)
// Appended to poke-engine-py/src/lib.rs by build_spike_engine.sh; never shipped.

// Both sides' legal options exactly as the MCTS root enumerates them (movechoice_to_string format:
// "<move>", "<move>-tera", "switch <species>", "No Move").
#[pyfunction]
fn spike_options(py_state: PyState) -> (Vec<String>, Vec<String>) {
    let state: State = py_state.into();
    let (s1, s2) = state.root_get_all_options();
    (
        s1.iter().map(|o| movechoice_to_string(&state.side_one, o)).collect(),
        s2.iter().map(|o| movechoice_to_string(&state.side_two, o)).collect(),
    )
}

struct SpikeRng(u64);
impl SpikeRng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn spike_all_fainted(side: &Side) -> bool {
    side.pokemon.pkmn.iter().all(|p| p.hp <= 0)
}

fn spike_n_fainted(state: &State) -> usize {
    state.side_one.pokemon.pkmn.iter().filter(|p| p.hp <= 0).count()
        + state.side_two.pokemon.pkmn.iter().filter(|p| p.hp <= 0).count()
}

// One uniformly random game inside the engine (pure Rust). Returns
// (winner 1/2/0 tie/-2 too long, turns, decisions, first faint turn or -1) and adds phase times (ns).
fn spike_play(
    start: &State,
    rng: &mut SpikeRng,
    max_decisions: u32,
    t: &mut [u128; 4],
) -> (i32, u32, u32, i32) {
    let t0 = std::time::Instant::now();
    let mut state = start.clone();
    t[3] += t0.elapsed().as_nanos();
    let (mut turns, mut decisions, mut first_faint) = (0u32, 0u32, -1i32);
    loop {
        let s1_dead = spike_all_fainted(&state.side_one);
        let s2_dead = spike_all_fainted(&state.side_two);
        if s1_dead || s2_dead {
            let w = if s1_dead && s2_dead { 0 } else if s1_dead { 2 } else { 1 };
            return (w, turns, decisions, first_faint);
        }
        if decisions >= max_decisions {
            return (-2, turns, decisions, first_faint);
        }
        let full_turn = !state.side_one.force_switch
            && !state.side_two.force_switch
            && state.side_one.get_active_immutable().hp > 0
            && state.side_two.get_active_immutable().hp > 0;
        if full_turn {
            turns += 1;
        }
        let fainted_before = spike_n_fainted(&state);
        let a = std::time::Instant::now();
        let (s1, s2) = state.root_get_all_options();
        let c1 = s1[rng.below(s1.len())].clone();
        let c2 = s2[rng.below(s2.len())].clone();
        let b = std::time::Instant::now();
        let branches = generate_instructions_from_move_pair(&mut state, &c1, &c2, true);
        let c = std::time::Instant::now();
        let total: f32 = branches.iter().map(|x| x.percentage).sum();
        let mut u = rng.unit() * total;
        let mut k = branches.len() - 1;
        for (i, br) in branches.iter().enumerate() {
            if u < br.percentage {
                k = i;
                break;
            }
            u -= br.percentage;
        }
        state.apply_instructions(&branches[k].instruction_list);
        let d = std::time::Instant::now();
        t[0] += (b - a).as_nanos();
        t[1] += (c - b).as_nanos();
        t[2] += (d - c).as_nanos();
        decisions += 1;
        if first_faint < 0 && spike_n_fainted(&state) > fainted_before {
            first_faint = turns as i32;
        }
    }
}

// Random-play games from each start state; returns (seconds total, [options, generate, sample+apply, clone] seconds,
// per game (state index, winner, turns, decisions, first faint turn)); winner -1 = the engine panicked.
#[pyfunction]
fn spike_random_games(
    states: Vec<PyState>,
    games_per_state: u32,
    max_decisions: u32,
    seed: u64,
) -> (f64, Vec<f64>, Vec<(u32, i32, u32, u32, i32)>) {
    let starts: Vec<State> = states.into_iter().map(|s| s.into()).collect();
    let mut rng = SpikeRng(seed | 1);
    let mut t = [0u128; 4];
    let mut out = Vec::with_capacity(starts.len() * games_per_state as usize);
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let start_time = std::time::Instant::now();
    for (i, s) in starts.iter().enumerate() {
        for _ in 0..games_per_state {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                spike_play(s, &mut rng, max_decisions, &mut t)
            }));
            match r {
                Ok((w, turns, dec, ff)) => out.push((i as u32, w, turns, dec, ff)),
                Err(_) => out.push((i as u32, -1, 0, 0, -1)),
            }
        }
    }
    let secs = start_time.elapsed().as_secs_f64();
    std::panic::set_hook(prev_hook);
    (secs, t.iter().map(|x| *x as f64 / 1e9).collect(), out)
}
