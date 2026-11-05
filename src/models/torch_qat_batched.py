import torch
import torch.nn.functional as F

from src.models.torch_qat import alpha_beta_at, learning_rate_at
from src.models.torch_mlp import ACTIVATIONS
from src.quantization.torch_quantize import fixed_point_quantize
from src.quantization.torch_stair import stair_geometry

"""
Batched train_qat: many independent QAT runs of the same MLP shape trained in one set of tensors,
so a GPU does the work of all of them with one kernel launch per operation. A single run of this
small MLP is launch-bound on the GPU and no faster than one CPU core; batching is what pays.

Layout: run r's layer-i weight is W_i[r] with shape (out, in) (nn.Linear layout), its bias b_i[r]
has shape (1, out), its data X[r] has shape (N, features). The loss is the sum of the per-run MSEs,
so the runs never interact: each keeps its own data, start weights, lr, final alpha, grad mode,
optimizer state, early stopping and best checkpoint.

Every per-run rule is the one in train_qat (same stair arithmetic, same stopping rule, same order
of operations), so on CPU a batch reproduces train_qat run by run; experiments/test_torch_qat_batched.py
checks this. The optimizer is the unmodified torch.optim.Adam / SGD: runs that share an lr trajectory
form one parameter group (their own leaf tensors), and the forward pass concatenates the groups.

Runs that stop (converged or aborted) leave the batch, so the batch shrinks as training proceeds.

Supported: schedule_type "alpha_linear" (alpha_start -> alpha_end per run over transition_epochs),
lr schedule None or any learning_rate_at schedule, stair grad modes exact / clamped / ste mixed in
one batch, or hard STE for the whole batch. Test data is required (the exact test MSE is logged).
"""


class _BatchedStair(torch.autograd.Function):
    """
    StairQuantize with per-run geometry: geo[k] has shape (R, 1, 1) and broadcasts over x (R, ., .).
    hard: optional (R, 1, 1) bool; where True the forward is the real fixed-point rounding (as hard STE) and only the
    backward uses the stair slopes, masked where the rounding saturates (the hybrid).
    """

    @staticmethod
    def forward(ctx, x, a_half, slope_trap, slope_transition, grad_trap, grad_transition,
                total_bits, frac_bits, mask_saturation, hard=None):
        lsb = 2.0 ** (-frac_bits)
        centers = torch.round(x / lsb) * lsb
        diff = x - centers
        abs_diff = diff.abs()
        is_trap = abs_diff <= a_half

        x_soft = torch.where(
            is_trap,
            centers + diff * slope_trap,
            centers + torch.sign(diff) * (a_half * slope_trap + (abs_diff - a_half) * slope_transition),
        )

        q_max = ((2 ** (total_bits - 1)) - 1) * lsb
        q_min = -(2 ** (total_bits - 1)) * lsb
        out = torch.clamp(x_soft, q_min, q_max)
        in_range = (x_soft >= q_min) & (x_soft <= q_max)
        if hard is not None:
            out = torch.where(hard, fixed_point_quantize(x, total_bits, frac_bits), out)
            # fake_quantize's straight-through mask: the rounded integer lies in the representable range
            in_range = torch.where(hard, (centers >= q_min) & (centers <= q_max), in_range)

        local_grad = torch.where(is_trap, grad_trap, grad_transition)
        if mask_saturation:
            local_grad = local_grad * in_range.to(x.dtype)

        ctx.save_for_backward(local_grad)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (local_grad,) = ctx.saved_tensors
        return grad_output * local_grad, None, None, None, None, None, None, None, None, None


def _geometry(lsb, alpha, beta, grad_mode, grad_floor=None):
    """
    (a/2, flat slope, steep slope, backward flat slope, backward steep slope), as in StairQuantize.
    grad_floor: optional lower bound on the backward flat-zone slope (forward unchanged).
    """
    a, _, _, _, slope_trap, slope_transition = stair_geometry(lsb, alpha, beta)
    if grad_mode == "exact":
        trap_g, trans_g = slope_trap, slope_transition
    elif grad_mode == "clamped":
        trap_g = min(max(slope_trap, 0.1), 1.0)
        trans_g = min(max(slope_transition, 0.0), 1.0)
    elif grad_mode == "ste":
        trap_g = trans_g = 1.0
    else:
        raise ValueError(f"Unknown grad_mode {grad_mode!r}")
    if grad_floor is not None:
        trap_g = max(trap_g, grad_floor)
    return a / 2.0, slope_trap, slope_transition, trap_g, trans_g


def _forward(Ws, bs, x, act, q_param, q_act, quantize_activations, total_bits, frac_bits, diag=None):
    """
    forward_qat / forward_hard_ste of QATMLP, batched over runs. diag: optional dict that receives,
    per run, the share of pre-activations and of hidden activations outside the representable range.
    """
    a = fixed_point_quantize(x, total_bits, frac_bits) if quantize_activations else x
    n = len(Ws)
    lsb = 2.0 ** (-frac_bits)
    q_max, q_min = ((2 ** (total_bits - 1)) - 1) * lsb, -(2 ** (total_bits - 1)) * lsb
    sat_z, sat_a, n_z, n_a = 0, 0, 0, 0
    for i in range(n):
        z = a @ q_param(Ws[i]).transpose(1, 2) + q_param(bs[i])
        last = i == n - 1
        if diag is not None:
            with torch.no_grad():
                sat_z = sat_z + ((z < q_min) | (z > q_max)).sum(dim=(1, 2))
                n_z += z[0].numel()
                if not last:
                    za = act(z)
                    sat_a = sat_a + ((za < q_min) | (za > q_max)).sum(dim=(1, 2))
                    n_a += z[0].numel()
        if quantize_activations and not last:
            z = q_act(z)
        a = z if last else act(z)
        if quantize_activations:
            a = q_act(a)
    if diag is not None:
        diag["sat_preact"] = sat_z / n_z
        diag["sat_act"] = sat_a / n_a
    return a


def _forward_exact(Ws, bs, x, act, total_bits, frac_bits):
    """TorchMLP.forward_quantized (everything quantized, hard rounding), batched over runs."""
    def q(t):
        return fixed_point_quantize(t, total_bits, frac_bits)

    a = q(x)
    n = len(Ws)
    for i in range(n):
        z = a @ q(Ws[i]).transpose(1, 2) + q(bs[i])
        a = q(z) if i == n - 1 else q(act(q(z)))
    return a


def _mse(pred, y):
    """Per-run MSE, shape (R,)."""
    return F.mse_loss(pred, y, reduction="none").mean(dim=(1, 2))


def train_qat_batched(
    runs,
    max_epochs=50000,
    total_bits=8,
    frac_bits=4,
    alpha_start=1.0,
    transition_epochs=10000,
    lr_power=0.5,
    mask_saturation=True,
    quantize_activations=False,
    patience=2000,
    min_delta=1e-6,
    hard_ste=False,
    save_state_at=None,
    abort_val_above=None,
    abort_epoch=60,
    optimizer_name="sgd",
    activation="relu",
    device="cpu",
    dtype=torch.float64,
    verbose=False,
    on_done=None,
    float_train=False,
    diagnostics=False,
):
    """
    Train len(runs) QAT runs at once. Arguments as in train_qat, but batch-wide; per run:

    runs : list of dicts with
        state       -- start parameters, a QATMLP state_dict (layers.<i>.weight / layers.<i>.bias)
        data        -- dict X_train, y_train, X_val, y_val, X_test, y_test (same shapes across runs)
        lr          -- initial learning rate
        alpha_end   -- final alpha (ignored with hard_ste)
        grad_mode   -- "exact" | "clamped" | "ste" (ignored with hard_ste)
        lr_schedule -- optional learning_rate_at schedule name (default None = constant)
        optional, defaulting to the batch-wide argument / train_qat's behaviour:
        transition  -- alpha ramp length in epochs (transition_epochs)
        beta        -- stair beta (1.0)
        grad_floor  -- lower bound on the backward flat-zone slope (None)
        betas       -- Adam betas ((0.9, 0.999))
        patience    -- early-stopping patience (patience)
        abort_above -- abort threshold at abort_epoch (abort_val_above)
        warmup      -- linear lr warm-up over this many epochs, multiplying the scheduled lr (0 = none)
        hard_forward -- stair runs only: real fixed-point rounding in the forward (as hard STE), the stair slopes of
                    grad_mode in the backward (the hybrid); False = the stair forward (False)
        beta_ramp   -- True: beta goes linearly from 1 to beta together with the alpha ramp, so training starts at the
                    identity (alpha = beta = 1); False: beta is fixed from the first epoch (False)
        judge_after_ramp -- hard_ste / float only: with an lr_schedule, the lr follows the alpha ramp the stair
                    runs would see (alpha_end, transition), so hard STE gets the same lr trajectory; True also
                    starts the stopping rule / best selection after the ramp, as for stair runs (False)

    float_train : train in float (no quantization in the training forward, no alpha), with the same
        exact fixed-point stopping rule from epoch 0: the float-continuation control.
    diagnostics : also log per epoch the share of parameters / pre-activations / hidden activations
        outside the representable range (training forward) and the share of hard-quantized
        parameters whose grid value changed in this epoch's step and the share that returned to the grid
        value of two steps before (oscillation; both NaN right after the batch shrinks).

    on_done : optional callback on_done(index, history, state), called as soon as a run stops, so a
        long batch can save finished runs on the way.

    Returns a list, one entry per run in input order, of (history, state): history has the keys of
    train_qat's history; state is the final state_dict on CPU (the best checkpoint when one exists,
    as train_qat restores it).
    """
    act = ACTIVATIONS[activation]
    lsb = 2.0 ** (-frac_bits)
    save_state_at = set(save_state_at or [])
    n_layers = len([k for k in runs[0]["state"] if k.endswith(".weight")])
    to = dict(device=device, dtype=dtype)

    def prep(t, *shape):
        return t.detach().to(**to).reshape(*shape) if shape else t.detach().to(**to)

    data = [{"X": prep(r["data"]["X_train"]), "y": prep(r["data"]["y_train"], -1, 1),
             "Xv": prep(r["data"]["X_val"]), "yv": prep(r["data"]["y_val"], -1, 1),
             "Xt": prep(r["data"]["X_test"]), "yt": prep(r["data"]["y_test"], -1, 1)} for r in runs]
    cfg = [{"lr": float(r["lr"]), "alpha_end": float(r.get("alpha_end", 10.0)),
            "grad_mode": r.get("grad_mode", "exact"), "lr_schedule": r.get("lr_schedule"),
            "transition": int(r.get("transition", transition_epochs)), "beta": float(r.get("beta", 1.0)),
            "grad_floor": r.get("grad_floor"), "betas": tuple(r.get("betas", (0.9, 0.999))),
            "patience": int(r.get("patience", patience)), "abort": r.get("abort_above", abort_val_above),
            "warmup": int(r.get("warmup", 0)), "judge_after_ramp": bool(r.get("judge_after_ramp", False)),
            "beta_ramp": bool(r.get("beta_ramp", False)), "hard_forward": bool(r.get("hard_forward", False))}
           for r in runs]
    fixed = hard_ste or float_train   # no alpha, stopping rule active from epoch 0

    h = [{"alpha": [], "beta": [], "lr": [], "train_loss": [], "val_qat_loss": [], "val_exact_mse": [],
          "test_exact_mse": [], "states": {}, "grad_stats": [], "steep_fraction": [], "grad_snapshots": {},
          "best_epoch": None, "best_val_exact_mse": None, "stopped_epoch": None, "converged": False,
          "aborted": False} for _ in runs]
    if diagnostics:
        for hr in h:
            hr.update({"sat_params": [], "sat_preact": [], "sat_act": [], "wq_change": [], "wq_flipback": []})
    best = [float("inf")] * len(runs)
    best_state = [None] * len(runs)
    since_best = [0] * len(runs)
    final_state = [None] * len(runs)

    def finish(r, state):
        if h[r]["stopped_epoch"] is None:
            h[r]["stopped_epoch"] = len(h[r]["train_loss"]) - 1
        if best_state[r] is not None:
            state = best_state[r]
            h[r]["best_val_exact_mse"] = best[r]
        final_state[r] = state
        if on_done is not None:
            on_done(r, h[r], state)

    # one parameter group per lr trajectory; the schedule may depend on alpha, so alpha_end joins the key
    # (and the ramp length); Adam betas live in the parameter group too
    def group_key(r):
        c = cfg[r]
        alpha_dep = c["lr_schedule"] is not None
        return (c["lr"], c["lr_schedule"], c["alpha_end"] if alpha_dep else None,
                c["transition"] if alpha_dep else None, c["warmup"], c["betas"])

    def build(groups_runs, params_of, state_of=None):
        """Leaf tensors per group (+ optimizer, carrying over sliced state)."""
        groups = []
        for key, members in groups_runs:
            Ws = [torch.stack([params_of(r)[f"layers.{i}.weight"] for r in members]).requires_grad_()
                  for i in range(n_layers)]
            bs = [torch.stack([params_of(r)[f"layers.{i}.bias"].reshape(1, -1) for r in members]).requires_grad_()
                  for i in range(n_layers)]
            groups.append({"key": key, "runs": members, "W": Ws, "b": bs})
        param_groups = [{"params": g["W"] + g["b"], "lr": g["key"][0]} for g in groups]
        if optimizer_name == "adam":
            for pg, g in zip(param_groups, groups):
                pg["betas"] = g["key"][-1]
        if optimizer_name == "sgd":
            opt = torch.optim.SGD(param_groups, lr=param_groups[0]["lr"])
        elif optimizer_name == "adam":
            opt = torch.optim.Adam(param_groups, lr=param_groups[0]["lr"])
        else:
            raise ValueError(f"Unknown optimizer_name {optimizer_name!r}; expected 'sgd' or 'adam'")
        if state_of is not None:
            for g in groups:
                for p in g["W"] + g["b"]:
                    st = state_of(g, p)
                    if st:
                        opt.state[p] = st
        return groups, opt

    order = []
    for r in range(len(runs)):
        k = group_key(r)
        if k not in [key for key, _ in order]:
            order.append((k, []))
        next(m for key, m in order if key == k).append(r)
    init = [{k: v.detach().to(**to) for k, v in r["state"].items()} for r in runs]
    groups, opt = build(order, lambda r: init[r])

    def stack_data():
        alive = [r for g in groups for r in g["runs"]]
        return alive, {k: torch.stack([data[r][k] for r in alive]) for k in data[0]}

    alive, D = stack_data()

    def params():
        Ws = [torch.cat([g["W"][i] for g in groups]) if len(groups) > 1 else groups[0]["W"][i] for i in range(n_layers)]
        bs = [torch.cat([g["b"][i] for g in groups]) if len(groups) > 1 else groups[0]["b"][i] for i in range(n_layers)]
        return Ws, bs

    def run_state(Ws, bs, j):
        return {**{f"layers.{i}.weight": Ws[i][j].detach().cpu().clone() for i in range(n_layers)},
                **{f"layers.{i}.bias": bs[i][j, 0].detach().cpu().clone() for i in range(n_layers)}}

    def quantized_params(Ws, bs):
        return torch.cat([fixed_point_quantize(t, total_bits, frac_bits).reshape(len(alive), -1) for t in Ws + bs], 1)

    nan = float("nan")
    prev_q = prev2_q = None
    for epoch in range(max_epochs):
        ramp = [min(epoch / (cfg[r]["transition"] - 1), 1.0) if cfg[r]["transition"] > 1 else 1.0 for r in alive]
        ramp_alphas = [(alpha_beta_at(p, "alpha_linear", alpha_start, cfg[r]["alpha_end"])[0],
                        1.0 + (cfg[r]["beta"] - 1.0) * p if cfg[r]["beta_ramp"] else cfg[r]["beta"])
                       for p, r in zip(ramp, alive)]
        if fixed:
            prog = [p if cfg[r]["judge_after_ramp"] else 1.0 for p, r in zip(ramp, alive)]
            alphas = [(nan, nan)] * len(alive)
        else:
            prog, alphas = ramp, ramp_alphas
        lrs = [learning_rate_at(epoch, max_epochs, cfg[r]["lr_schedule"], cfg[r]["lr"],
                                ramp_alphas[j][0], ramp_alphas[j][1], alpha_start, lr_power)
               * (min(1.0, (epoch + 1) / cfg[r]["warmup"]) if cfg[r]["warmup"] > 0 else 1.0)
               for j, r in enumerate(alive)]
        pos = 0
        for g, pg in zip(groups, opt.param_groups):
            pg["lr"] = lrs[pos]
            pos += len(g["runs"])

        if float_train:
            def q_param(t):
                return t
            q_act = q_param
            a_half = None
        elif hard_ste:
            def q_param(t):
                return fixed_point_quantize(t, total_bits, frac_bits, straight_through=True)
            q_act = q_param
            a_half = None
        else:
            geo = torch.tensor([_geometry(lsb, al, be, cfg[r]["grad_mode"], cfg[r]["grad_floor"])
                                for (al, be), r in zip(alphas, alive)],
                               **to).reshape(len(alive), 5, 1, 1).unbind(1)
            a_half = geo[0]
            hard_fwd = (torch.tensor([cfg[r]["hard_forward"] for r in alive], device=device).reshape(-1, 1, 1)
                        if any(cfg[r]["hard_forward"] for r in alive) else None)

            def q_param(t):
                return _BatchedStair.apply(t, *geo, total_bits, frac_bits, mask_saturation, hard_fwd)
            q_act = q_param

        qa = quantize_activations and not float_train
        diag = {} if diagnostics else None
        opt.zero_grad()
        Ws, bs = params()
        losses = _mse(_forward(Ws, bs, D["X"], act, q_param, q_act, qa, total_bits, frac_bits, diag), D["y"])
        losses.sum().backward()

        # per-layer |grad| stats and steep fractions, computed before the step as in train_qat
        stats = []
        for g_i in range(n_layers):
            gw = torch.cat([g["W"][g_i].grad for g in groups]) if len(groups) > 1 else groups[0]["W"][g_i].grad
            gb = torch.cat([g["b"][g_i].grad for g in groups]) if len(groups) > 1 else groups[0]["b"][g_i].grad
            gg = torch.cat([gw.reshape(len(alive), -1), gb.reshape(len(alive), -1)], dim=1).abs()
            stats += [gg.mean(1), gg.median(1).values, gg.amax(1), gg.amin(1), gg.norm(dim=1)]
        with torch.no_grad():
            if diagnostics:
                q_max, q_min = ((2 ** (total_bits - 1)) - 1) * lsb, -(2 ** (total_bits - 1)) * lsb
                flat = torch.cat([t.reshape(len(alive), -1) for t in Ws + bs], 1)
                diag["sat_params"] = ((flat < q_min) | (flat > q_max)).to(dtype).mean(1)
            if fixed:
                steep = [torch.full((len(alive),), nan, **to)] * n_layers
            else:
                steep = [((W - torch.round(W / lsb) * lsb).abs() > a_half).to(torch.float64).mean(dim=(1, 2)).to(dtype)
                         for W in Ws]

        opt.step()

        with torch.no_grad():
            Ws, bs = params()
            val_qat = _mse(_forward(Ws, bs, D["Xv"], act, q_param, q_act, qa, total_bits, frac_bits), D["yv"])
            val_exact = _mse(_forward_exact(Ws, bs, D["Xv"], act, total_bits, frac_bits), D["yv"])
            test_exact = _mse(_forward_exact(Ws, bs, D["Xt"], act, total_bits, frac_bits), D["yt"])
            extra = []
            if diagnostics:
                q_now = quantized_params(Ws, bs)
                change = ((q_now != prev_q).to(dtype).mean(1) if prev_q is not None
                          else torch.full((len(alive),), nan, **to))
                # oscillation: grid value left in the last step and returned in this one (A -> B -> A)
                flipback = (((q_now == prev2_q) & (prev_q != prev2_q)).to(dtype).mean(1) if prev2_q is not None
                            else torch.full((len(alive),), nan, **to))
                prev2_q, prev_q = prev_q, q_now
                extra = [diag["sat_params"], diag["sat_preact"].to(dtype), diag["sat_act"].to(dtype), change, flipback]
            row = torch.stack([losses.detach(), val_qat, val_exact, test_exact] + stats + steep + extra,
                              dim=1).cpu().tolist()

        stopped = []
        for j, r in enumerate(alive):
            v = row[j]
            hr = h[r]
            if epoch in save_state_at:
                hr["states"][epoch] = run_state(Ws, bs, j)
            hr["alpha"].append(alphas[j][0]); hr["beta"].append(alphas[j][1]); hr["lr"].append(lrs[j])
            hr["train_loss"].append(v[0]); hr["val_qat_loss"].append(v[1])
            hr["val_exact_mse"].append(v[2]); hr["test_exact_mse"].append(v[3])
            hr["grad_stats"].append([dict(zip(("mean", "median", "max", "min", "norm"), v[4 + 5 * i: 9 + 5 * i]))
                                     for i in range(n_layers)])
            hr["steep_fraction"].append(v[4 + 5 * n_layers: 4 + 6 * n_layers])
            if diagnostics:
                for k, x in zip(("sat_params", "sat_preact", "sat_act", "wq_change", "wq_flipback"), v[4 + 6 * n_layers:]):
                    hr[k].append(x)

            val = v[2]
            abort_above = cfg[r]["abort"]
            if abort_above is not None and epoch + 1 == abort_epoch and val > abort_above:
                hr["aborted"] = True
                hr["stopped_epoch"] = epoch
                stopped.append((j, r))
                continue
            if prog[j] >= 1.0:
                if val < best[r] - min_delta:
                    best[r], since_best[r] = val, 0
                    best_state[r] = run_state(Ws, bs, j)
                    hr["best_epoch"] = epoch
                else:
                    since_best[r] += 1
                    if since_best[r] >= cfg[r]["patience"]:
                        hr["converged"] = True
                        hr["stopped_epoch"] = epoch
                        stopped.append((j, r))

        if verbose and (epoch % 500 == 0 or epoch == max_epochs - 1):
            print(f"Epoch {epoch}: {len(alive)} runs alive, mean val_exact {sum(x[2] for x in row) / len(row):.5f}",
                  flush=True)

        if stopped:
            for j, r in stopped:
                finish(r, run_state(Ws, bs, j))
            done = {r for _, r in stopped}
            remaining = [(g["key"], [r for r in g["runs"] if r not in done]) for g in groups]
            remaining = [(k, m) for k, m in remaining if m]
            if not remaining:
                break
            # carry each surviving run's parameters and optimizer state into the smaller batch
            old = {g["key"]: g for g in groups}
            cur = {}
            for g in groups:
                for i, r in enumerate(g["runs"]):
                    cur[r] = (g, i)

            def params_of(r):
                g, i = cur[r]
                return {**{f"layers.{l}.weight": g["W"][l][i].detach() for l in range(n_layers)},
                        **{f"layers.{l}.bias": g["b"][l][i, 0].detach() for l in range(n_layers)}}

            old_opt = opt

            def state_of(g_new, p_new):
                g_old = old[g_new["key"]]
                idx = torch.tensor([g_old["runs"].index(r) for r in g_new["runs"]], device=device)
                k = next(i for i, p in enumerate(g_new["W"] + g_new["b"]) if p is p_new)
                st = old_opt.state.get((g_old["W"] + g_old["b"])[k])
                if not st:
                    return None
                return {name: (t.clone() if t.dim() == 0 else t.index_select(0, idx).clone())
                        for name, t in st.items()}

            groups, opt = build(remaining, params_of, state_of)
            alive, D = stack_data()
            prev_q = prev2_q = None
    else:
        Ws, bs = params()
        for j, r in enumerate(alive):
            finish(r, run_state(Ws, bs, j))

    return [(h[r], final_state[r]) for r in range(len(runs))]
