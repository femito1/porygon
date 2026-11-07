"""Figure: the fitted leaf evaluation's corrections to the stock score, Generation 4 vs Generation 9.

Each bar is a correction weight (logits per unit of the feature, on top of the stock score) times the feature's
standard deviation over ~270k engine self-play positions of that generation (data/feature_sd.json, from the fit's
training data): the change in predicted win logit for a typical (1 SD) change in the feature. Features are our side
minus the opponent's, so a negative bar means the fit values the feature less than the stock score does.
The 20 features with the largest effect in either generation are shown.

    python docs/paper/fig_weights.py   # needs matplotlib; writes fig_weights.pdf / .png next to this file
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "src/porygon/data"
NAMES = ['stock', 'alive_1', 'alive_2', 'alive_3', 'alive_4', 'alive_5', 'alive_6', 'hp_0_25', 'hp_25_50', 'hp_50_75',
         'hp_75_100', 'active_hp', 'burn_phys', 'burn_phys_hp', 'freeze', 'freeze_hp', 'sleep', 'sleep_hp', 'rest',
         'rest_hp', 'paralyze', 'paralyze_hp', 'toxic', 'toxic_hp', 'poison', 'poison_hp', 'status_good', 'item',
         'item_choice', 'item_consumable', 'sash_intact', 'sr_damage', 'spikes_damage', 'tspikes', 'substitute',
         'leech_seed', 'confusion', 'taunt', 'yawn', 'perish', 'curse', 'ingrain_aquaring', 'locked_move',
         'must_recharge', 'atk_boost', 'atk_boost_phys', 'spa_boost', 'spa_boost_spec', 'def_boost', 'spd_boost',
         'spe_boost', 'acc_boost', 'eva_boost', 'off_boost_hp', 'faster', 'reflect', 'light_screen', 'safeguard',
         'tailwind', 'mist', 'lucky_chant', 'wish', 'future_sight', 'toxic_count']  # pe.eval_feature_names()
LABELS = {
    "spikes_damage": "Spikes damage, per grounded mon", "spe_boost": "Speed boost (active)",
    "hp_75_100": "HP in top quarter, per mon", "hp_50_75": "HP in third quarter, per mon",
    "hp_25_50": "HP in second quarter, per mon", "hp_0_25": "HP in bottom quarter, per mon",
    "sash_intact": "Focus Sash at full HP", "atk_boost": "Attack boost (active)", "locked_move": "Locked into a move",
    "spa_boost": "Sp. Atk boost (active)", "wish": "Pending Wish", "freeze": "Frozen mon", "def_boost": "Defense boost",
    "item_consumable": "Holds a consumable item", "status_good": "Status that helps (Guts etc.)",
    "alive_6": "Exactly 6 mons alive", "alive_5": "Exactly 5 mons alive", "alive_4": "Exactly 4 mons alive",
    "alive_3": "Exactly 3 mons alive", "alive_2": "Exactly 2 mons alive", "alive_1": "Exactly 1 mon alive",
    "toxic": "Badly poisoned mon", "toxic_hp": "Badly poisoned mon x HP", "toxic_count": "Toxic counter (active)",
    "curse": "Cursed (active)", "leech_seed": "Leech Seeded (active)", "item": "Holds any item",
    "substitute": "Behind a Substitute", "sleep": "Asleep mon", "sleep_hp": "Asleep mon x HP", "faster": "Faster active",
    "active_hp": "Active mon's HP", "paralyze": "Paralyzed mon", "paralyze_hp": "Paralyzed mon x HP",
    "rest": "Asleep from Rest", "rest_hp": "Asleep from Rest x HP", "sr_damage": "Stealth Rock damage, per mon",
    "tspikes": "Toxic Spikes, per grounded mon", "off_boost_hp": "Offensive boosts x HP", "taunt": "Taunted",
    "spd_boost": "Sp. Def boost", "confusion": "Confused", "item_choice": "Holds a choice item",
    "burn_phys": "Burned physical attacker", "burn_phys_hp": "Burned physical attacker x HP",
    "poison": "Poisoned mon", "poison_hp": "Poisoned mon x HP", "future_sight": "Pending Future Sight",
    "reflect": "Reflect up", "light_screen": "Light Screen up", "tailwind": "Tailwind up", "yawn": "Drowsy (Yawn)",
    "must_recharge": "Must recharge", "atk_boost_phys": "Attack boost x physical", "acc_boost": "Accuracy boost",
    "spa_boost_spec": "Sp. Atk boost x special",
}
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (dataviz reference palette; validated, light mode)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOP = 20


def main() -> None:
    w4 = np.array(json.loads((DATA / "eval_r1.json").read_text())["weights"])
    w9 = np.array(json.loads((DATA / "eval_gen9_r1.json").read_text())["weights"])
    sd = json.loads((HERE / "data/feature_sd.json").read_text())
    e4, e9 = w4 * np.array(sd["gen4"]["sd"]), w9 * np.array(sd["gen9"]["sd"])
    idx = sorted(range(1, 64), key=lambda i: -max(abs(e4[i]), abs(e9[i])))[:TOP]
    idx.sort(key=lambda i: e4[i] + e9[i])  # most undervalued by the stock score at the top

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.5,
                         "xtick.color": INK2, "ytick.color": INK, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 4.1))  # IEEE single column
    y = np.arange(len(idx))[::-1]
    h = 0.38
    ax.barh(y + h / 2 + 0.01, e4[idx], height=h, color=BLUE, label="Generation 4", zorder=3)
    ax.barh(y - h / 2 - 0.01, e9[idx], height=h, color=ORANGE, label="Generation 9", zorder=3)
    ax.axvline(0, color=INK2, linewidth=0.6, zorder=4)
    ax.set_yticks(y, [LABELS.get(NAMES[i], NAMES[i]) for i in idx])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2, width=0.5)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Correction (logit per SD)", color=INK2)
    ax.set_ylim(-0.7, len(idx) - 0.3)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, handlelength=1.2,
               columnspacing=1.5, borderaxespad=0.2)
    lim = max(abs(e4[idx]).max(), abs(e9[idx]).max()) * 1.1
    ax.set_xlim(-lim, lim)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.965))  # the top strip holds the legend
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_weights.{ext}", dpi=300)
    for i in idx[::-1]:
        print(f"{NAMES[i]:16s} gen4 {e4[i]:+.3f}  gen9 {e9[i]:+.3f}")


if __name__ == "__main__":
    main()
