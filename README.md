# Navigation Dataset

This dataset documents human navigation behavior within a grid world and how humans infer the goals of other players.


## Grid Wrold of Size 6*6

We recruited a total of 800 participants to play navigation and belief inference games in a 6x6 grid world.

- **Navigation Game**: Human participants used the keyboard to move a player {up, down, right, left} from a starting position to a designated goal position (marked by a red star). The experimental data is available in [Experiment 1](experiment1/).
- **Belief Inference Game**: Human participants observed a robot's movements and inferred which goal (green triangle or red star) the robot was heading towards. The experimental data is available in [Experiment 2](experiment2/).

<div style="display: flex; justify-content: space-around;">
    <div style="text-align: center; margin-right: 20px;">
        <img src="figures/navigation-game.png" alt="Image 1" style="width: 300px;">
        <p style="margin-top: 0;">Experiment 1: Navigation Game Interface</p>
    </div>
    <div style="text-align: center;">
        <img src="figures/belief-inference.png" alt="Image 2" style="width: 300px;">
        <p style="margin-top: 0;">Experiment 2: Belief Inference Game Interface</p>
    </div>
</div>


## Grid Wrold of Size 8*8

We recruited a total of 600 participants to play navigation and belief inference games in an 8x8 grid world.

- **Navigation Game**: Human participants played navigation games alongside a pre-defined AI model. Participants used the keyboard to control player 0 (the blue player) {up, down, right, left, stay} from a starting position to one of the goal positions (marked by a blue star or blue triangle). The AI model controlled player 1 (the green player) to move to one of the goal positions (marked by a green star or green triangle). Participants received a positive reward of +10 if both players reached the same type of goal (either both stars or both triangles). No reward was given if the players collided (moved to the same position) or reached different types of goals. The experimental data is available in [Experiment 3](experiment3/).
- **Belief Inference Game**: Human participants observed the movements of two players and inferred which goal one of the players was heading towards. The experimental data is available in [Experiment 4](experiment4/).

<div style="display: flex; justify-content: space-around;">
    <div style="text-align: center; margin-right: 20px;">
        <img src="figures/navigation-game-two-player.png" alt="Image 3" style="width: 300px;">
        <p style="margin-top: 0;">Experiment 3: Two Player Navigation Game Interface</p>
    </div>
    <div style="text-align: center;">
        <img src="figures/belief-inference-two-player.png" alt="Image 4" style="width: 300px;">
        <p style="margin-top: 0;">Experiment 4: Two Player Belief Inference Interface</p>
    </div>
</div>


## Notebooks

Detailed data and game layouts are provided in [Notebooks/human_exp.ipynb](Notebooks/human_exp.ipynb).

## Citations

If you use this dataset for research purposes, please cite it as follows:

[link to add]
