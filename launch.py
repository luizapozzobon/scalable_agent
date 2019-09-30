import argparse
import datetime
import getpass
import itertools
import os
import sys

import coolname  # pip install coolname
import gym

os.environ["OMP_NUM_THREADS"] = "1"

N_RUNS = 3
N_ACTORS = 48

ENVS = [
    "AdventureNoFrameskip-v4",
    "AirRaidNoFrameskip-v4",
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "AsteroidsNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "BerzerkNoFrameskip-v4",
    "BowlingNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "CarnivalNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "ChopperCommandNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    # "DefenderNoFrameskip-v4",  # gym.make never returns in py3.6.
    "DemonAttackNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "ElevatorActionNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "GravitarNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "IceHockeyNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "JourneyEscapeNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MontezumaRevengeNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "PitfallNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "PooyanNoFrameskip-v4",
    "PrivateEyeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "RobotankNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "TennisNoFrameskip-v4",
    "TimePilotNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4",
    "WizardOfWorNoFrameskip-v4",
    "YarsRevengeNoFrameskip-v4",
    "ZaxxonNoFrameskip-v4",
]

parser = argparse.ArgumentParser(description="Atari training")
parser.add_argument("--dry", action="store_true")


def make_experiment_arguments(env_name, xpid, log_dir):
    params = dict(
        level_name=env_name,
        num_actions=gym.make(env_name).action_space.n,
        mode="train",
        logdir=f"{log_dir}/{xpid}",
        num_actors=N_ACTORS,
        total_environment_frames=int(200e6),
        batch_size=32,
        unroll_length=20,
        entropy_cost=0.01,
        baseline_cost=0.5,
        discount=0.99,
        reward_clipping="abs_one",
        learning_rate=0.0006,
        decay=0.99,
        momentum=0,
        epsilon=0.01,
        grad_norm_clipping=40.0,
    )
    return params


def dict2flags(params):
    flags = []
    for flag, v in params.items():
        # flag = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                # NOTE: assumes we only use store_true as action for bools
                flags.append((f"--{flag}",))
        else:
            flags.append((f"--{flag}", str(v)))
    return flags


def flags2str(flags):
    return " ".join(list(itertools.chain(*flags)))


def launch_experiment(experiment_args):
    import multiprocessing as mp

    def launch_experiment(experiment_args):
        import subprocess
        import itertools
        import sys

        python_exec = sys.executable
        args = itertools.chain(*experiment_args)
        subprocess.call([python_exec, "experiment.py"] + list(args))

    experiment_process = mp.Process(target=launch_experiment, args=[experiment_args])
    experiment_process.start()
    experiment_process.join()


def main():
    flags = parser.parse_args()

    log_dir = "~/logs/torchbeast/tfimpala/logs"

    now = datetime.datetime.now().strftime("%Y%m%d")
    rootdir = f"{now}-{coolname.generate_slug()}"

    output_dir = os.path.expanduser(log_dir)
    os.makedirs(output_dir, exist_ok=True)

    job_index = 0
    SWEEP = list(itertools.product(range(N_RUNS), ENVS))
    for n, env in SWEEP:
        job_index += 1
        print("########## Job {:>4}/{} ##########".format(job_index, len(SWEEP)))

        xpid = f"{rootdir}/{env}-{n:03}"

        experiment_args = make_experiment_arguments(env, xpid, log_dir)

        print("$ python experiment.py %s" % flags2str(dict2flags(experiment_args)))

        if flags.dry:
            continue

        # Just launch. Might want to send to some scheduler instead.
        launch_experiment(experiment_args)


if __name__ == "__main__":
    main()
