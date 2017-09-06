print("Starting Script")
from artemis.experiments.demo_experiments import demo_linear_regression

if __name__ == "__main__":

    # Open a menu that allows you to run experiments and view old ones.
    # slurm_kwargs = {"--gres":"gpu:1","-C":"TitanX", "-N":"1", "-t":"0-00:10:00"}
    slurm_kwargs = {"-t":"0-00:10:00"}
    demo_linear_regression.browse(slurm_kwargs=slurm_kwargs)
    # demo_linear_regression.browse(command="run all -p")
