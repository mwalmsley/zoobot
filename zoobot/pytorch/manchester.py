import logging
import os


# from lightning_lite.plugins.environments import SLURMEnvironment

# # https://pytorch-lightning.readthedocs.io/en/stable/_modules/lightning_lite/plugins/environments/slurm.html#SLURMEnvironment
# # https://github.com/Lightning-AI/lightning/blob/9c20cad40e4142f8a5e945fe26e919e598f2bd56/src/lightning_lite/plugins/environments/slurm.py
# class ManchesterEnvironment(SLURMEnvironment):

#     def __init__(self, auto_requeue: bool = True, requeue_signal=None) -> None:
#         logging.info('Using Manchester SLURM environment')
#         super().__init__(auto_requeue, requeue_signal)

#     # @staticmethod
#     # def resolve_root_node_address(nodes: str) -> str:
#     #     root_node_address = super().resolve_root_node_address(nodes)
#     #     logging.info(f'root_node_address: {root_node_address}')
#     #     return root_node_address

#     @staticmethod
#     def detect() -> bool:
#         return True
        
#     @property
#     def main_port(self) -> int:
#         main_port = super().main_port
#         logging.info(f'main_port: {main_port}')
#         return main_port
#         # MASTER_PORT will override


from pytorch_lightning.plugins.environments import SLURMEnvironment
class GalahadEnvironment(SLURMEnvironment):
    def __init__(self, **kwargs):
        ntasks_per_node = os.environ["SLURM_TASKS_PER_NODE"].split("(")[0]
        os.environ["SLURM_NTASKS_PER_NODE"] = ntasks_per_node
        # os.environ["SLURM_NTASKS"] = str(os.environ["SLURM_NTASKS_PER_NODE"])
        super().__init__(**kwargs)
        self.nnodes = int(os.environ["SLURM_NNODES"])


# if __name__ == '__main__':

#     logging.basicConfig(level=logging.INFO)

#     # slurm_nodelist = "compute-0-[0,9]" # 0,9 works
#     slurm_nodelist = "compute-0-[0,11]"  # 0,11 hangs
#     # 70017 8-9 works

#     env = GalahadEnvironment()
#     root = env.resolve_root_node_address(slurm_nodelist)
#     print(root)

#     print(env.detect())

#     print(env.main_port)