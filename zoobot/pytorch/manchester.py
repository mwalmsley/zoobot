import logging

from lightning_lite.plugins.environments import SLURMEnvironment

# https://pytorch-lightning.readthedocs.io/en/stable/_modules/lightning_lite/plugins/environments/slurm.html#SLURMEnvironment
class ManchesterEnvironment(SLURMEnvironment):

    def __init__(self, auto_requeue: bool = True, requeue_signal=None) -> None:
        logging.info('Using Manchester SLURM environment')
        super().__init__(auto_requeue, requeue_signal)

    # @staticmethod
    # def resolve_root_node_address(nodes: str) -> str:
    #     root_node_address = super().resolve_root_node_address(nodes)
    #     logging.info(f'root_node_address: {root_node_address}')
    #     return root_node_address

    @staticmethod
    def detect() -> bool:
        return True
        
    @property
    def main_port(self) -> int:
        main_port = super().main_port
        logging.info(f'main_port: {main_port}')
        return main_port
        # MASTER_PORT will override

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    slurm_nodelist = "compute-0-[0,9]"

    env = ManchesterEnvironment()
    root = env.resolve_root_node_address(slurm_nodelist)
    print(root)

    print(env.detect())

    print(env.main_port)