import os
import argparse

from mosec import Server

from backend_model_info import SeqXGPT2_ModelInfoContainer as modelInfo
from backend_model import SnifferModel
from config_manager import ConfigManager
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, default="gpt2", help= "The model to use. Supported models: [gpt2, gptneo, gptj].",
    )
    parser.add_argument("--gpu", type=str, required=False, default='0', help="Set os.environ['CUDA_VISIBLE_DEVICES'].")
    parser.add_argument("--port", help="Sets the port used by the mosec server.")
    parser.add_argument("--timeout", help="Sets a timeout on responding to API requests on the mosec server.")
    parser.add_argument("--debug", action="store_true", help="mosec args.")
    parser.add_argument("--host", default="0.0.0.0", help="Sets the accessible host address of the mosec server. Default: 0.0.0.0")
    return parser.parse_args()


if __name__ == "__main__":
    # --model: [damo, gpt2, gptj, gptneo, wenzhong, skywork, llama]
    # python backend_api.py --port 6006 --timeout 30000 --debug --model=damo --gpu=3
    args = parse_args()
    config_manager = ConfigManager(args.model)
    config_manager.write_args(args)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    selectParams = modelInfo.MODEL_MAP[args.model]
    def dynamicInit(self):
        for k, v in selectParams.items:
            super.__setattr__(self,k,v)
                    
    sniffer_model = type("SnifferDynamicModel", (SnifferModel),  { "__init__": dynamicInit })
    server = Server()
    server.append_worker(sniffer_model)
    server.run()