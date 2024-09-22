import os
import argparse

from mosec import Server

from backend_model_info import SeqXGPT2_ModelInfoContainer as modelInfo
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, default="gpt2", help= "The model to use. Supported models: [gpt2, gptneo, gptj].",
    )
    parser.add_argument("--gpu", type=str, required=False, default='0', help="Set os.environ['CUDA_VISIBLE_DEVICES'].")
    parser.add_argument("--port", help="Sets the port used by the mosec server.")
    parser.add_argument("--timeout", help="Sets a timeout on responding to API requests on the mosec server.")
    parser.add_argument("--debug", action="store_true", help="mosec args.")
    return parser.parse_args()


if __name__ == "__main__":
    # --model: [damo, gpt2, gptj, gptneo, wenzhong, skywork, llama]
    # python backend_api.py --port 6006 --timeout 30000 --debug --model=damo --gpu=3
    args = parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sniffer_model = modelInfo.MODEL_MAP[args.model]
    server = Server()
    server.append_worker(sniffer_model)
    server.run()