import os
import json
import logging
import argparse
from eval_agent.utils.datatypes import State
logger = logging.getLogger("agent_frame")

def main(args:argparse.Namespace):
    # Load the config
    search_dir = output_path = os.path.join("outputs")
    state_list = []
    count_none = 0
    count_zero = 0
    if os.path.exists(search_dir):
        for file in os.listdir(search_dir):
            # 合并所有以args.base_modle_name.replace('/', '_')开头的文件夹
            if file.startswith(args.base_modle_name.replace('/', '_')):
                result_path = os.path.join(search_dir, file, args.exp_config+args.exp_name)
                for file_inner in os.listdir(result_path):
                    if not file_inner.endswith('json'):
                        continue
                    state = State.load_json(json.load(open(os.path.join(result_path, file_inner))))
                    state_list.append(state)
        reward_list = []
        success_list = []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
                if state.reward == 0:
                    count_zero += 1
            else:
                count_none += 1
            success_list.append(state.success)

        if len(reward_list) != 0:
            logger.warning(f"Average reward all: {sum(reward_list)/len(success_list):.4f}")
        logger.warning(f"Success rate all: {sum(success_list)/len(success_list):.4f}")
        logger.warning(f"None reward number: {count_none}")
        logger.warning(f"Zero reward number: {count_zero}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--base_modle_name",
        type=str,
        required=False,
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="webshop",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    args = parser.parse_args()
    main(args)