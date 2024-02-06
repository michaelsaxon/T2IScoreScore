import argparse
import multiprocessing as mp
import os
import sys
import time
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from llm_descriptor.global_descriptor import GlobalDescriptor
from llm_descriptor.local_descriptor import LocalDescriptor
from llm_descriptor.visual_descriptor import VisualDescriptor
from llm_evaluator.evaluation_instruction import EvaluationInstructor

sys.path.insert(0, 'submodule/CenterNet2')
sys.path.insert(0, 'submodule/detectron2')
sys.path.insert(0, 'submodule/')
import pandas as pd
from centernet.config import add_centernet_config
from grit.config import add_grit_config
from grit.predictor import VisualizationDemo
from icecream import ic
from PIL import Image

WINDOW_NAME = "LLMScore(BLIPv2+GRiT+GPT-4)"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin llm_descriptor
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="submodule/grit/configs/GRiT_B_DenseCap_ObjectDet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--image",
        default="sample/sample.png",
    )
    parser.add_argument(
        "--llm_id",
        default="gpt-4",
    )
    parser.add_argument(
        "--text_prompt",
        default="a red car and a white sheep",
        help="text prompt",
    )
    parser.add_argument(
        "--output",
        default="sample/sample_result.png",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--versionidx",
        type=int,
        default=0,
        help="version",
    )
    parser.add_argument(
        "--startidx",
        type=int,
        default=0,
        help="startidx",
    )
    parser.add_argument(
        "--endidx",
        type=int,
        default=3000,
        help="endidx",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='DenseCap',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/grit_b_densecap_objectdet.pth"],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--metadata",
        help="Path to meta-data CSV file",
        default='data/metadata.csv',
    )
    parser.add_argument(
        "--image_folder",
        help="Base path for image files.",
        default='data/T2IScoreScore/',
    )
    return parser


def process_row(args, demo, llm_descriptor, llm_evaluator, row, logger):
    text_prompt = row['target_prompt']
    img_src = os.path.join(args.image_folder, row['file_name'])
    try:
        img = read_image(img_src, format="BGR")
    except:
        return None
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    logger.info(
        "{}: {} in {:.2f}s".format(
            img_src,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )
    local_description = local_descriptor.dense_pred_to_caption(predictions)
    out_filename = args.output
    visualized_output.save(out_filename)
    global_description = global_descriptor.get_global_description(img_src)
    image = Image.open(img_src)
    width, height = image.size
    scene_description = llm_descriptor.generate_multi_granularity_description(global_description, local_description, width, height)
    try:
        overall, error_counting = llm_evaluator.generate_score_with_rationale(scene_description, text_prompt)
    except:
        overall, error_counting = -1, -1
    return overall, error_counting

def process_csv_row(args, demo, llm_descriptor, llm_evaluator, row, logger):
    overall, error_counting = process_row(args, demo, llm_descriptor, llm_evaluator, row, logger)
    row['Overall'] = overall
    row['Error_Counting'] = error_counting
    return row

def load_data_and_process_images(args, demo, llm_descriptor, llm_evaluator):

    data = pd.read_csv(args.metadata)
    output_file = f'/output/LLMScore_{args.startidx}_{args.endidx}_{args.versionidx}.csv'
    with open(output_file, 'w', newline='') as file:
        writer = None
        for index, row in tqdm(data.iterrows(), total=len(data)):
            if args.startidx <= index <= args.endidx:
                processed_row = process_csv_row(args, demo, llm_descriptor, llm_evaluator, row, logger)
                if processed_row is not None:
                    row_df = processed_row.to_frame().T
                    if writer is None:
                        row_df.to_csv(file, header=True, index=False)
                        writer = True
                    else:
                        row_df.to_csv(file, header=False, index=False)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    openai_key = os.environ['OPENAI_KEY']
    global_descriptor = GlobalDescriptor()
    local_descriptor = LocalDescriptor()
    llm_descriptor = VisualDescriptor(openai_key, args.llm_id)
    llm_evaluator = EvaluationInstructor(openai_key, args.llm_id)

    load_data_and_process_images(args, demo, llm_descriptor, llm_evaluator)
