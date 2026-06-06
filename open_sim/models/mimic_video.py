import torch
from mimic_video.cosmos_predict import CosmosPredictWrapper
from mimic_video import MimicVideo

def create_mimic_video_idm():
    video_wrapper = CosmosPredictWrapper(
        extract_layer = 1,
        random_weights = True,
        tiny = True
    )

    # We match ALOHA's dimensions
    model = MimicVideo(
        512, 
        video_wrapper,
        dim_action=14,
        dim_joint_state=14,
        action_chunk_len=15
    )
    return model
