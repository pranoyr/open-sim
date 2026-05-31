import torch
from torch.utils.data import Dataset, DataLoader
import decord
import numpy as np
import cv2
import math
import os
import glob
import zipfile
import tempfile

class MittyFlowMatchingDataset(Dataset):
    def __init__(self, split="train", resolution=(224, 416), num_frames=41, mock=True):
        """
        Loads the paired human-robot videos from the Mitty dataset.
        If mock=True, generates high-fidelity synthetic demo animations.
        If mock=False, reads directly from the locally cached Hugging Face zip files.
        """
        self.mock = mock
        self.resolution = resolution
        self.num_frames = num_frames
        self.split = split
        
        if not self.mock:
            home_dir = os.path.expanduser("~")
            cache_pattern = os.path.join(home_dir, ".cache/huggingface/hub/datasets--showlab--Mitty_Dataset/snapshots/*/EPIC-KITCHENS.zip")
            matches = glob.glob(cache_pattern)
            
            if not matches:
                snapshot_dir = os.path.join(home_dir, ".cache/huggingface/hub/datasets--showlab--Mitty_Dataset/snapshots/3c94a01259c7b271789c83999604f27410061cdb")
                self.epic_zip_path = os.path.join(snapshot_dir, "EPIC-KITCHENS.zip")
                self.h2r_zip_path = os.path.join(snapshot_dir, "Human2Robot.zip")
            else:
                snapshot_dir = os.path.dirname(matches[0])
                self.epic_zip_path = matches[0]
                self.h2r_zip_path = os.path.join(snapshot_dir, "Human2Robot.zip")
                
            if not os.path.exists(self.epic_zip_path) or not os.path.exists(self.h2r_zip_path):
                raise FileNotFoundError(
                    f"Could not locate Mitty Dataset ZIP files in Hugging Face cache.\n"
                    f"Expected paths:\n"
                    f"  - {self.epic_zip_path}\n"
                    f"  - {self.h2r_zip_path}\n"
                    f"Please make sure the dataset is fully downloaded from Hugging Face."
                )
                
            self.pairs = []
            
            def get_zip_pairs(zip_path):
                zip_pairs = []
                with zipfile.ZipFile(zip_path, 'r') as z:
                    names = z.namelist()
                    human_files = [f for f in names if '/human/' in f and f.endswith('.mp4')]
                    robot_set = set(f for f in names if '/robot/' in f and f.endswith('.mp4'))
                    for h in human_files:
                        r = h.replace('/human/', '/robot/')
                        if r in robot_set:
                            zip_pairs.append((zip_path, h, r))
                return zip_pairs

            print("Indexing local zip archives...")
            h2r_pairs = get_zip_pairs(self.h2r_zip_path)
            epic_pairs = get_zip_pairs(self.epic_zip_path)
            all_pairs = h2r_pairs + epic_pairs
            print(f"Indexed {len(h2r_pairs)} pairs from Human2Robot and {len(epic_pairs)} pairs from EPIC-KITCHENS.")
            
            all_pairs.sort(key=lambda x: (x[0], x[1]))
            split_idx = int(len(all_pairs) * 0.9)
            
            if split == "train":
                self.pairs = all_pairs[:split_idx]
            else:
                self.pairs = all_pairs[split_idx:]
                
            print(f"Initialized {split} split with {len(self.pairs)} samples.")
            
    def __len__(self):
        if self.mock:
            return 100
        return len(self.pairs)

    def _decode_video_from_zip(self, zip_path, file_in_zip):
        """
        Extracts a single video from the local zip archive to a temporary file,
        decodes its frames using decord, resizes them, normalizes to [-1, 1],
        and returns a tensor of shape (C, T, H, W).
        """
        H, W = self.resolution
        T = self.num_frames
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                extracted_path = z.extract(file_in_zip, path=temp_dir)
                
            vr = decord.VideoReader(extracted_path)
            total_frames = len(vr)
            
            indices = np.linspace(0, total_frames - 1, T, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            
            resized_frames = []
            for frame in frames:
                if frame.shape[0] != H or frame.shape[1] != W:
                    frame = cv2.resize(frame, (W, H))
                resized_frames.append(frame)
            frames_np = np.stack(resized_frames, axis=0)
            
            frames_normalized = (frames_np.astype(np.float32) / 127.5) - 1.0
            
            frames_tensor = torch.from_numpy(frames_normalized).permute(3, 0, 1, 2)
            return frames_tensor

    def _decode_video_mock(self, video_type, idx):
        """
        Generates premium synthetic mock videos for testing.
        """
        H, W = self.resolution
        T = self.num_frames
        frames = []
        
        obj_x = int(W * 0.75)
        obj_y = int(H * 0.6)
        
        for t in range(T):
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            
            if t < T // 2:
                fraction = t / (T // 2)
                cx = int(W * 0.1 + (obj_x - W * 0.1 - 30) * fraction)
                cy = int(H * 0.3 + (obj_y - H * 0.3) * fraction)
            else:
                fraction = (t - T // 2) / (T // 2)
                cx = int(obj_x - 30 + 10 * math.sin(fraction * math.pi))
                cy = int(obj_y - 20 * fraction)
            
            if video_type == "human":
                # Render wooden table background
                frame[:] = [160, 200, 230]
                for dy in range(20, H, 30):
                    cv2.line(frame, (0, dy), (W, dy + 5), (130, 170, 205), 1)
                    
                # Render target apple
                cv2.circle(frame, (obj_x, obj_y), 16, (40, 40, 220), -1)
                cv2.circle(frame, (obj_x - 6, obj_y - 2), 14, (40, 40, 220), -1)
                cv2.line(frame, (obj_x - 2, obj_y - 12), (obj_x - 5, obj_y - 22), (20, 70, 100), 2)
                cv2.ellipse(frame, (obj_x - 8, obj_y - 20), (6, 3), -30, 0, 360, (50, 180, 50), -1)
                
                # Render reaching human arm & fingers
                sleeve_color = (200, 100, 50)
                cv2.line(frame, (0, H // 2), (cx - 45, cy), sleeve_color, 24)
                cv2.circle(frame, (cx - 45, cy), 12, sleeve_color, -1)
                
                skin_color = (180, 200, 255)
                cv2.line(frame, (cx - 45, cy), (cx, cy), skin_color, 16)
                cv2.circle(frame, (cx, cy), 11, skin_color, -1)
                
                finger_length = 14
                for angle_offset in [-0.6, -0.3, 0.0, 0.3, 0.6]:
                    curl = 0.8 if t >= T // 2 else 1.0
                    f_angle = angle_offset * curl
                    fx = int(cx + finger_length * math.cos(f_angle))
                    fy = int(cy + finger_length * math.sin(f_angle))
                    cv2.line(frame, (cx, cy), (fx, fy), skin_color, 4)
                
                cv2.putText(frame, "HUMAN DEMONSTRATION", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 50, 20), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Reaching for object...", (15, H - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
                
            else:
                # Render metallic lab bench background
                frame[:] = [190, 190, 190]
                for dx in range(40, W, 80):
                    cv2.line(frame, (dx, 0), (dx - 10, H), (150, 150, 150), 2)
                cv2.line(frame, (0, int(H * 0.7)), (W, int(H * 0.7)), (140, 140, 140), 3)
                
                # Render target block
                cv2.rectangle(frame, (obj_x - 14, obj_y - 14), (obj_x + 14, obj_y + 14), (0, 215, 255), -1)
                cv2.rectangle(frame, (obj_x - 14, obj_y - 14), (obj_x + 14, obj_y + 14), (0, 140, 180), 2)
                
                # Render multi-jointed robot arm & gripper
                base_x, base_y = 40, H - 40
                elbow_x = int((base_x + cx) / 2 - 20)
                elbow_y = int((base_y + cy) / 2 + 30)
                
                cv2.circle(frame, (base_x, base_y), 15, (60, 60, 60), -1)
                cv2.line(frame, (base_x, base_y), (elbow_x, elbow_y), (40, 40, 40), 12)
                cv2.line(frame, (base_x, base_y), (elbow_x, elbow_y), (100, 100, 100), 4)
                cv2.circle(frame, (elbow_x, elbow_y), 10, (50, 50, 200), -1)
                cv2.line(frame, (elbow_x, elbow_y), (cx, cy), (160, 160, 160), 8)
                cv2.line(frame, (elbow_x, elbow_y), (cx, cy), (220, 220, 220), 2)
                cv2.circle(frame, (cx, cy), 8, (60, 60, 60), -1)
                
                claw_open_width = 16 if t < T // 2 else 4
                cv2.rectangle(frame, (cx - 8, cy - 6), (cx + 8, cy + 6), (80, 80, 80), -1)
                cv2.line(frame, (cx - 6, cy + 6), (cx - claw_open_width, cy + 18), (120, 120, 120), 3)
                cv2.line(frame, (cx - claw_open_width, cy + 18), (cx - claw_open_width + 4, cy + 24), (120, 120, 120), 3)
                cv2.line(frame, (cx + 6, cy + 6), (cx + claw_open_width, cy + 18), (120, 120, 120), 3)
                cv2.line(frame, (cx + claw_open_width, cy + 18), (cx + claw_open_width - 4, cy + 24), (120, 120, 120), 3)
                
                cv2.putText(frame, "ROBOT EXECUTION", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Executing trajectory...", (15, H - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_normalized = (frame_rgb.astype(np.float32) / 127.5) - 1.0
            frames.append(frame_normalized)
            
        frames_np = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
        return frames_tensor

    def __getitem__(self, idx):
        if self.mock:
            robot_video = self._decode_video_mock("robot", idx)
            human_video = self._decode_video_mock("human", idx)
        else:
            zip_path, human_file, robot_file = self.pairs[idx]
            robot_video = self._decode_video_from_zip(zip_path, robot_file)
            human_video = self._decode_video_from_zip(zip_path, human_file)
        
        robot_reference_img = robot_video[:, 0:1, :, :]
        
        return {
            "target_video": robot_video,
            "condition_video": human_video,
            "reference_image": robot_reference_img
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Load real Hugging Face dataset instead of mock data")
    args = parser.parse_args()

    mock_mode = not args.real
    print(f"Initializing dataset (mode: {'Mock' if mock_mode else 'Real'})...")
    
    try:
        dataset = MittyFlowMatchingDataset(split="train", mock=mock_mode)
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        sample = dataset[0]
        
        robot_video = sample["target_video"]
        human_video = sample["condition_video"]
        
        C, T, H, W = robot_video.shape
        print(f"Playing video with {T} frames side-by-side. Press 'q' to quit.")
        
        window_name = f"Mitty Dataset (Left: Human, Right: Robot) - {'Mock' if mock_mode else 'Real'}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        for t in range(T):
            # Human frame conversion
            h_img = human_video[:, t, :, :].cpu().numpy()
            h_img = np.transpose(h_img, (1, 2, 0))
            h_img = np.clip((h_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            h_img_bgr = cv2.cvtColor(h_img, cv2.COLOR_RGB2BGR)
            
            # Robot frame conversion
            r_img = robot_video[:, t, :, :].cpu().numpy()
            r_img = np.transpose(r_img, (1, 2, 0))
            r_img = np.clip((r_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            r_img_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
            
            combined = np.hstack((h_img_bgr, r_img_bgr))
            cv2.imshow(window_name, combined)
            
            if cv2.waitKey(66) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        print("Playback finished.")
    except Exception as e:
        print(f"Error during visualization: {e}")