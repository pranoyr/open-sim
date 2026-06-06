import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
import argparse
import gymnasium as gym
import gym_aloha
import json
import imageio
from tqdm import tqdm

def create_dataset(env_id, num_episodes, steps_per_episode, output_file, seed=42):
    env = gym.make(env_id, render_mode="rgb_array")
    
    # Extract resolution from a dummy render
    obs, info = env.reset(seed=seed)
    dummy_frame = env.render()
    if dummy_frame is None:
        raise ValueError("Environment render returned None. Ensure render_mode='rgb_array' is supported.")
    frame_h, frame_w, frame_c = dummy_frame.shape
    
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_id}")
    print(f"Action space: {action_dim}D")
    print(f"Frame resolution: {frame_w}x{frame_h}")
    print(f"Collecting {num_episodes} episodes with {steps_per_episode} steps each...")
    
    total_steps = num_episodes * steps_per_episode
    
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        obs, info = env.reset(seed=seed + ep)
        
        frames = []
        actions = []
        
        for step in range(steps_per_episode):
            # Motor babbling: sample random action
            action = env.action_space.sample()
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Render frame
            frame = env.render()
            frames.append(frame)
            
            # Convert action to python float list for JSON serialization
            actions.append(action.tolist())
            
            if terminated or truncated:
                break
        
        # Save video for this episode
        video_filename = f"{output_file}_ep{ep}.mp4"
        imageio.mimsave(video_filename, frames, fps=30)
        
        # Save actions to JSON for this episode
        json_filename = f"{output_file}_ep{ep}.json"
        with open(json_filename, 'w') as f:
            json.dump({"actions": actions}, f, indent=4)
            
    env.close()
    print(f"Dataset successfully saved with prefix {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a blabbering (motor babbling) dataset using ALOHA")
    parser.add_argument("--env_id", type=str, default="gym_aloha/AlohaInsertion-v0", help="ALOHA Gym environment ID")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to record")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps per episode")
    parser.add_argument("--output", type=str, default="IDM-dataset/aloha_babbling", help="Output file prefix (will append _ep0.mp4, _ep0.json, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    create_dataset(args.env_id, args.episodes, args.steps, args.output, args.seed)
