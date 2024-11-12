import os
import tqdm
import json
import argparse
import time 
import ast

import torch
import gc

from counting.run_count import run
from counting.count import args

import pandas as pd

def tensor_to_dict(count_per_class):
    # Dictionary keys for the selected vehicle types
    vehicle_types = ["bicycle", "car", "motorcycle", "bus", "truck"]
    
    # Indices corresponding to the vehicle types in the tensor
    indices = [1, 2, 3, 5, 7]
    
    # Create the dictionary
    vehicle_counts = {vehicle: count_per_class[idx].item() for vehicle, idx in zip(vehicle_types, indices)}
    return vehicle_counts

def main(vid_strides):
    
    args.counting_approach = "tracking_with_line_crossing"
    args.save = False
    args.verbose = False
    args.use_mask = False
    args.save_csv_count = False
    args.tracking_method  = "ocsort"
    
    folder_path = os.path.join(os.getcwd(), "dataset")
    videos = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    video_names = [f"kech{idx}.mp4" for idx in range(2, len(videos) + 2)]
    lines_of_counting = list(pd.read_csv(os.path.join(os.getcwd(), "dataset", "actual_counts.csv"))["line_of_counting"].apply(ast.literal_eval))
    
    

    for vid_stride in vid_strides:
        print(f"stride: {vid_stride}")
        args.vid_stride = vid_stride
        overall_results = {}
        
        for video , line_of_counting in zip(video_names , lines_of_counting):
            print(f"video: {video}")
            args.source = os.path.join(os.getcwd(), "dataset", video)
            dict_count, dict_runtime = {}, {}

            args.line_point11 , args.line_point12 = line_of_counting
         
        
            counter_yolo, profilers, _ = run(args)

            dict_count = {
                "count_per_class": tensor_to_dict(counter_yolo.count_per_class),
                "total_count": int(counter_yolo.counter)
            }
            dict_runtime = {
                "Preprocessing": profilers[0].t,
                "Detection": profilers[1].t,
                "Postprocessing": profilers[2].t,
                "Tracking" : profilers[3].t , 
                "Counting": profilers[4].t
            }

            del counter_yolo, profilers
            torch.cuda.empty_cache()
            gc.collect()

            overall_results[video] = {"counting": dict_count, "processing time": dict_runtime}

        path = os.path.join(os.getcwd(), "experiments", "count_track_line_cross_method" , f"ocsort_vid_stride_{vid_stride}.json")
        
        with open(path, 'w') as outfile:
            json.dump(overall_results, outfile)
        print(f"File ocsort_vid_stride_{vid_stride}.json is saved!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run counting tracking_with_line_crossing experiment  for specified strides!")
    parser.add_argument('--vid_strides', type=int, nargs='+', required=True, help='List of video strides to choose (e.g., 1 2)')
    t1 =time.time()
    args_parser = parser.parse_args()
    main(args_parser.vid_strides)
    print(f"Time required to run these experiments is: {(time.time()-t1)/60} minutes!")
