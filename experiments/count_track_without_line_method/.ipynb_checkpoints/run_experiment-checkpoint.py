import os
import tqdm
import json
import argparse

import torch
import gc

from counting.run_count import run
from counting.count import args

def tensor_to_dict(count_per_class):
    # Dictionary keys for the selected vehicle types
    vehicle_types = ["bicycle", "car", "motorcycle", "bus", "truck"]
    
    # Indices corresponding to the vehicle types in the tensor
    indices = [1, 2, 3, 5, 7]
    
    # Create the dictionary
    vehicle_counts = {vehicle: count_per_class[idx].item() for vehicle, idx in zip(vehicle_types, indices)}
    
    return vehicle_counts

def main(trackers):
    
    args.counting_approach = "tracking_without_line"
    args.save = False
    args.verbose = False
    args.use_mask = False
    args.save_csv_count = False

    folder_path = os.path.join(os.getcwd(), "dataset")
    videos = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    video_names = [f"kech{idx}.mp4" for idx in range(2, len(videos) + 2)]
    strides = [1, 2, 3, 4]





    for tracker_name in trackers:
        
        print(f"tracker_name: {tracker_name}")
        args.tracking_method  = tracker_name
        overall_results = {}
        
        for video in video_names:
            
            print(f"video: {video}")
            args.source = os.path.join(os.getcwd(), "dataset", video)
            dict_count, dict_runtime = {}, {}
    
            for stride in strides:
            
                print(f"stride: {stride}")
                
                args.vid_stride = stride
                counter_yolo , profilers , _  = run(args)
    
                dict_count[f"stride_{stride}"] = {"count_per_class" : tensor_to_dict(counter_yolo.count_per_class) , "total_count" : int(counter_yolo.counter.item())}
                dict_runtime[f"stride_{stride}"] = {"Preprocessing" : profilers[0].t , "Detection" : profilers[1].t , "Postprocessing" : profilers[2].t , "Tracking" : profilers[3].t , "Counting" : profilers[4].t}
                
                del counter_yolo , profilers
                torch.cuda.empty_cache()
                gc.collect()
                
            overall_results[video] = {"counting": dict_count, "processing time": dict_runtime}
    
        path = os.path.join(os.getcwd(), "experiments" , "count_track_without_line_method" , f"{tracker_name}.json")
        with open(path, 'w') as outfile:
            json.dump(overall_results, outfile)

        print(f"File {tracker_name}.json is saved in path {path}!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run counting tracking_without_line experiment for a specified tracker")
    parser.add_argument('--trackers', type=str, nargs='+', required=True, help='List of the trackers (e.g., botsort, deepocsort, etc.)')
    
    args_parser = parser.parse_args()
    main(args_parser.trackers)