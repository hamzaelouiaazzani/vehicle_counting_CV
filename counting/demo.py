import time

from counting.run_count import run
from counting.count import args


args.source = "kech.mp4"
args.name = "kech"
args.project = "runs/count"

args.vid_stride = 1
args.verbose = True

args.counting_approach = "tracking_with_line_vicinity"        # tracking_with_line_vicinity , tracking_with_line_crossing, tracking_with_line_crossing_vicinity, tracking_with_two_lines

args.line_point11 = (0.0, 0.25)
args.line_point12 = (1.0, 0.75)
args.line_vicinity = 1.5

#args.line_point21 = (0.6, 0.0)
#args.line_point22 = (0.7, 1.0)

args.use_mask = False

args.save = True

ti = time.time()
counter_yolo , profilers , results  = run(args)
tf = time.time()

print(f"The total time required to process the whole program is: {tf-ti}s.")

# Counting Results
print(f"The number of vehicles counted by the algorithm is: {counter_yolo.counter}")



def tensor_to_dict(count_per_class):
    # Dictionary keys for the selected vehicle types
    vehicle_types = ["bicycle", "car", "motorcycle", "bus", "truck"]

    # Indices corresponding to the vehicle types in the tensor
    indices = [1, 2, 3, 5, 7]

    # Create the dictionary
    vehicle_counts = {vehicle: int(count_per_class[idx].item()) for vehicle, idx in zip(vehicle_types, indices)}

    return vehicle_counts

print(f"The number of vehicles per type counted by the algorithm is: {tensor_to_dict(counter_yolo.count_per_class)}")


print(f"The time required for the PRE-PROCESSING step is: {profilers[0].t}s.")
print(f"The time required for the DETECTION (Inference) step is: {profilers[1].t}s.")
print(f"The time required for the POS-PROCESSING step is: {profilers[2].t}s")
print(f"The time required for the TRACKING step is: {profilers[3].t}s.")
print(f"The time required for the COUNTING step is: {profilers[4].t}s.")

print(f"The average time per frame required for the PRE-PROCESSING step is: {profilers[0].dt * 1000}ms.")
print(f"The average time per frame required for the DETECTION (Inference) step is: {profilers[1].dt * 1000}ms.")
print(f"The average time per frame required for the POS-PROCESSING step is: {profilers[2].dt * 1000}ms.")
print(f"The average time per frame required for the TRACKING step is: {profilers[3].dt * 1000}ms.")
print(f"The average time per frame required for the COUNTING step is: {profilers[4].dt * 1000}ms.")