from counting.run_count import run
from counting.count import args


args.source = "kech.mp4"
args.name = "kech"
args.project = "runs/count"

args.vid_stride = 1
args.verbose = True

args.counting_approach = "tracking_with_two_lines"
args.line_point11 = (0.4, 0.0)
args.line_point12 = (0.4, 1.0)
args.line_vicinity = 0.05
args.line_point21 = (0.6, 0.0)
args.line_point22 = (0.7, 1.0)
args.use_mask = True


counter_yolo , profilers , results  = run(args)