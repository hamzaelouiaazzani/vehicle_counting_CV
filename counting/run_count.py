import torch
import cv2

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT
from boxmot.utils.checks import TestRequirements

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from counting.count import counter_YOLO

from ultralytics.utils import LOGGER , ops , colorstr

from functools import partial
from pathlib import Path
import csv 
import time
import os
from datetime import datetime





def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    counter_yolo = counter_YOLO(args)
    # return counter_yolo
    
    results = counter_yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    counter_yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    counter_yolo.predictor.custom_args = args
    
    
    if counter_yolo.predictor.args.verbose:
        LOGGER.info('')

    # Setup model
    model = None
    if not counter_yolo.predictor.model:
        counter_yolo.predictor.setup_model(model)

    # Setup source every time predict is called
    source = args.source
    counter_yolo.predictor.setup_source(source if source is not None else counter_yolo.predictor.args.source)

    # Check if save_dir/ label file exists
    if counter_yolo.predictor.args.save or counter_yolo.predictor.args.save_txt or args.save_csv_count:
        (counter_yolo.predictor.save_dir / 'labels' if counter_yolo.predictor.args.save_txt else counter_yolo.predictor.save_dir).mkdir(parents=True, exist_ok=True)

    # Warmup model
    if not counter_yolo.predictor.done_warmup:
        counter_yolo.predictor.model.warmup(imgsz=(1 if counter_yolo.predictor.model.pt or counter_yolo.predictor.model.triton else counter_yolo.predictor.dataset.bs, 3, *counter_yolo.predictor.imgsz))
        counter_yolo.predictor.done_warmup = True

    counter_yolo.predictor.seen, counter_yolo.predictor.windows, counter_yolo.predictor.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    counter_yolo.predictor.run_callbacks('on_predict_start')
    

    if args.save_csv_count and args.counting_approach=="tracking_with_two_lines":

            
        first_frame_epoch = 569030400000
        csv_file_path = os.path.join(os.getcwd(), args.project, 'vehicle_counts.csv')
        
        # Delete the previous file if it exists
        if os.path.isfile(csv_file_path):
            os.remove(csv_file_path)
        
        # Create a new file and write the header
        fieldnames = ['time (ms)', 'frame', 'IDs', 'vehicle_type']
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    
    results = []    
    
    for batch in counter_yolo.predictor.dataset:
        counter_yolo.predictor.run_callbacks('on_predict_batch_start')
        counter_yolo.predictor.batch = batch
        path, im0s, vid_cap, s = batch
        
        counter_yolo.frame_number = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))

        im0s , im , profilers = counter_yolo.run_pipeline(im0s , path , profilers)


        n = len(im0s)

        for i in range(n):

            counter_yolo.predictor.seen += 1

            # Counting Phase
            with profilers[4]:
                counter_yolo.run_counting(i)

            result = counter_yolo.predictor.results[i]
            
            results.append(result)
            
            result.speed = {
                'preprocess': profilers[0].dt * 1E3 / n,
                'inference': profilers[1].dt * 1E3 / n,
                'postprocess': profilers[2].dt * 1E3 / n,
                'tracking': profilers[3].dt * 1E3 / n,
                'counting': profilers[4].dt * 1E3 / n}


            p, im0 = path[i], None if counter_yolo.predictor.source_type.tensor else im0s[i]
            p = Path(p)


            if counter_yolo.predictor.args.verbose or counter_yolo.predictor.args.save or counter_yolo.predictor.args.save_txt or counter_yolo.predictor.args.show:

                s += counter_yolo.write_results(i, counter_yolo.predictor.results, (p, im, im0))

            if counter_yolo.predictor.args.save or counter_yolo.predictor.args.save_txt:
                counter_yolo.predictor.results[i].save_dir = counter_yolo.predictor.save_dir.__str__()

            if counter_yolo.predictor.args.show and counter_yolo.predictor.plotted_img is not None:
                counter_yolo.predictor.show(p)

            if counter_yolo.predictor.args.save and counter_yolo.predictor.plotted_img is not None:
                counter_yolo.predictor.save_preds(vid_cap, i, str(counter_yolo.predictor.save_dir / p.name))

            


            if args.save_csv_count:

                current_time = (counter_yolo.frame_number/counter_yolo.video_attributes["frame_rate"])*1000 + first_frame_epoch

                with open(csv_file_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # Initialize writer inside the loop
                    for id, cls_index in zip(counter_yolo.ids_filtered, counter_yolo.cls_filtered):
                        writer.writerow({
                            'time (ms)': current_time,
                            'frame': counter_yolo.frame_number,
                            'IDs': int(id),
                            'vehicle_type': counter_yolo.counting_attributes["index_to_label"][str(cls_index.item())]
                        })


        counter_yolo.predictor.run_callbacks('on_predict_batch_end')


        # Print time (inference-only)
        if counter_yolo.predictor.args.verbose:
            LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        
    # Release assets
    if isinstance(counter_yolo.predictor.vid_writer[-1], cv2.VideoWriter):
        counter_yolo.predictor.vid_writer[-1].release()  # release final video writer

    # Print results

    if counter_yolo.predictor.args.verbose and counter_yolo.predictor.seen:
        t = tuple(x.t / counter_yolo.predictor.seen * 1E3 for x in profilers)  # speeds per image
        LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking, %.1fms counting per image at shape '
                    f'{(1, 3, *im.shape[2:])}' % t)
    if counter_yolo.predictor.args.save or counter_yolo.predictor.args.save_txt or counter_yolo.predictor.args.save_crop:
        nl = len(list(counter_yolo.predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {counter_yolo.predictor.save_dir / 'labels'}" if counter_yolo.predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', counter_yolo.predictor.save_dir)}{s}")


    return counter_yolo , profilers , results