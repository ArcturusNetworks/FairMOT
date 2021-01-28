from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


# added by Thinula
import zmq
import flatbuffers
from datasets.dataset.fb_schemas.streamproc.models.fbs import Frame
from datasets.dataset.fb_schemas.streamproc.models.fbs import Mat
from datasets.dataset.fb_schemas.streamproc.models.fbs import Box
from datasets.dataset.fb_schemas.streamproc.models.fbs import Detection
from datasets.dataset.fb_schemas.streamproc.models.fbs import Detections


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def exec_frame(opt, executor):
    # Run centernet on single frame and send to outgoing stream
    # tracker = JDETracker(opt, frame_rate=frame_rate)
    print("initializing JDE Tracker...")
    # frame rate can be passed as an argument here, need to find out what it is
    # used for...default is 30
    tracker = JDETracker(opt)
    print("Complete.")
    
    # Initialize zmq router for sending outgoing data
    router_addr = b"tcp://*:" + bytearray(opt.out_port)
    streamId = b"SinkDetections" # hardcoded out stream id (temporary)

    outContext = zmq.Context()
    outContext.linger = 0
    outStream = outContext.socket(zmq.ROUTER)
    # outStream.bind("tcp://*:5558")
    print(f"Outgoing stream binding to {router_addr}...")
    outStream.bind(router_addr)

    while True:
        try:
            print("Outgoing stream waiting for connection request...")
            data = outStream.recv()
            buf = bytearray(data)
            print(f"Received: {buf}")
            if buf == b"Connect":
                outStream.send(streamId, zmq.SNDMORE)
                outStream.send(b"Accepted")
                print("Outgoing stream connection request accepted")
                break
        except Exception as e:
            if str(e) == "Resource temporarily unavailable":
                print("Receive timed out, reconnecting...")
            else:
                print(e)
                sys.exit(0)

    timer = Timer()
    results = []
    fps_counter = 0

    for i, (frame_id, frame_ts, imgGPU, imgCPU) in enumerate(executor):
        # for calculating fps over 20 frames
        if fps_counter % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(fps_counter, 1. / max(1e-5, timer.average_time)))

        # begin timer and run tracking
        timer.tic()
        
        blob = torch.from_numpy(imgGPU).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, imgCPU)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        
        # For encoding tracking data
        obj_vec = []
        num_tracks = 0
        builder = flatbuffers.Builder(0)
        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            conf = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
                
                # tlwh: (top left x, top left y, width, height)
                Box.BoxStart(builder)
                print("Box parameters: ", tlwh)
                print("Confidence: ", conf)
                print("Tracking ID: ", tid)
                # need to convert tlwh elements to ints to pass into Box
                xCoord, yCoord = int(tlwh[0]), int(tlwh[1])
                width, height = int(tlwh[2]) + 1, int(tlwh[3]) + 1
                Box.BoxAddX(builder, xCoord)
                Box.BoxAddY(builder, yCoord)
                Box.BoxAddWidth(builder, width)
                Box.BoxAddHeight(builder, height)
                box = Box.BoxEnd(builder)

                label = builder.CreateString(str("person"))
                Detection.DetectionStart(builder)
                Detection.DetectionAddId(builder, tid)
                Detection.DetectionAddConf(builder, conf)
                Detection.DetectionAddLabel(builder, label)
                Detection.DetectionAddBbox(builder, box)

                obj_vec.append(Detection.DetectionEnd(builder))
                num_tracks += 1
        
        # Build objects vector
        Detections.DetectionsStartDataVector(builder, num_tracks)
        for i in range(num_tracks):
            builder.PrependUOffsetTRelative(obj_vec[i])
        objects = builder.EndVector(num_tracks)

        timestamp = builder.CreateString(frame_ts)
        Detections.DetectionsStart(builder)
        Detections.DetectionsAddId(builder, frame_id)
        Detections.DetectionsAddTimestamp(builder, timestamp)
        Detections.DetectionsAddData(builder, objects) #fData)
        finalBuffer = Detections.DetectionsEnd(builder)
        builder.Finish(finalBuffer)
        buf = builder.Output()

        # Send data via outgoing zmq connection
        while True:
            try:
                print("Waiting to receive request from analytics filter")
                replyData = outStream.recv()
                reply = bytearray(replyData)

                if reply == b"Request" or reply == b"":
                    print("Request received, sending streamId...")
                    outStream.send(streamId, zmq.SNDMORE)
                    print("Sending tracking data...")
                    outStream.send(buf)
                    print("Sent tracking data to port 5558")
                    break
                elif reply == b"Connect":
                    print("Received a connect request, reconnecting")
                    outStream.send(streamId, zmq.SNDMORE)
                    outStream.send(b"Accepted")
                    print("Complete, accepted sent")
                else:
                    print(f"Received: {reply}")
            except Exception as e:
                if str(e) == "Resource temporarily unavailable":
                    print("Receive timed out, reconnecting...")
                else:
                    print(e)
                    sys.exit(0)

        timer.toc()

    return fps_counter, timer.average_time, timer.calls

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    print("Created JDETracker model")
    
    # Setup outgoing stream
    context = zmq.Context()
    context.linger = 0
    repSocket = context.socket(zmq.ROUTER)
    repSocket.bind("tcp://*:5558")
    print("Binding to the outgoing port at tcp://*:5558")

    # Hardcoded out stream id (temporary)
    streamId = b"SinkDetections"

    while True:
        try:
            print("Waiting for incoming stream connection...")
            data = repSocket.recv()
            buf = bytearray(data)
            print(f"Received: {buf}")
            if buf == b"Connect":
                print("Ready to connect, sending streamId")
                repSocket.send(streamId, zmq.SNDMORE)
                print("Sending accepted")
                repSocket.send(b"Accepted")
                print("Exiting while loop!")
                break
        except Exception as e:
            if str(e) == "Resource temporarily unavailable":
                print("Receive timed out, reconnecting...")
            else:
                print(e)
                sys.exit(0)

    timer = Timer()
    results = []
    frame_id = 0
    print("Executing CenterNet..")
    #for path, img, img0 in dataloader:
    for i, (path, fbData, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        # added by Thinula, this will happen when a frame is not read in from port 5555
        if img is None:
            frame_id += 1
            continue
        
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []

        
        # Decode flatbuffer data
        sp_frame_id = fbData.Id()
        timestamp = fbData.Timestamp().decode("utf-8")
        frame_data = fbData.Mat().DataAsNumpy().tobytes()
        
        obj_vec = []
        num_tracks = 0
        builder = flatbuffers.Builder(0) # Size will grow automatically if needed
        Frame.FrameStart(builder)

        builder.Bytes[builder.head : (builder.head + len(frame_data))] = frame_data
        fData = builder.EndVector(len(frame_data))

        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            conf = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
                
                # tlwh: (top left x, top left y, width, height)
                Box.BoxStart(builder)
                print("Box parameters: ", tlwh)
                print("Confidence: ", conf)
                print("Tracking ID: ", tid)
                # need to convert tlwh elements to ints to pass into Box
                xCoord, yCoord = int(tlwh[0]), int(tlwh[1])
                width, height = int(tlwh[2]) + 1, int(tlwh[3]) + 1
                Box.BoxAddX(builder, xCoord)
                Box.BoxAddY(builder, yCoord)
                Box.BoxAddWidth(builder, width)
                Box.BoxAddHeight(builder, height)
                box = Box.BoxEnd(builder)

                label = builder.CreateString(str("person"))
                Detection.DetectionStart(builder)
                Detection.DetectionAddId(builder, tid)
                Detection.DetectionAddConf(builder, conf)
                Detection.DetectionAddLabel(builder, label)
                Detection.DetectionAddBbox(builder, box)

                obj_vec.append(Detection.DetectionEnd(builder))
                num_tracks += 1

        
        # Build objects vector
        Detections.DetectionsStartDataVector(builder, num_tracks)
        for i in range(num_tracks):
            builder.PrependUOffsetTRelative(obj_vec[i])
        objects = builder.EndVector(num_tracks)

        timestamp_out = builder.CreateString(timestamp)
        Detections.DetectionsStart(builder)
        Detections.DetectionsAddId(builder, sp_frame_id)
        Detections.DetectionsAddTimestamp(builder, timestamp_out)
        Detections.DetectionsAddData(builder, objects) #fData)
        finalBuffer = Detections.DetectionsEnd(builder)
        builder.Finish(finalBuffer)
        buf = builder.Output()
        
        fbData = Detections.Detections.GetRootAsDetections(buf, 0)
        sp_frame_id = fbData.Id()
        print("Frame ID: ", sp_frame_id)

        numObjects = fbData.DataLength()
        print("Number of Objects: ", numObjects)
        for boxNum in range(numObjects):
            obj_data = fbData.Data(boxNum)
            bbox = obj_data.Bbox()
            x = bbox.X()
            y = bbox.Y()
            width = bbox.Width()
            height = bbox.Height()
            box_params = [x, y, width, height]
            print("Bounding box: ", box_params)
            print("Confidence: ", obj_data.Conf())
            print("Tracking ID: ", obj_data.Id())

        # Send data via outgoing zmq connection
        while True:
            try:
                print("Waiting to receive request from analytics filter")
                replyData = repSocket.recv()
                reply = bytearray(replyData)

                if reply == b"Request" or reply == b"":
                    print("Request received, sending streamId...")
                    repSocket.send(streamId, zmq.SNDMORE)
                    print("Sending tracking data...")
                    repSocket.send(buf)
                    print("Sent tracking data to port 5558")
                    break
                elif reply == b"Connect":
                    print("Received a connect request, reconnecting")
                    repSocket.send(streamId, zmq.SNDMORE)
                    repSocket.send(b"Accepted")
                    print("Complete, accepted sent")
                else:
                    print(f"Received: {reply}")
            except Exception as e:
                if str(e) == "Resource temporarily unavailable":
                    print("Receive timed out, reconnecting...")
                else:
                    print(e)
                    sys.exit(0)

        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        #seqs_str = '''MOT17-01-SDP
                      #MOT17-06-SDP
                      #MOT17-07-SDP
                      #MOT17-12-SDP
                      #'''
        #seqs_str = '''MOT17-01-SDP MOT17-07-SDP MOT17-12-SDP MOT17-14-SDP'''
        #seqs_str = '''MOT17-03-SDP'''
        #seqs_str = '''MOT17-06-SDP MOT17-08-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        #seqs_str = '''MOT17-02-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        #seqs_str = '''Venice-2'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
