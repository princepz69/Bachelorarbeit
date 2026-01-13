import cv2
from ultralytics import YOLO
import utils

ball_detection_model = YOLO("ba_ball_detection_model.pt")
keypoint_detection_model = YOLO("ba_keypoint_detection_model.pt")
person_detection_model = YOLO("yolo11n.pt")
video_to_analyze = "X:/BA Videos/OTC Halle/Messvideos_kurz/deuce15.mp4"
MAX_FRAMES = 20

# load video
capture = cv2.VideoCapture(video_to_analyze)
if not capture.isOpened():
    raise Exception(f"Unable to open video:  {video_to_analyze}")

# extract fps, resolution and the total frame count from video
fps, res_w, res_h, total_frame_count = utils.get_video_properties(capture)

# calculate frame step
frame_step = utils.calc_frame_step(total_frame_count, MAX_FRAMES)

# initialize empty coordinate lists for ball detections, keypoints detections and a frame counter
ball_detection_coord_list = []
ball_detection_coord_and_frame_list = []
keypoints_detection_coords_list = []
frame_counter = 0

while True:
    # go frame by frame
    frame_read, frame = capture.read()
    if not frame_read:
        print("Ball and keypoint detection ended or error while reading video")
        break

    # YOLO inference of current frame for ball detection
    ball_detection_result = ball_detection_model(frame, verbose = False)[0]

    # add center coordinates of the detected ball to the list of ball detection coordinates
    ball_detection_coords = utils.get_ball_detection_coords(ball_detection_result)
    if ball_detection_coords is not None:
        ball_detection_coord_and_frame_list.append([ball_detection_coords, frame_counter + 1])
        ball_detection_coord_list.append(ball_detection_coords)


    # YOLO inference of every frame_step-th frame for keypoint detection
    if frame_counter % frame_step == 0:
        keypoint_detection_result = keypoint_detection_model(frame, verbose = False)[0]
    
        # add coordinates of the detected keypoints to the list of keypoint detection coordinates
        keypoints_detection_coords = utils.get_keypoints_detection_coords(keypoint_detection_result)
        if keypoints_detection_coords is not None:
            keypoints_detection_coords_list.append(keypoints_detection_coords)

    frame_counter += 1

# get the mean of the keypoints' coordinates
mean_keypoints_coords = utils.calculate_mean_keypoints_coords(keypoints_detection_coords_list)

# build homography matrix
homography_matrix = utils.compute_homography(mean_keypoints_coords)


# identify the impact point
impact_frame, impact_coords, impact_frame_index, peak_coords = utils.find_impact(ball_detection_coord_list, ball_detection_coord_and_frame_list, mean_keypoints_coords)

if impact_frame is not None and impact_coords is not None:
    # get distances between detections after impact
    points_after_impact, distances, cum_dist = utils.get_points_and_distances_after_impact(ball_detection_coord_list, impact_frame_index)

    # RANSAC fit
    a, b, inlier_mask = utils.fit_ransac_line(points_after_impact, res_h)

    # find serve position on baseline
    player_foot_y_coord = utils.get_player_ground_y_before_impact(video_to_analyze, person_detection_model, impact_frame_index, mean_keypoints_coords)
    serve_position_coords, deuce_side = utils.get_serve_position_coords(mean_keypoints_coords, player_foot_y_coord)

    # meter coordiantes of serve_position_coords using homography
    serve_point_meter_coords = utils.service_position_point_px_to_m(serve_position_coords, homography_matrix)

    # calculate ground distance from serve position to service line center point
    distance_player_to_service_line_center_meter = utils.get_distance_player_to_service_line_center(serve_point_meter_coords)

    # local scale in m/px from Court-Keypoints
    meters_per_pixel = utils.compute_local_scale_m_per_px(mean_keypoints_coords, distance_player_to_service_line_center_meter, serve_position_coords)

    # calculate ball speed
    utils.calculate_speed(meters_per_pixel, ball_detection_coord_and_frame_list[impact_frame_index:], fps, inlier_mask, mean_keypoints_coords, deuce_side)
    
else:
    print("No impact detected! Analysis cannot be performed")



# plot for visualisation of all detections and measurements
utils.create_plot(ball_detection_coord_list, points_after_impact, inlier_mask, impact_coords, a, b, mean_keypoints_coords, serve_position_coords, peak_coords)

capture.release()
print("Finished.\n")