import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from sklearn.linear_model import RANSACRegressor, LinearRegression
from math import hypot
NUM_KP = 6
CONF_THR = 0.5

def get_video_properties(capture):
    # fps of video
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"fps value missing or invalid: \t {fps:.3f}")
    print(f"[INFO] fps value: {fps:.3f}")

    # resolution of video
    res_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    res_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if res_w <= 0 or res_h <= 0:
        raise ValueError(f"resolution values missing or invalid: \t width{res_w} x height{res_h}")
    print(f"[INFO] resolution (w x h): {res_w} x {res_h}")

    # total frame count of video
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frame_count <= 0:
        raise ValueError(f"frame count value missing or invalid: \t {total_frame_count}")
    print(f"[INFO] total frame count: {total_frame_count}")
    
    return fps, res_w, res_h, total_frame_count


def get_ball_detection_coords(ball_detection_result):
    if ball_detection_result.boxes is None:
        return None
    
    # extract bounding boxes
    boxes = ball_detection_result.boxes

    # check for general detection
    if len(boxes) < 1:
        return None

    # get box with highest confidence score
    best_box = boxes[boxes.conf.argmax()]

    # return center coords
    cx, cy = best_box.xywh[0][0:2].tolist()
    return(cx, cy)
    

def get_keypoints_detection_coords(keypoint_detection_result):
    if keypoint_detection_result.keypoints is None:
        return None
    
    # extract keypoints
    keypoints = keypoint_detection_result.keypoints

    # check for general detection
    if keypoints.xy.shape[0] == 0:
        return None
    if keypoints.xy.shape[1] != NUM_KP:
        return None

    # get the keypoints' coordinates and confidence scores
    kps_xy = keypoints.xy[0].cpu().numpy()
    kps_conf = keypoints.conf[0].cpu().numpy()

    # only use coordinates with high confidence scores
    for c in kps_conf:
        if c < CONF_THR:
            return None

    # set up list with keypoints' coordinates
    keypoints_coords = []

    for x, y in kps_xy:
        keypoints_coords.append((float(x), float(y)))
        
    return keypoints_coords


def create_plot(coord_list, points_after_impact, inlier_mask, impact_coords, a, b, mean_keypoints_coords, serve_position_coords, peak_coords):
    # extract x- and y-coordinates of ball detections separately
    xs = [p[0] for p in coord_list]
    ys = [p[1] for p in coord_list]

    kp_xs = mean_keypoints_coords[:, 0]
    kp_ys = mean_keypoints_coords[:, 1]
    
    # x-coordinates after imapct
    xs_inlier = [p[0] for p, ok in zip(points_after_impact, inlier_mask) if ok]

    # Linie über x-Bereich der Punkte zeichnen
    x_line = np.linspace(min(xs_inlier), max(xs_inlier), 200)
    y_line = a * x_line + b

    # plot settings and labelling
    plt.figure(figsize=(12, 6))

    # all ball detections
    plt.scatter(xs, ys, s=12, color="red", label="Ballpositionen")
    # impact point
    plt.scatter(impact_coords[0], impact_coords[1], s=50, color="blue", marker="x", label="Treffpunkt")
    # flugbahn after impact als RANSAC gerade
    plt.plot(x_line, y_line, linewidth=2, zorder=1, label="RANSAC-Gerade")
    # serve position
    plt.scatter(serve_position_coords[0], serve_position_coords[1], s=70, color="orange", marker="x", linewidth=2, label="Aufschlagposition")
    # peak of ball toss
    plt.scatter(peak_coords[0], peak_coords[1], s=60, color="green", marker="x", label="höchster Punkt des Ballwurfs")
    # court keypoints
    plt.scatter(kp_xs, kp_ys, s = 60, color="black", label="Platz Keypoints")
    for i, (x, y) in enumerate(mean_keypoints_coords):
        plt.text(x - 20, y - 25, f"KP{i}", color="black", fontsize=10, weight="bold")
    
    skeleton = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]
    for i, j in skeleton:
        x_kp_connections = [mean_keypoints_coords[i][0], mean_keypoints_coords[j][0]]
        y_kp_connections = [mean_keypoints_coords[i][1], mean_keypoints_coords[j][1]]

        plt.plot(x_kp_connections, y_kp_connections, color="black", linewidth=2, alpha=0.75, zorder=1)
    
    plt.gca().invert_yaxis()
    plt.xlabel("X-Position (Pixel)")
    plt.ylabel("Y-Position (Pixel)")
    plt.title("Trajektorie des Tennisballs (Pixel-Koordinaten)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()
    #plt.savefig("trajectory.png", dpi=300)


def find_impact(coord_list, coord_and_frame_list, mean_keypoints_coords):
    ratio_threshold = 2.0

    # list for horizontal movement
    dx = []
    # list for vertical movement
    dy = []
    # list only for y-values
    ys = [] 

    # record movement from detection to detection
    for i in range(1, len(coord_list)):
        x1, y1 = coord_list[i-1]
        x2, y2 = coord_list[i]

        dx.append(x2 - x1)
        dy.append(y2 - y1)

    # find highest point of ball toss
    for toss_coord in coord_list:
        if toss_coord[0] > mean_keypoints_coords[0][0] and toss_coord[0] < (mean_keypoints_coords[2][0] * 1.2):
            ys.append(toss_coord[1])
        else:
            ys.append(np.inf)
    
    peak_index = int(np.argmin(ys))
    peak_coords = coord_list[peak_index]
    print(f"Highest point of ball toss: {coord_list[peak_index]}")

    # calculate all ratios (|dx| / |dy|)
    ratios = []
    for i in range(len(dx)):
        # no vertical movement
        if dy[i] == 0:
            ratio = np.inf
        elif dx[i] < 5:
            ratio = 0.1
        else:
            ratio = abs(dx[i]) / abs(dy[i])
        ratios.append(ratio)

    # impact will be after peak position
    for i in range(peak_index + 1, len(ratios)):
        if ratios[i] > ratio_threshold:
            # only balls in the correct x-range
            if coord_list[i][0] > mean_keypoints_coords[0][0] and coord_list[i][0] < (mean_keypoints_coords[2][0] * 1.25):
                print(f"Impact Frame: {coord_and_frame_list[i][1]}")
                print(f"Impact Position: {coord_list[i]}")

                return coord_and_frame_list[i][1], coord_list[i], i, peak_coords


def get_points_and_distances_after_impact(coord_list, impact_frame_index):
    # only points after impact
    points_after_impact = coord_list[impact_frame_index:]

    if len(points_after_impact) < 2:
        print("Too few points detected!")
        return points_after_impact, [], []

    distances = []
    cumulative_distances = [0.0]

    # distance calculation of consecutive points
    for i in range(1, len(points_after_impact)):
        x1, y1 = points_after_impact[i - 1]
        x2, y2 = points_after_impact[i]
        d = math.hypot(x2 - x1, y2 - y1)

        distances.append(d)
        cumulative_distances.append(cumulative_distances[-1] + d)

    return points_after_impact, distances, cumulative_distances


def fit_ransac_line(points_after_impact, res_h):

    if len(points_after_impact) < 4:
        print(f"Not enough points for reliable RANSAC: {len(points_after_impact)}")

    # X: matrix(n,1) only x-values, y: array(n,) y-values
    X = np.array([[p[0]] for p in points_after_impact], dtype=float)
    y = np.array([p[1] for p in points_after_impact], dtype=float)

    # calculate value for residual threshold
    residual_threshold = res_h * 0.02

    # RANSAC config
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=residual_threshold,
        min_samples=3,
        random_state=42
    )

    # execute fit
    ransac.fit(X, y)

    # extract slope a and y-intercept b
    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.intercept_)

    inlier_mask = ransac.inlier_mask_

    print(f"RANSAC: f(x) = {a:.6f} * x + {b:.3f}")
    print(f"Inlier: {sum(inlier_mask)}/{len(inlier_mask)}")

    return a, b, inlier_mask


def calc_frame_step(total_frame_count, max_frames):
    return int(total_frame_count / max_frames)


def calculate_mean_keypoints_coords(keypoints_detection_coords_list):
    # init empty list that saves coords for every keypoint seperately in a sublist
    sorted_keypoint_list = [[] for _ in range(NUM_KP)]

    # iterate over all frames and save keypoint coords accordingly in the sorted keypoint list
    for keypoints_per_frame in keypoints_detection_coords_list:
        for i, keypoint_coord in enumerate(keypoints_per_frame):
            sorted_keypoint_list[i].append(keypoint_coord)

    # set up 6x2 matrix filled with all 0 for the median and standard deviation
    median_keypoints = np.zeros((NUM_KP, 2), dtype=float)
    robust_std_keypoints = np.zeros((NUM_KP, 2), dtype=float)

    # calculate the median coord values and a robust standard deviation for every keypoint by calculating the median absolute deviation
    for i in range(NUM_KP):
        # check for enough samples
        if len(sorted_keypoint_list[i]) < 5:
            print(f"KP{i}: not enough keypoint coordinate values ({len(sorted_keypoint_list[i])}) for a meaningful calculation of the mean and standard deviation.")
            return None

        median, robust_std = calculate_mad(sorted_keypoint_list[i])

        median_keypoints[i] = median
        robust_std_keypoints[i] = robust_std

    # get an enhanced list of keypoint coords 
    enhanced_keypoint_list = remove_keypoint_outliers(median_keypoints, robust_std_keypoints, sorted_keypoint_list)

    # set up 6x2 matrix filled with all 0 for the enhanced mean and standard deviation
    mean_keypoints = np.zeros((NUM_KP, 2), dtype=float)

    # calculate the enhanced mean coord values and standard deviation for every keypoint
    for i in range(NUM_KP):
        if len(enhanced_keypoint_list[i]) < 5:
            print(f"KP{i}: not enough keypoint coordinate values ({len(enhanced_keypoint_list[i])}) for a meaningful calculation of the enhanced mean and standard deviation.")
            return None

        mean_keypoints[i] = np.mean(np.array(enhanced_keypoint_list[i], dtype=float), axis=0)

    print("Court keypoints: ")
    for i, kp in enumerate(mean_keypoints):
        print(f"KP{i}: {kp}")

    return mean_keypoints


def remove_keypoint_outliers(median_keypoints, robust_std_keypoints, sorted_keypoint_list):
    # init empty list that saves coords for every validated keypoint coords seperately in a sublist
    enhanced_keypoint_list = [[] for _ in range(NUM_KP)]

    # sigma factor for standard deviation and epsilon for protection against division by 0
    k = 3
    eps = 1e-6

    # only add validated coord values that are in range of 3x the robust standard deviation
    for i in range(NUM_KP):
        median_x, median_y = median_keypoints[i]
        std_x, std_y = robust_std_keypoints[i]

        # in case x or y value is 0, take eps
        std_x = max(std_x, eps)
        std_y = max(std_y, eps)
    
        for x, y in sorted_keypoint_list[i]:
            # standard deviation units distance from median per axis
            z_x = (x - median_x) / std_x
            z_y = (y - median_y) / std_y

            # check if within radius of k
            if (z_x**2 + z_y**2) <= k**2:
                enhanced_keypoint_list[i].append((x, y))

    return enhanced_keypoint_list


def calculate_mad(sorted_list_of_1_keypoint):
    # init a NumPy array with the coordinates of the current iteration's keypoint
    coords_per_keypoint = np.array(sorted_list_of_1_keypoint, dtype=float)

    # median coords for current keypoint
    median = np.median(coords_per_keypoint, axis=0)

    # absolute deviation from the median coords of the current keypoint
    abs_dev = np.abs(coords_per_keypoint - median)

    # median absolute deviation
    mad = np.median(abs_dev, axis=0)

    # translate mad to standard deviation by multiplying with factor 1.4826
    robust_std = 1.4826 * mad

    return median, robust_std


def compute_homography(mean_keypoints_coords):
    """ official court meter measurements like a view from above with origin at KP2 = baseline left corner
    KP0 = (8.23, 0)                 
    KP1 = (4.115, 0)                
    KP2 = (0, 0)                    
    KP3 = (0, 5.485)                
    KP4 = (4.115, 5.485)            
    KP5 = (8.23, 5.485)             
    """

    # set correct datatype
    image_points = np.array(mean_keypoints_coords, dtype=np.float32)

    # keypoints in meter coordinates
    court_points = np.array([
        [8.23, 0.0],
        [4.115, 0.0],
        [0.0, 0.0],
        [0.0, 5.485],
        [4.115, 5.485],
        [8.23, 5.485],
    ], dtype=np.float32)

    # calculate homography
    H, status = cv2.findHomography(image_points, court_points)

    if H is None:
        raise RuntimeError("Homography computation failed.")
    
    if status is not None:
        inliers = int(status.sum())
        if inliers < len(status):
            print(f"Warning: {len(status) - inliers} point(s) were rejected during homography estimation.")

    return H


def compute_local_scale_m_per_px(mean_keypoints_coords, distance_player_to_service_line_center_meter, serve_position_coords):
    # required keypoint coords
    KP1 = mean_keypoints_coords[1]  # baseline center
    KP4 = mean_keypoints_coords[4]  # service line center
    KP5 = mean_keypoints_coords[5]  # service line right side

    # serving from deuce side
    if KP4[1] < serve_position_coords[1]:
        print("Serve from Deuce side")
        x1, y1 = KP4
        x2, y2 = KP5

        # linear equation between KP4 and KP5
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        # determine x for given y
        schnittpunkt_foot_y_on_service_line = (serve_position_coords[1] - b) / a
        schnittpunkt_coords = np.array([schnittpunkt_foot_y_on_service_line, serve_position_coords[1]])

        px_dist = float(np.linalg.norm(schnittpunkt_coords - np.array(serve_position_coords)))
    
    # serving from ad side
    else:
        print("Serve from Advantage side")
        px_dist = float(np.linalg.norm(KP4 - KP1))
    
    
    if px_dist < 1e-6:
        raise ValueError("Pixel distance between KP4 and KP1 is too small.")
    
    meters_per_pixel = distance_player_to_service_line_center_meter / px_dist

    print(f"pixel distance: {px_dist}, meters_per_pixel: {meters_per_pixel}")

    return meters_per_pixel


def calculate_speed(meters_per_pixel, coord_and_frame_list_after_impact, fps, inlier_mask, mean_keypoints_coords, deuce_side):
    total_distance = 0
    total_duration = 0
    inlier_coord_and_frame_list = []
    protected_inlier_coord_and_frame_list = []


    if deuce_side:
        stop_point_x = (mean_keypoints_coords[4][0] + mean_keypoints_coords[5][0]) / 2
    else:
        stop_point_x = mean_keypoints_coords[5][0]


    if len(inlier_mask) != len(coord_and_frame_list_after_impact):
        raise ValueError("inlier_mask und coord_and_frame_list have to be of same length")
    
    # outlier protection
    for ix in range(len(inlier_mask)):
        if inlier_mask[ix] == True:
            inlier_coord_and_frame_list.append(coord_and_frame_list_after_impact[ix])

    # detections after flight protection
    for ii in range(len(inlier_coord_and_frame_list)):
        if inlier_coord_and_frame_list[ii][0][0] < stop_point_x:
            protected_inlier_coord_and_frame_list.append(inlier_coord_and_frame_list[ii])
        else:
            break

    for i in range(1, len(protected_inlier_coord_and_frame_list)):
        # distance
        p1 = np.array(protected_inlier_coord_and_frame_list[i][0])
        p2 = np.array(protected_inlier_coord_and_frame_list[i-1][0])
        distance_px = float(np.linalg.norm(p1 - p2))
        distance_m = distance_px * meters_per_pixel

        # time
        f1 = protected_inlier_coord_and_frame_list[i][1]
        f2 = protected_inlier_coord_and_frame_list[i-1][1]
        duration_frames = f1 - f2
        duration_seconds = duration_frames / fps

        # speed
        v_in_ms = distance_m / duration_seconds
        v_in_kmh = v_in_ms * 3.6

        total_distance += distance_m
        total_duration += duration_seconds
        
    print(f"\ncalculated speed total distance: {(total_distance/total_duration)*3.6}\n")
    return (total_distance/total_duration)*3.6


def get_player_ground_y_before_impact(video_path, model, impact_frame_index, mean_keypoints_coords):
    conf_threshold=0.6

    cap = cv2.VideoCapture(video_path)

    frame_idx = 1
    player_foot_ys = []

    while cap.isOpened():
        frame_read, frame = cap.read()
        if not frame_read:
            print("Person detection ended or error while reading video")
            break

        # only frames before impact
        if frame_idx >= impact_frame_index:
            break

        # YOLO inference only for class "person"
        result = model.predict(frame, conf=conf_threshold, classes=[0], verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            frame_idx += 1
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        # list of candidates per frame: (y2, dist_to_baseline)
        candidates = []

        for box in boxes:
            x1, y1, x2, y2 = box

            # only players on the focused court
            if y2 < mean_keypoints_coords[0][1] and y2 > mean_keypoints_coords[2][1]:
                # distance of foot position to baseline
                foot_position = np.array([(x1+x2) / 2, y2])

                distance = np.linalg.norm(mean_keypoints_coords[1] - foot_position)

                candidates.append((y2, distance))

        if candidates:
            # person with least distance to baseline
            best_y2 = min(candidates, key=lambda t: t[1])[0]
            player_foot_ys.append(best_y2)

        frame_idx += 1

    cap.release()

    if len(player_foot_ys) == 0:
        return None

    player_foot_position_y = float(np.mean(player_foot_ys))
    
    return player_foot_position_y


def get_serve_position_coords(mean_keypoints_coords, player_y):
    kp0 = mean_keypoints_coords[0]
    kp1 = mean_keypoints_coords[1]
    kp2 = mean_keypoints_coords[2]

    # player serving from deuce side
    if kp1[1] < player_y:
        deuce_side = True
        p_start = kp0
        p_end = kp1
    
    # player serving from ad side
    else:
        deuce_side = False
        p_start = kp2
        p_end = kp1

    x1, y1 = p_start
    x2, y2 = p_end

    # f(x) = a*x + b
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    # determine x for given y
    player_x = (player_y - b) / a

    print(f"Serve position: ({player_x}, {player_y})")
    return (player_x, player_y), deuce_side



def service_position_point_px_to_m(point_px, H):
    # formatting service position point
    pt = np.array([[point_px]], dtype=np.float32)  
    out = cv2.perspectiveTransform(pt, H)          
    x, y = out[0, 0]

    print(f"meter coords service point using homography: {x}, {y}")
    return (float(x), float(y))


def get_distance_player_to_service_line_center(service_point_meter_coords):
    distance_meter_kp1_to_service_point = np.abs(service_point_meter_coords[0] - 4.115)
    distance_kp1_to_kp4 = 5.485

    length_hypothenuse = hypot(distance_meter_kp1_to_service_point, distance_kp1_to_kp4)

    print(f"distance service point to service line center in metres: {length_hypothenuse}")
    return length_hypothenuse
