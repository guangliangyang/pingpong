from flask import Flask, request, redirect, url_for, render_template, jsonify,send_from_directory
import os
import threading
import json
import time
import signal
import psutil
import cv2
from werkzeug.utils import secure_filename
from pose_estimation import PoseEstimation, estimate_met, calculate_calories_burned, calculate_calories_burned_per_hour

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        unique_id = str(int(time.time()))  # 使用时间戳作为唯一标识符
        threading.Thread(target=process_video, args=(file_path, unique_id)).start()
        return jsonify({"status": "processing", "filename": filename, "unique_id": unique_id})
    return redirect(request.url)


@app.route('/progress_and_results/<unique_id>')
def progress_and_results(unique_id):
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        progress = results.get("progress", {})
        status = "completed" if progress.get("progress", 0) == 100 else "processing"
        return jsonify({"status": status, "progress": progress, "results": results})
    else:
        return jsonify(
            {"status": "processing", "progress": {"progress": 0, "elapsed_time": "00:00:00", "estimated_time_remaining": "00:00:00"}})

@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02}:{int(mins):02}:{int(secs):02}"


def get_video_duration(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        return total_frames / fps
    else:
        return 0

def process_video(file_path, unique_id):
    pose_estimation = PoseEstimation()
    cap = cv2.VideoCapture(file_path)
    total_exercise_duration_seconds = get_video_duration(cap)
    processed_frames = 0

    def update_progress():
        nonlocal processed_frames
        elapsed_time = time.time() - start_time
        progress = (processed_frames / total_frames) * 100
        estimated_time_remaining = (elapsed_time / processed_frames) * (
                    total_frames - processed_frames) if processed_frames > 0 else 0
        progress_data = {"progress": progress, "elapsed_time": elapsed_time,
                         "estimated_time_remaining": estimated_time_remaining}

        # 更新结果文件中的进度信息
        results["progress"] = {
            "progress": progress,
            "elapsed_time": format_time(elapsed_time),
            "estimated_time_remaining": format_time(estimated_time_remaining)
        }
        with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_results.json"), 'w') as f:
            json.dump(results, f, indent=4)

    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_exercise_duration_formatted = format_time(total_exercise_duration_seconds)

    results = {
        "speeds": {},
        "calories_burned": 0,
        "calories_burned_per_hour": 0,
        "intensity": "",
        "swing_count": 0,
        "step_count": 0,
        "highlight_ratios": {},
        "covered_area": 0,
        "match_counts": {},
        "templates": pose_estimation.templates,
        "total_exercise_duration": total_exercise_duration_formatted,
        "progress": {"progress": 0, "elapsed_time": "00:00:00", "estimated_time_remaining": "00:00:00"}
    }

    with pose_estimation.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                      model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose_estimation.process_video(frame, pose)
            processed_frames += 1

            if processed_frames % 100 == 0:  # 每处理100帧更新一次进度
                update_progress()

    cap.release()

    speeds = pose_estimation.speeds
    swing_count = sum(pose_estimation.template_match_counts["Arm"].values())
    step_count = sum(pose_estimation.template_match_counts["Footwork"].values())

    average_speed = speeds['overall']['avg']
    estimated_met = estimate_met(average_speed, step_count, swing_count)
    calories_burned = calculate_calories_burned(estimated_met, 70, total_exercise_duration_seconds / 60)
    calories_burned_per_hour, intensity = calculate_calories_burned_per_hour(calories_burned, total_exercise_duration_seconds / 60)

    highlight_ratios = {str(tuple(map(tuple, vertices))): 0 for vertices in pose_estimation.grid_rects}
    for cell_points_tuple in pose_estimation.highlight_counts:
        if str(cell_points_tuple) in highlight_ratios:
            highlight_ratios[str(cell_points_tuple)] = (pose_estimation.highlight_counts[cell_points_tuple] / sum(
                pose_estimation.highlight_counts.values())) * 100

    covered_area = pose_estimation.calculate_covered_area({eval(k): v for k, v in highlight_ratios.items()})

    match_counts = pose_estimation.template_match_counts
    results.update({
        "speeds": speeds,
        "calories_burned": calories_burned,
        "calories_burned_per_hour": calories_burned_per_hour,
        "intensity": intensity,
        "swing_count": swing_count,
        "step_count": step_count,
        "highlight_ratios": highlight_ratios,
        "covered_area": covered_area,
        "match_counts": match_counts,
        "progress": {
            "progress": 100,
            "elapsed_time": format_time(time.time() - start_time),
            "estimated_time_remaining": format_time(0)
        }
    })

    with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)




def kill_previous_process(port=5000):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        for conn in proc.info['connections']:
            if conn.laddr.port == port:
                os.kill(proc.info['pid'], signal.SIGTERM)
                break


if __name__ == "__main__":
    #kill_previous_process()
    app.run(debug=True, port=5000)
