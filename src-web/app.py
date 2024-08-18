from flask import Flask, request, redirect, url_for, render_template, jsonify,send_from_directory
import os
import threading
import json
import time
import signal
import psutil
import cv2
from openai import OpenAI

client = OpenAI(api_key='find-from-oneNotes')
from werkzeug.utils import secure_filename
from pose_estimation import PoseEstimation, estimate_met, calculate_calories_burned, calculate_calories_burned_per_hour

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi'}

# Initialize the OpenAI API key

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

@app.route('/refresh_analysis', methods=['POST'])
def refresh_analysis():
    data = request.json
    gpt_analysis = analyze_athlete_data_with_gpt(data)
    return jsonify({"status": "success", "gpt_analysis": gpt_analysis})
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


    # Generate the analysis using GPT
    gpt_analysis = analyze_athlete_data_with_gpt(results)

    # Update results with GPT analysis
    results['gpt_analysis'] = gpt_analysis


    with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)


def analyze_athlete_data_with_gpt_test(results):
    return "### Athlete Performance Analysis\n\n**Overview:**\nBased on the provided statistics, the athlete demonstrates a low current performance level across various metrics, including speed, calories burned, and area covered. However, there are indications of potential capabilities with higher maximum and average metrics. Let\u2019s analyze the advantages and disadvantages of their performance and suggest strategies for improvement.\n\n**1. Speed Statistics:**\n   - **Current Speed:** 0.0 across all axes (forward, sideways, depth) suggests that the athlete is either stationary or very recently inactive.\n   - **Maximum Speed:** The athlete's potential speeds show some promise, particularly in sideways movement (0.1199 m/s) and overall (0.1217 m/s), which are useful for agility and quick changes in direction during play.\n   - **Average Speed:** The averages indicate that there is potential for movement but are considerably below optimal performance levels in a competitive environment.\n\n**Advantages:**\n- Despite the current speed being at zero, the maximum speeds indicate that the athlete has the potential for reasonable quickness, especially in lateral movement.\n- The swing count of 98 suggests a high level of engagement in activity, indicating either practice or competition situations, which is beneficial for skill development.\n- The athlete's average calories burned per hour (308.65 kcal) implies some ongoing exertion, indicating that they can increase this metric with strategic conditioning.\n\n**Disadvantages:**\n- Current speeds being zero indicates a lack of effective movement during the time period analyzed. This could be due to inactivity, injury, or inefficiency in transitioning from stationary to active states.\n- Low coverage area (9.0 m\u00b2) suggests limited movement space or lack of utilization of the available space, which impacts overall effectiveness in competitive play.\n- A high footwork count (208) with no current speed raises questions about the effectiveness of that footwork, possibly indicating that the athlete is not translating agility into applied speed.\n\n**2. Strategies for Improvement:**\n- **Conditioning and Speed Training:** Implement structured drills focused on increasing the current speeds, particularly emphasizing sideways and forward movements. Sprinting drills or agility ladder work can help boost quickness and transition speed.\n  \n- **Footwork Utilization:** Engage in dynamic footwork drills that emphasize not just movement quantity but the efficiency and application of footwork in context (e.g., spaced-out agility drills, shadowing drills).\n  \n- **Endurance Training:** While the average calories burned indicate some stamina, incorporating interval training"
def analyze_athlete_data_with_gpt(results):
    prompt = f"""
    Based on the following statistics of an athlete, provide a theoretical analysis of their performance:
    - Speed Statistics: {results['speeds']}
    - Calories Burned: {results['calories_burned']} kcal, Average Calories Burned per Hour: {results['calories_burned_per_hour']} kcal
    - Covered Area: {results['covered_area']} m²
    - Swing Count: {results['swing_count']}
    - Footwork: {results['step_count']}

    Analyze the athlete's advantages, disadvantages, and suggest strategies for improvement.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 可以使用 gpt-3.5-turbo 替代 gpt-4o-mini
    messages=[{"role": "system", "content": "You are an expert sports analyst."},
              {"role": "user", "content": prompt}],
    max_tokens=500)

    analysis = response.choices[0].message.content
    return analysis

def kill_previous_process(port=5000):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        for conn in proc.info['connections']:
            if conn.laddr.port == port:
                os.kill(proc.info['pid'], signal.SIGTERM)
                break


if __name__ == "__main__":
    #kill_previous_process()
    app.run(debug=True, port=5001)
